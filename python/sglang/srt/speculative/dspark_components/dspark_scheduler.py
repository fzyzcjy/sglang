from __future__ import annotations

import logging

import msgspec
import torch

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    SpsCostTable,
    load_sps_table_from_path,
)

logger = logging.getLogger(__name__)


class DSparkScheduleConfig(msgspec.Struct):
    gamma: int
    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6

    def resolved_max_verify_len(self) -> int:
        return self.max_verify_len or self.gamma

    def validate(self) -> None:
        max_len = self.resolved_max_verify_len()
        if self.gamma < 1:
            raise ValueError(f"DSpark gamma must be >= 1, got {self.gamma}.")
        if not (0 <= self.min_verify_len <= max_len <= self.gamma):
            raise ValueError(
                "DSpark verify-len config must satisfy 0 <= min <= max <= gamma, "
                f"got min={self.min_verify_len}, max={max_len}, gamma={self.gamma}."
            )
        if self.survival_eps < 0:
            raise ValueError(f"survival_eps must be >= 0, got {self.survival_eps}.")


def compute_verify_token_budget(
    *,
    history_survival_probs: torch.Tensor,
    sps_table: SpsCostTable,
    cfg: DSparkScheduleConfig,
) -> int:
    cfg.validate()
    num_requests = history_survival_probs.shape[0]
    max_len = cfg.resolved_max_verify_len()

    candidates = history_survival_probs[:, cfg.min_verify_len : max_len].flatten()
    candidates = candidates[candidates >= cfg.survival_eps].to(torch.float64)
    candidates_sorted = torch.sort(candidates, descending=True).values
    prefix_sum = torch.cumsum(candidates_sorted, dim=0)

    best_extra = 0
    best_theta = float("-inf")
    for extra in range(candidates_sorted.numel() + 1):
        top_sum = float(prefix_sum[extra - 1]) if extra > 0 else 0.0
        tau_star = num_requests + top_sum
        theta = tau_star * sps_table.lookup(num_requests + extra)
        if theta > best_theta:
            best_theta = theta
            best_extra = extra

    return best_extra


def schedule_verify_lens_topk(
    *,
    survival_probs: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    cfg.validate()
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device

    selected_extra = torch.zeros(num_requests, dtype=torch.int64, device=device)
    if budget > 0:
        candidate_window = survival_probs[:, cfg.min_verify_len : max_len]
        num_candidates = candidate_window.numel()
        if num_candidates > 0:
            request_index = (
                torch.arange(num_requests, device=device)
                .view(num_requests, 1)
                .expand_as(candidate_window)
            )
            position_index = (
                torch.arange(cfg.min_verify_len, max_len, device=device)
                .view(1, candidate_window.shape[1])
                .expand_as(candidate_window)
            )
            valid = candidate_window >= cfg.survival_eps

            flat_prob = candidate_window.flatten().to(torch.float64)
            flat_request = request_index.flatten()
            flat_position = position_index.flatten()
            flat_valid = valid.flatten()

            order = _value_independent_descending_order(
                probs=flat_prob,
                positions=flat_position,
                requests=flat_request,
                valid=flat_valid,
            )

            num_valid = int(flat_valid.sum().item())
            take = min(int(budget), num_valid)
            chosen = order[:take]
            chosen_requests = flat_request[chosen]
            selected_extra.scatter_add_(
                0,
                chosen_requests,
                torch.ones_like(chosen_requests),
            )

    min_len = torch.full(
        (num_requests,), cfg.min_verify_len, dtype=torch.int64, device=device
    )
    verify_lens = min_len + selected_extra
    # verify_lens counts tokens including the anchor (= 1 + ell_r), so it must be
    # >= 1 for every request: RaggedVerifyLayout rejects < 1 and _cap_correct_len
    # reads ell_r = verify_lens - 1. The lower bound is max(min_verify_len, 1) so
    # an explicit min_verify_len=0 still cannot produce an anchor-less request.
    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = torch.clamp(verify_lens, min=lower_bound, max=max_len)
    return verify_lens.to(torch.int32)


def _value_independent_descending_order(
    *,
    probs: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    masked_prob = torch.where(valid, probs, torch.full_like(probs, float("-inf")))
    num_candidates = masked_prob.numel()
    keys = [
        (-float(masked_prob[i]), int(positions[i]), int(requests[i]), i)
        for i in range(num_candidates)
    ]
    keys.sort()
    return torch.tensor([k[3] for k in keys], dtype=torch.int64, device=probs.device)


class ConfidencePrefixScheduler:
    def __init__(self, *, sps_table: SpsCostTable, cfg: DSparkScheduleConfig) -> None:
        cfg.validate()
        self.sps_table = sps_table
        self.cfg = cfg

    def compute_verify_lens(
        self, *, k_survival: torch.Tensor, sort_survival: torch.Tensor
    ) -> torch.Tensor:
        budget = compute_verify_token_budget(
            history_survival_probs=k_survival,
            sps_table=self.sps_table,
            cfg=self.cfg,
        )
        verify_lens = schedule_verify_lens_topk(
            survival_probs=sort_survival,
            budget=budget,
            cfg=self.cfg,
        )
        verify_lens_64 = verify_lens.to(torch.int64)
        # Measure admitted extra against the effective floor max(min_verify_len, 1)
        # so the anchor padding added by the lower-bound clamp is not miscounted as
        # budget overflow when an explicit min_verify_len=0 is clamped up to 1.
        effective_floor = max(self.cfg.min_verify_len, 1)
        total_extra = int((verify_lens_64 - effective_floor).sum().item())
        assert (
            total_extra <= budget
        ), f"DSpark verify-len budget violated: extra={total_extra} > budget={budget}"
        return verify_lens


def build_sps_cost_table(
    *,
    server_args: ServerArgs,
    verify_num_draft_tokens: int,
    tp_rank: int,
) -> SpsCostTable:
    # Load a pre-profiled table when a path is given, else a flat constant-SPS
    # table (budget = verify-all-up-to-max). The expected scheduler-on workflow
    # is to build the table offline with sglang.benchmark.dspark_sps_profiler
    # and pass it via --speculative-dspark-sps-table-path (see
    # docs/advanced_features/dspark_sps_table.md); flat is the inert fallback
    # for cap-accept, which has zero throughput gain.
    #
    # Until a profiled (non-flat) table ships the hardware-aware scheduler is a
    # no-op: lookup() returns a constant, so the verify-token budget degenerates
    # to verify-all and every request keeps verify_len == gamma (the scheduler's
    # resolved_max_verify_len caps at gamma, so compact verifies the anchor plus
    # up to gamma-1 drafts; this is lossless -- _cap_correct_len caps accept and
    # the bonus is re-read from the target distribution). The
    # verify_lens >= 1 anchor contract (see DSparkScheduleConfig.min_verify_len
    # and schedule_verify_lens_topk's lower-bound clamp) MUST be in place before
    # any profiled table is supplied, because a non-flat table yields small K
    # and would otherwise drive verify_len to 0.
    sps_table_path = server_args.speculative_dspark_sps_table_path
    if sps_table_path:
        return load_sps_table_from_path(sps_table_path)
    if tp_rank == 0:
        # Loud, once (per worker init on rank 0): the scheduler is enabled but
        # the verify budget silently degenerates to verify-all without a
        # profiled table. Warn rather than no-op silently so a misconfigured
        # scheduler-on run is visible. Pass --speculative-dspark-sps-table-path
        # to opt into the hardware-aware schedule.
        logger.warning(
            "DSpark ragged-verify scheduler is enabled but no "
            "--speculative-dspark-sps-table-path was supplied. Falling back to a "
            "flat constant-SPS table: the hardware-aware verify budget "
            "degenerates to verify-all-up-to-gamma and yields zero throughput "
            "gain. Profile a table offline with "
            "`python -m sglang.benchmark.dspark_sps_profiler` and pass it via "
            "--speculative-dspark-sps-table-path (see "
            "docs/advanced_features/dspark_sps_table.md)."
        )
    max_batch_tokens = max(
        1,
        int(server_args.max_running_requests or 1) * verify_num_draft_tokens,
    )
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[1.0],
        max_batch_tokens=max_batch_tokens,
    )
