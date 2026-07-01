from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    SpsCostTable,
    load_sps_table_from_path,
)
from sglang.srt.utils.async_probe import maybe_assert_async

# Sentinel for a host-carry row that was never written (or written for a different
# request); the budget planner's freshness guard rejects rows still stamped with it.
_CONFIDENCE_RELAY_UNSET_SEQ_LEN: int = -1


class DSparkScheduleConfig(msgspec.Struct):
    gamma: int
    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6

    def resolved_max_verify_len(self) -> int:
        # The full verify window is gamma+1 (anchor + all gamma drafts) =
        # num_draft_tokens, which is what static verifies. Capping at gamma would
        # verify only gamma-1 drafts (the last draft never checked) and drop accept
        # below static.
        return self.max_verify_len or (self.gamma + 1)

    def validate(self) -> None:
        max_len = self.resolved_max_verify_len()
        if self.gamma < 1:
            raise ValueError(f"DSpark gamma must be >= 1, got {self.gamma}.")
        if not (0 <= self.min_verify_len <= max_len <= self.gamma + 1):
            raise ValueError(
                "DSpark verify-len config must satisfy 0 <= min <= max <= gamma+1, "
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

    # Candidates span all gamma draft slots (cols 0..gamma-1) so the budget can reach
    # gamma (verify the full window); slicing [min_verify_len:max_len] gives gamma-1
    # slots and caps the budget one draft short.
    candidates = history_survival_probs[:, :max_len].flatten()
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
    # GPU-native sort (no per-element D2H). survival_probs is the CURRENT step's
    # confidence cumprod (lag 0, on the forward stream); budget is a host int (the
    # relay-fed K). Everything below runs device-side so the captured graph can
    # consume verify_lens with zero compute-stream sync.
    cfg.validate()
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device

    selected_extra = torch.zeros(num_requests, dtype=torch.int64, device=device)
    if budget > 0:
        # Window spans all gamma draft slots (cols 0..gamma-1) so selected_extra can
        # reach gamma -> verify_len reaches gamma+1 (full window, == static); slicing
        # [min_verify_len:max_len] gives gamma-1 slots and caps verify_len at gamma
        # (the last draft never verified).
        candidate_window = survival_probs[:, :max_len]
        num_candidates = candidate_window.numel()
        if num_candidates > 0:
            request_index = (
                torch.arange(num_requests, device=device)
                .view(num_requests, 1)
                .expand_as(candidate_window)
            )
            position_index = (
                torch.arange(candidate_window.shape[1], device=device)
                .view(1, candidate_window.shape[1])
                .expand_as(candidate_window)
            )
            valid = candidate_window >= cfg.survival_eps

            flat_prob = candidate_window.reshape(-1).to(torch.float64)
            flat_request = request_index.reshape(-1)
            flat_position = position_index.reshape(-1)
            flat_valid = valid.reshape(-1)

            order = _value_independent_descending_order(
                probs=flat_prob,
                positions=flat_position,
                requests=flat_request,
                valid=flat_valid,
            )

            # take = min(budget, num_candidates) -- both host ints, no D2H (the old
            # min(budget, num_valid) needed a .item() sync). Invalid candidates sort
            # to the tail of `order`, so scatter-adding their valid flag (0) rather
            # than a 1 reproduces "skip invalid" exactly: when budget <= num_valid
            # every chosen row is valid; when budget > num_valid the surplus rows are
            # invalid and contribute 0, leaving selected_extra == num_valid as before.
            take = min(int(budget), num_candidates)
            chosen = order[:take]
            chosen_requests = flat_request[chosen]
            chosen_valid = flat_valid[chosen].to(torch.int64)
            selected_extra.scatter_add_(0, chosen_requests, chosen_valid)

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
    # Device-native value-independent ordering: primary survival descending, with a
    # deterministic tie-break of position ascending then request ascending. Each
    # (position, request) pair is unique across the flattened candidate window, so
    # those two keys fully determine the order (the old host implementation's
    # original-index 4th key never activated). Implemented as an LSD radix of stable
    # argsorts (least-significant key first) so the result is identical to the old
    # `keys.sort()` order, but without the O(bs*gamma) per-element float()/int() D2H.
    # Invalid candidates get -inf survival -> +inf sort key -> ordered last; the
    # caller masks their selection via the valid flag.
    masked_prob = torch.where(valid, probs, torch.full_like(probs, float("-inf")))
    num_candidates = masked_prob.numel()
    order = torch.arange(num_candidates, device=probs.device)
    order = order[torch.argsort(requests[order], stable=True)]
    order = order[torch.argsort(positions[order], stable=True)]
    order = order[torch.argsort(-masked_prob[order], stable=True)]
    return order


class ConfidencePrefixScheduler:
    def __init__(self, *, sps_table: SpsCostTable, cfg: DSparkScheduleConfig) -> None:
        cfg.validate()
        self.sps_table = sps_table
        self.cfg = cfg

    def compute_verify_lens(
        self, *, two_steps_prior_k_survival: torch.Tensor, sort_survival: torch.Tensor
    ) -> torch.Tensor:
        budget = compute_verify_token_budget(
            history_survival_probs=two_steps_prior_k_survival,
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
        maybe_assert_async(
            (verify_lens_64 - effective_floor).sum() <= budget,
            f"DSpark verify-len budget violated (budget={budget})",
        )
        return verify_lens


class HostConfidenceBudgetPlanner:
    """Host-side verify-budget source (paper section 5.2 two-steps-prior barrier).

    Owns a per-request-row host carry that shifts the FutureMap relay's natural
    lag-1 confidence to the configured causal lag (default 2), applies the H1
    freshness guard, and runs the pure-CPU greedy. Everything is a host tensor, so
    the budget K is produced with zero D2H sync. Two feed paths share the carry +
    guard + greedy:

    - overlap: ``prepare_budget(resolved, req_pool_indices_cpu)`` consumes
      ``FutureMap.resolve_confidence_cpu`` in the scheduler prepare window.
    - non-overlap: ``compute_budget(...)`` is fed from a synchronous worker-side
      ``.cpu()`` (the async relay is absent without overlap).

    Losslessness never depends on the budget (guaranteed by the accept-cap in
    _cap_correct_len); the carry only affects scheduling quality and the lag.
    """

    def __init__(
        self,
        *,
        sps_table: SpsCostTable,
        cfg: DSparkScheduleConfig,
        model_runner,
        relay_lag_steps: int = 1,
    ) -> None:
        cfg.validate()
        self.sps_table = sps_table
        self.cfg = cfg
        # The carry buffer is sized from the req-pool, but req_to_token_pool is not
        # populated until after model-runner init (this planner is built during
        # scheduler/worker __init__), so the size is read lazily in _ensure_carry
        # (mirrors the former ConfidenceRelay.ensure_buffers).
        self._model_runner = model_runner
        # Total causal lag (>= 1 already yields the barrier; default 2 reproduces the
        # paper, tunable via env). The feed already supplies relay_lag_steps of lag (1
        # under the async overlap relay, 0 in the synchronous non-overlap fallback);
        # the host carry supplies the remainder.
        self.lag_steps = max(
            int(envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.get()), 1
        )
        self.carry_steps = max(self.lag_steps - int(relay_lag_steps), 0)
        self._carry_confidence: Optional[torch.Tensor] = None
        self._carry_seq_lens: Optional[torch.Tensor] = None
        self._carry_pos = 0

    def compute_budget(
        self,
        *,
        confidence: torch.Tensor,
        seq_lens_stamp: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> int:
        # confidence [bs, gamma], seq_lens_stamp [bs], prefix_lens [bs] -- all host,
        # the relay's lag-1 snapshot for this batch's rows. Shift to lag, guard,
        # greedy. req_pool_indices_cpu scatters/gathers the per-row carry.
        lagged_confidence, lagged_stamp = self._shift_to_lag(
            confidence=confidence,
            seq_lens_stamp=seq_lens_stamp,
            req_pool_indices_cpu=req_pool_indices_cpu,
        )
        survival = self._two_steps_prior_survival(
            lagged_confidence=lagged_confidence,
            lagged_stamp=lagged_stamp,
            prefix_lens=prefix_lens,
        )
        return compute_verify_token_budget(
            history_survival_probs=survival,
            sps_table=self.sps_table,
            cfg=self.cfg,
        )

    def _shift_to_lag(
        self,
        *,
        confidence: torch.Tensor,
        seq_lens_stamp: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # carry_steps == 0 (lag == 1): the relay's lag-1 value is used directly.
        # Otherwise read the carry slot written carry_steps steps ago for these
        # rows (= the lag-steps-prior confidence + its seq_len stamp, B2), then write
        # this step's relay value back. Rows idle for a cycle keep a stale carry whose
        # stamp the freshness guard rejects.
        if self.carry_steps == 0:
            return confidence, seq_lens_stamp
        self._ensure_carry(gamma=confidence.shape[-1])
        slot = self._carry_pos % self.carry_steps
        rows = req_pool_indices_cpu.to(torch.int64)
        lagged_confidence = self._carry_confidence[slot, rows].clone()
        lagged_stamp = self._carry_seq_lens[slot, rows].clone()
        self._carry_confidence[slot, rows] = confidence.to(torch.float32)
        self._carry_seq_lens[slot, rows] = seq_lens_stamp.to(torch.int64)
        self._carry_pos += 1
        return lagged_confidence, lagged_stamp

    def _two_steps_prior_survival(
        self,
        *,
        lagged_confidence: torch.Tensor,
        lagged_stamp: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> torch.Tensor:
        # cumprod of the lag-steps-prior confidence, with the H1 identity guard: a
        # row is the SAME request iff its carried stamp is present (>= 0), strictly
        # smaller than the current prefix (a live request commits >= 1 token/step),
        # and within lag_steps of growth (<= lag_steps * (gamma + 1)). Stale / new /
        # just-switched rows fall back to verify-all (survival = 1.0 everywhere).
        k_survival = torch.cumprod(lagged_confidence.to(torch.float32), dim=1)
        growth = prefix_lens.to(torch.int64) - lagged_stamp.to(torch.int64)
        max_growth = self.lag_steps * (self.cfg.gamma + 1)
        fresh = (
            (lagged_stamp.to(torch.int64) >= 0) & (growth >= 1) & (growth <= max_growth)
        ).view(-1, 1)
        return torch.where(fresh, k_survival, torch.ones_like(k_survival))

    def _ensure_carry(self, *, gamma: int) -> None:
        if self._carry_confidence is not None:
            return
        req_pool_size = int(self._model_runner.req_to_token_pool.req_to_token.shape[0])
        self._carry_confidence = torch.zeros(
            (self.carry_steps, req_pool_size, gamma), dtype=torch.float32
        )
        self._carry_seq_lens = torch.full(
            (self.carry_steps, req_pool_size),
            _CONFIDENCE_RELAY_UNSET_SEQ_LEN,
            dtype=torch.int64,
        )


def build_sps_cost_table(
    *,
    server_args: ServerArgs,
    verify_num_draft_tokens: int,
) -> SpsCostTable:
    # A real --speculative-dspark-sps-table-path loads the pre-profiled,
    # hardware-aware table; the literal "const" sentinel deliberately opts into
    # a flat constant-SPS table (budget = verify-all-up-to-gamma, zero
    # throughput gain); anything else unset raises. The expected scheduler-on
    # workflow is to build the table offline with
    # sglang.benchmark.dspark_sps_profiler and pass it via
    # --speculative-dspark-sps-table-path (see
    # docs/advanced_features/dspark_sps_table.md).
    #
    # The flat "const" table makes the hardware-aware scheduler a no-op:
    # lookup() returns a constant, so the verify-token budget degenerates to
    # verify-all and every request keeps verify_len == gamma+1 (resolved_max_verify_len
    # = gamma+1, so compact verifies the anchor plus all gamma drafts = the full
    # window, matching static; this is lossless -- _cap_correct_len caps accept and
    # the bonus is re-read from the target distribution). The
    # verify_lens >= 1 anchor contract (see DSparkScheduleConfig.min_verify_len
    # and schedule_verify_lens_topk's lower-bound clamp) MUST be in place before
    # any profiled table is supplied, because a non-flat table yields small K
    # and would otherwise drive verify_len to 0.
    sps_table_path = server_args.speculative_dspark_sps_table_path
    if not sps_table_path:
        raise ValueError(
            "DSpark ragged-verify scheduler is enabled (mode != static) but "
            "--speculative-dspark-sps-table-path was not supplied. Build a "
            "hardware-aware SPS cost table offline against a plain (non-speculative) "
            "server with `python -m sglang.benchmark.dspark_sps_profiler` and pass "
            "its JSON path (see docs/advanced_features/dspark_sps_table.md). To "
            "deliberately run with a flat constant-SPS table instead "
            "(verify-all-up-to-gamma, zero throughput gain), pass "
            "--speculative-dspark-sps-table-path=const."
        )
    if sps_table_path != "const":
        return load_sps_table_from_path(sps_table_path)
    max_batch_tokens = max(
        1,
        int(server_args.max_running_requests or 1) * verify_num_draft_tokens,
    )
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[1.0],
        max_batch_tokens=max_batch_tokens,
    )
