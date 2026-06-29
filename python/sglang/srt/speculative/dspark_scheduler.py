from __future__ import annotations

import logging
from typing import Optional

import msgspec
import torch

from sglang.srt.speculative.dspark_sps_table import SpsCostTable

logger = logging.getLogger(__name__)


class DSparkScheduleConfig(msgspec.Struct):
    gamma: int
    min_verify_len: int = 0
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
    verify_lens = torch.clamp(verify_lens, min=cfg.min_verify_len, max=max_len)
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


def schedule_verify_lens_greedy(
    *,
    survival_probs: torch.Tensor,
    sps_table: SpsCostTable,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    cfg.validate()
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()

    candidate_window = survival_probs[:, cfg.min_verify_len : max_len].to(torch.float64)
    entries: list[tuple[float, int, int]] = []
    for request in range(num_requests):
        for offset in range(candidate_window.shape[1]):
            prob = float(candidate_window[request, offset])
            if prob >= cfg.survival_eps:
                entries.append((prob, cfg.min_verify_len + offset, request))

    entries.sort(key=lambda e: (-e[0], e[1], e[2]))

    selected_extra = [0] * num_requests
    tau_star = float(num_requests)
    best_theta = tau_star * sps_table.lookup(num_requests)
    for index, (prob, _position, request) in enumerate(entries, start=1):
        candidate_tau_star = tau_star + prob
        candidate_theta = candidate_tau_star * sps_table.lookup(num_requests + index)
        if candidate_theta >= best_theta:
            best_theta = candidate_theta
            tau_star = candidate_tau_star
            selected_extra[request] += 1
        else:
            break

    verify_lens = torch.tensor(
        [
            min(max(cfg.min_verify_len + extra, cfg.min_verify_len), max_len)
            for extra in selected_extra
        ],
        dtype=torch.int32,
        device=survival_probs.device,
    )
    return verify_lens


class ConfidencePrefixScheduler:
    def __init__(self, *, sps_table: SpsCostTable, cfg: DSparkScheduleConfig) -> None:
        cfg.validate()
        self.sps_table = sps_table
        self.cfg = cfg
        self.cached_budget: Optional[int] = None

    def update_budget_from_history(
        self, *, history_survival_probs: torch.Tensor
    ) -> int:
        self.cached_budget = compute_verify_token_budget(
            history_survival_probs=history_survival_probs,
            sps_table=self.sps_table,
            cfg=self.cfg,
        )
        return self.cached_budget

    def compute_verify_lens(self, *, survival_probs: torch.Tensor) -> torch.Tensor:
        num_requests, _gamma = survival_probs.shape
        if self.cached_budget is None:
            budget = num_requests * self.cfg.resolved_max_verify_len()
        else:
            budget = self.cached_budget
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival_probs,
            budget=budget,
            cfg=self.cfg,
        )
        verify_lens_64 = verify_lens.to(torch.int64)
        total_extra = int((verify_lens_64 - self.cfg.min_verify_len).sum().item())
        assert (
            total_extra <= budget
        ), f"DSpark verify-len budget violated: extra={total_extra} > budget={budget}"
        return verify_lens


def schedule_verify_lens(
    *,
    scheduler: Optional[ConfidencePrefixScheduler],
    survival_probs: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if scheduler is None or survival_probs is None:
        return None
    return scheduler.compute_verify_lens(survival_probs=survival_probs)
