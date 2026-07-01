from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.speculative.dspark_components.dspark_scheduler import (
        DSparkScheduleConfig,
    )

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_SCHEDULE_TOPK.get()


class ScheduleVerifyLensTopk:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        survival_probs: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk(
            survival_probs=survival_probs, budget=budget, cfg=cfg
        )

    @classmethod
    def triton(
        cls,
        *,
        survival_probs: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "ScheduleVerifyLensTopk.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_SCHEDULE_TOPK=torch until the triton kernel lands."
        )


def schedule_verify_lens_topk(
    *,
    survival_probs: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    # GPU-native sort (no per-element D2H). survival_probs is the CURRENT step's
    # confidence cumprod (lag 0, on the forward stream); budget is a host int (the
    # relay-fed K). Everything below runs device-side so the captured graph can
    # consume verify_lens with zero compute-stream sync. cfg validated once at
    # planner construction (immutable) -> no per-step re-validation here.
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
