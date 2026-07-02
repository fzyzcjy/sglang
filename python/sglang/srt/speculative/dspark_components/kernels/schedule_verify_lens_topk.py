from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

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
        confidence: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk(confidence=confidence, budget=budget, cfg=cfg)

    @classmethod
    def triton(
        cls,
        *,
        confidence: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk_triton(
            confidence=confidence, budget=budget, cfg=cfg
        )


def compute_sort_survival(confidence: torch.Tensor) -> torch.Tensor:
    # Row cumprod of the per-position confidence -> survival sort keys. Folded into
    # both impls (the planner no longer launches a separate float-cast + cumprod per
    # step); exposed standalone for the planner's env-gated debug log.
    return torch.cumprod(confidence.to(torch.float32), dim=1)


def schedule_verify_lens_topk(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    return schedule_verify_lens_topk_from_survival(
        survival_probs=compute_sort_survival(confidence), budget=budget, cfg=cfg
    )


def schedule_verify_lens_topk_from_survival(
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


@triton.jit
def _schedule_topk_prep_kernel(
    confidence_ptr,
    survival_ptr,
    selected_extra_ptr,
    gamma,
    cols,
    G_P2: tl.constexpr,
):
    # Per-row float32 cumprod of confidence into the candidate window buffer, plus
    # zeroing this row's selected_extra accumulator (folds the planner's separate
    # to(float32) + cumprod + zeros launches into one).
    row = tl.program_id(0)
    g = tl.arange(0, G_P2)
    conf = tl.load(
        confidence_ptr + row.to(tl.int64) * gamma + g, mask=g < gamma, other=1.0
    ).to(tl.float32)
    surv = tl.cumprod(conf, axis=0)
    tl.store(survival_ptr + row.to(tl.int64) * cols + g, surv, mask=g < cols)
    tl.store(selected_extra_ptr + row, 0)


@triton.jit
def _schedule_topk_finalize_kernel(
    selected_extra_ptr,
    out_ptr,
    min_verify_len,
    lower_bound,
    max_len,
    bs,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    extra = tl.load(selected_extra_ptr + offs, mask=mask, other=0).to(tl.int32)
    lens = min_verify_len + extra
    lens = tl.maximum(lens, lower_bound)
    lens = tl.minimum(lens, max_len)
    tl.store(out_ptr + offs, lens, mask=mask)


@triton.jit
def _schedule_topk_selected_extra_kernel(
    survival_ptr,
    selected_extra_ptr,
    budget,
    cols,
    n,
    survival_eps,
    BLOCK_C: tl.constexpr,
    BLOCK_CP: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < n
    r = c // cols
    p = c % cols
    sp = tl.load(survival_ptr + c, mask=cmask, other=0.0)
    valid_c = sp >= survival_eps
    mp = tl.where(valid_c, sp, float("-inf"))
    # rank[c] = #candidates ordered before c under (survival desc, position asc,
    # request asc); (position, request) is unique per candidate so the order is total.
    # selected iff valid and rank < budget -> equals torch's "chosen = order[:budget]"
    # then per-request scatter-add of the valid flag. O(n^2), n <= bs*gamma is tiny.
    rank = tl.zeros([BLOCK_C], dtype=tl.int32)
    for cp0 in range(0, n, BLOCK_CP):
        cp = cp0 + tl.arange(0, BLOCK_CP)
        cpmask = cp < n
        rp = cp // cols
        pp = cp % cols
        spp = tl.load(survival_ptr + cp, mask=cpmask, other=0.0)
        validp = spp >= survival_eps
        mpp = tl.where(validp, spp, float("-inf"))
        gt = mpp[None, :] > mp[:, None]
        eq = mpp[None, :] == mp[:, None]
        pos_lt = pp[None, :] < p[:, None]
        pos_eq = pp[None, :] == p[:, None]
        req_lt = rp[None, :] < r[:, None]
        before = gt | (eq & (pos_lt | (pos_eq & req_lt)))
        before = before & cpmask[None, :]
        rank += tl.sum(before.to(tl.int32), axis=1)
    selected = valid_c & (rank < budget)
    tl.atomic_add(selected_extra_ptr + r, selected.to(tl.int32), mask=cmask)


def schedule_verify_lens_topk_triton(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    cols = min(max_len, gamma)
    n = num_requests * cols

    selected_extra = torch.empty(num_requests, dtype=torch.int32, device=device)
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    _schedule_topk_prep_kernel[(num_requests,)](
        confidence.contiguous(),
        survival,
        selected_extra,
        gamma,
        cols,
        G_P2=triton.next_power_of_2(max(gamma, 1)),
    )
    if budget > 0 and n > 0:
        BLOCK_C = 64
        BLOCK_CP = 256
        grid = (triton.cdiv(n, BLOCK_C),)
        _schedule_topk_selected_extra_kernel[grid](
            survival,
            selected_extra,
            int(budget),
            cols,
            n,
            float(cfg.survival_eps),
            BLOCK_C=BLOCK_C,
            BLOCK_CP=BLOCK_CP,
        )

    verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    BLOCK = 256
    _schedule_topk_finalize_kernel[(triton.cdiv(num_requests, BLOCK),)](
        selected_extra,
        verify_lens,
        int(cfg.min_verify_len),
        max(cfg.min_verify_len, 1),
        int(max_len),
        num_requests,
        BLOCK=BLOCK,
    )
    return verify_lens
