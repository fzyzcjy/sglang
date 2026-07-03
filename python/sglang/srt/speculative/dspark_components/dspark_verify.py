from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.srt.speculative.dspark_components.dspark_info import VerifyWindow
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func


def local_verify_tier_num_tokens(
    *,
    bs: int,
    verify_token_budget: Optional[int],
    verify_num_draft_tokens: int,
    min_verify_len: int,
) -> int:
    # -1 is the "no budget on this rank" sentinel: any rank contributing it
    # forces the whole DP group onto the legacy pinned tier for the step, so
    # a rank whose confidence relay missed (racy copy_done) can never diverge
    # from ranks that resolved a budget. The floor term is bs * min_verify_len
    # (not bs): the top-k allocator starts every request at min_verify_len and
    # only the extra tokens above that count against the budget.
    if verify_token_budget is None:
        return -1
    floor_tokens = bs * max(min_verify_len, 1)
    return min(floor_tokens + verify_token_budget, bs * verify_num_draft_tokens)


def dp_global_verify_tier_num_tokens(
    *,
    global_tier_num_tokens: Optional[list[int]],
) -> Optional[int]:
    if global_tier_num_tokens is None:
        return None
    if any(tier_num_tokens < 0 for tier_num_tokens in global_tier_num_tokens):
        return None
    max_tier_num_tokens = max(global_tier_num_tokens, default=0)
    return max_tier_num_tokens if max_tier_num_tokens > 0 else None


def dp_tier_budget(
    *, dp_tier_num_tokens: int, tier_num_reqs: int, min_verify_len: int
) -> int:
    return max(dp_tier_num_tokens - tier_num_reqs * max(min_verify_len, 1), 0)


def idle_ragged_layout(
    *,
    tier_num_reqs: int,
    dp_tier_num_tokens: Optional[int],
    device: torch.device,
    verify_num_draft_tokens: int,
    model_runner,
) -> Optional[RaggedVerifyLayout]:
    if dp_tier_num_tokens is None:
        return uniform_ragged_layout(
            bs=tier_num_reqs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            ragged_verify_mode=RaggedVerifyMode.COMPACT,
            model_runner=model_runner,
        )
    if ragged_layout_exceeds_captured_grid(
        num_reqs=tier_num_reqs,
        verify_num_draft_tokens=verify_num_draft_tokens,
        model_runner=model_runner,
        tier_tokens_hint=dp_tier_num_tokens,
    ):
        return None
    # All rows are padding on an idle rank, so the lens only need to be a
    # legal geometry whose bucket lands on the SAME budget tier the busy
    # ranks select: verify-all lens would sum past the trimmed tier and trip
    # the total<=graph_num_tokens invariant, while one anchor token per slot
    # keeps the sum minimal and lets the floor pick the tier.
    verify_lens_cpu = [1] * tier_num_reqs
    grid = verify_layout_grid(
        verify_lens_cpu=verify_lens_cpu,
        ragged_verify_mode=RaggedVerifyMode.COMPACT,
        model_runner=model_runner,
    )
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens_cpu,
        device=device,
        grid=grid,
        graph_num_tokens_floor=dp_tier_num_tokens,
    )


def uniform_ragged_layout(
    *,
    bs: int,
    device: torch.device,
    verify_num_draft_tokens: int,
    ragged_verify_mode: RaggedVerifyMode,
    model_runner,
    tier_num_reqs: Optional[int] = None,
) -> Optional[RaggedVerifyLayout]:
    tier_num_reqs = bs if tier_num_reqs is None else tier_num_reqs
    if ragged_layout_exceeds_captured_grid(
        num_reqs=tier_num_reqs,
        verify_num_draft_tokens=verify_num_draft_tokens,
        model_runner=model_runner,
    ):
        return None
    verify_lens_cpu = [verify_num_draft_tokens] * bs
    grid = verify_layout_grid(
        verify_lens_cpu=verify_lens_cpu,
        ragged_verify_mode=ragged_verify_mode,
        model_runner=model_runner,
    )
    graph_num_tokens_floor = verify_layout_graph_num_tokens_floor(
        num_reqs=tier_num_reqs,
        ragged_verify_mode=ragged_verify_mode,
        verify_num_draft_tokens=verify_num_draft_tokens,
        model_runner=model_runner,
    )
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens_cpu,
        device=device,
        grid=grid,
        graph_num_tokens_floor=graph_num_tokens_floor,
    )


def verify_lens_broadcast_group(*, tp_size: int) -> tuple:
    if is_dp_attention_enabled():
        return get_attention_tp_group(), get_attention_tp_size()
    return get_tp_group(), tp_size


def verify_layout_grid(
    *,
    verify_lens_cpu: list[int],
    ragged_verify_mode: RaggedVerifyMode,
    model_runner,
) -> list[int]:
    total = sum(verify_lens_cpu)
    if ragged_verify_mode is not RaggedVerifyMode.COMPACT:
        return [total]
    capture_num_tokens = ragged_capture_num_tokens(model_runner=model_runner)
    if capture_num_tokens is None:
        return [total]
    return capture_num_tokens


def verify_layout_graph_num_tokens_floor(
    *,
    num_reqs: int,
    ragged_verify_mode: RaggedVerifyMode,
    verify_num_draft_tokens: int,
    model_runner,
    verify_token_budget: Optional[int] = None,
    min_verify_len: int = 1,
) -> int:
    if (
        ragged_verify_mode is not RaggedVerifyMode.COMPACT
        or ragged_capture_num_tokens(model_runner=model_runner) is None
    ):
        return 0
    if verify_token_budget is not None:
        # Budget-tiered floor: every request verifies at least min_verify_len
        # tokens and the top-k allocator hands out at most `budget` tokens
        # above that, so num_reqs * min_verify_len + budget upper-bounds the
        # packed total. Both terms live on the host, so the tier choice stays
        # sync-free (no D2H on verify_lens). The clamp pins the invariant that
        # a budget tier never exceeds the legacy pinned tier (the packed total
        # is also bounded by the full verify window).
        floor_tokens = num_reqs * max(min_verify_len, 1)
        return min(
            floor_tokens + verify_token_budget, num_reqs * verify_num_draft_tokens
        )
    return num_reqs * verify_num_draft_tokens


def ragged_capture_num_tokens(*, model_runner) -> Optional[list[int]]:
    runner = model_runner.decode_cuda_graph_runner
    if runner is None or not runner.ragged_verify_mode:
        return None
    return runner.capture_num_tokens


def ragged_capture_max_slots(*, model_runner) -> Optional[int]:
    runner = model_runner.decode_cuda_graph_runner
    if runner is None or not runner.ragged_verify_mode:
        return None
    return runner.max_bs


def ragged_layout_exceeds_captured_grid(
    *,
    num_reqs: int,
    verify_num_draft_tokens: int,
    model_runner,
    tier_tokens_hint: Optional[int] = None,
) -> bool:
    capture_num_tokens = ragged_capture_num_tokens(model_runner=model_runner)
    if capture_num_tokens is None:
        return False
    max_slots = ragged_capture_max_slots(model_runner=model_runner)
    if max_slots is not None and num_reqs > max_slots:
        return True
    tier_tokens = (
        tier_tokens_hint
        if tier_tokens_hint is not None
        else num_reqs * verify_num_draft_tokens
    )
    return tier_tokens > capture_num_tokens[-1]


def alloc_verify_window(
    *,
    batch: ScheduleBatch,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    block_pos_offsets: torch.Tensor,
    model_runner,
) -> VerifyWindow:
    prefix_lens = batch.seq_lens
    verify_w = verify_num_draft_tokens
    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets
    verify_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_w,
        batch_size=bs,
        draft_token_num=verify_w,
        device=device,
    )
    verify_cache_loc_2d = verify_cache_loc.view(bs, verify_w)
    return VerifyWindow(
        positions_2d=positions_2d,
        verify_cache_loc=verify_cache_loc,
        verify_cache_loc_2d=verify_cache_loc_2d,
    )


def apply_logits_adjustments_strided(
    *,
    next_token_logits: torch.Tensor,
    sampling_info,
    verify_num_draft_tokens: int,
) -> None:
    if sampling_info is None:
        return
    apply_dflash_verify_logits_adjustments(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=verify_num_draft_tokens,
    )
