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


def uniform_ragged_layout(
    *,
    bs: int,
    device: torch.device,
    verify_num_draft_tokens: int,
    ragged_verify_mode: RaggedVerifyMode,
    model_runner,
) -> Optional[RaggedVerifyLayout]:
    # The degenerate uniform layout (verify_lens = [gamma+1] * bs) that a
    # layout-less compact verify carries so it hits the same token-keyed graph
    # the real ragged batch does (C3). Geometry matches the static full block.
    # A batch too large for any captured tier gets no layout and falls to the
    # bs-keyed eager path instead of crashing round_up_grid.
    if ragged_layout_exceeds_captured_grid(
        num_reqs=bs,
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
        num_reqs=bs,
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
    # Cross-rank shape consistency: rank 0's verify_lens is broadcast to its
    # peers. Under DP-attention each attention-TP group owns a different request
    # shard, so the broadcast MUST stay inside get_attention_tp_group() (mirror
    # sampler.py) -- broadcasting across the full TP group would overwrite peer
    # shards' verify_lens and desync the now layout-/token-count-affecting
    # schedule (H2). Without DP-attention the TP group is correct. Returns the
    # group and its world size; size <= 1 means no broadcast is needed.
    if is_dp_attention_enabled():
        return get_attention_tp_group(), get_attention_tp_size()
    return get_tp_group(), tp_size


def verify_layout_grid(
    *,
    verify_lens_cpu: list[int],
    ragged_verify_mode: RaggedVerifyMode,
    model_runner,
) -> list[int]:
    # COMPACT aligns the grid to the decode runner's token buckets so
    # graph_num_tokens lands on a captured tier; else [total] suffices.
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
) -> int:
    # The token-keyed capture grid {b * num_draft : b in capture_bs} ties each
    # token tier to one capture_bs, whose graph captured exactly b request
    # slots. A batch whose real total rounds up to a tier with fewer slots than
    # num_reqs would not fit, so floor the bucket to this batch's bs-derived
    # full block. Zero (no floor) when no token-keyed graph exists, where the
    # bucket is the real total run eager.
    if (
        ragged_verify_mode is not RaggedVerifyMode.COMPACT
        or ragged_capture_num_tokens(model_runner=model_runner) is None
    ):
        return 0
    return num_reqs * verify_num_draft_tokens


def ragged_capture_num_tokens(*, model_runner) -> Optional[list[int]]:
    # The decode runner owns the token-keyed capture grid (Plan B). Read it so
    # graph_num_tokens = round_up_grid(total, capture_num_tokens) selects a
    # captured graph. Returns None when no token-keyed graph exists (e.g.
    # cuda-graph disabled or an eager runner without ragged capture), in which
    # case full runs eager on exactly `total`. The runner type is polymorphic
    # (graph runner vs eager runner), so ragged_verify_mode may be absent.
    runner = model_runner.decode_cuda_graph_runner
    if runner is None or not getattr(runner, "ragged_verify_mode", False):
        return None
    return runner.capture_num_tokens


def ragged_layout_exceeds_captured_grid(
    *,
    num_reqs: int,
    verify_num_draft_tokens: int,
    model_runner,
) -> bool:
    # The token-keyed capture grid tops out at capture_num_tokens[-1] ==
    # max_capture_bs * (gamma+1). graph_num_tokens is floored to this batch's
    # bs-derived full block (num_reqs * verify_num_draft_tokens), so a batch
    # with num_reqs > max_capture_bs would drive round_up_grid past its max
    # tier and raise in from_verify_lens -- BEFORE the runner's
    # _can_run_ragged_verify_graph eager-fallback gate can run. Skip the ragged
    # layout for such a batch so the verify falls through to the bs-keyed eager
    # path (which rejects bs > max_bs). Inert for num_reqs <= max_capture_bs.
    capture_num_tokens = ragged_capture_num_tokens(model_runner=model_runner)
    if capture_num_tokens is None:
        return False
    return num_reqs * verify_num_draft_tokens > capture_num_tokens[-1]


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
