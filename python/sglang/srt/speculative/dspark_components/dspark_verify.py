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
from sglang.srt.speculative.dspark_components.dspark_info import (
    RaggedVerifyWindow,
    VerifyWindow,
)
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


def build_ragged_verify_window(
    *,
    batch: ScheduleBatch,
    layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    # Compact (real-N) verify window, sync-free: sized to the host-known padded
    # graph_num_tokens (no verify_lens D2H; see compact_row_index). Request r contributes
    # its [anchor, s_0..s_{ell_r-1}] = verify_len_r tokens; cache slots are over-allocated
    # to the full gamma+1 block (plan section 6). Padding rows (valid False) get position
    # 0 / slot 0 -- the runner's tail-zero contract (no stale KV slot / vocab id read).
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens

    req_id, within, valid = compact_row_index(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    safe_req = req_id.clamp(max=bs - 1)  # sink req_id == bs on padding rows
    positions = torch.where(
        valid,
        prefix_lens.to(torch.int64)[safe_req] + within,
        torch.zeros_like(within),
    )
    # Cache locs are written compactly (same ordering as req_id/within) into a bs*stride
    # buffer; pad to graph_num_tokens and zero padding rows (also clears the
    # [real_total, bs*stride) torch.empty tail).
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    verify_cache_loc = torch.nn.functional.pad(
        real_cache_loc, (0, padded_total - real_cache_loc.shape[0])
    )
    verify_cache_loc = torch.where(
        valid, verify_cache_loc, torch.zeros_like(verify_cache_loc)
    )

    verify_ids = compact_verify_ids(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )

    # Host verify seq_lens is NOT built here (keeps the window sync-free); the
    # dense-backend host pre-add lives in the worker's _run_ragged_target_verify.
    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )


def compact_verify_ids(
    *,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    layout: RaggedVerifyLayout,
    device: str,
) -> torch.Tensor:
    # Pack [anchor, s_0..s_{ell_r-1}] per request into a compact graph_num_tokens-row
    # tensor; padding tail (valid False) is zeroed. anchor = draft_block_ids[:, 0].
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    bs = layout.verify_lens.shape[0]
    safe_req = req_id.clamp(max=bs - 1)  # sink req_id == bs on padding rows
    anchors = draft_block_ids[:, 0]
    # within==0 -> anchor; else draft_tokens[:, within-1] (clamp masked at 0).
    drafts = draft_tokens[safe_req, (within - 1).clamp_min(0)]
    verify_ids = torch.where(within == 0, anchors[safe_req], drafts)
    verify_ids = torch.where(valid, verify_ids, torch.zeros_like(verify_ids))
    return verify_ids.to(torch.int64)


def scatter_compact_to_strided(
    *,
    compact: torch.Tensor,
    layout: RaggedVerifyLayout,
    fill_value: float,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    # compact->strided scatter (infra section 12.E). compact is [graph_num_tokens, dim];
    # scatter to the [bs*(gamma+1), dim] strided layout accept/inject expect (request i
    # owns rows [i*stride, i*stride+verify_len_i); intra-request pad = fill_value, kept
    # out of commit by the accept cap -> lossless). The compact padding tail (valid False)
    # routes to a throwaway sink row that the [:bs*stride] return drops, so the output is
    # shape/semantics-identical to the old real-N scatter -- accept/inject stay unchanged.
    stride = verify_num_draft_tokens
    bs = layout.verify_lens.shape[0]
    dim = compact.shape[1]
    device = compact.device
    # Under DP attention the compact verify input is padded up to the dp_gather
    # buffer (global_num_tokens = bs * draft_token_num >= graph_num_tokens), so the
    # target returns trailing pad rows past graph_num_tokens. The scatter below
    # indexes exactly graph_num_tokens rows (compact_row_index padded_total), so
    # trim compact to graph_num_tokens: the extra pad tokens are causal-after the
    # real tokens and their output is discarded (lossless), and it is a no-op
    # without DP (compact is already graph_num_tokens rows).
    compact = compact[: layout.graph_num_tokens]
    strided = torch.full(
        (bs * stride + 1, dim), fill_value, dtype=compact.dtype, device=device
    )
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    sink = bs * stride
    strided_pos = torch.where(
        valid,
        req_id.clamp(max=bs - 1) * stride + within,
        torch.full_like(within, sink),
    )
    strided.index_copy_(0, strided_pos, compact)
    return strided[: bs * stride]


def compact_row_index(
    *,
    verify_lens: torch.Tensor,
    padded_total: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (req_id, within, valid) for each row of a FIXED padded_total-row compact buffer.
    #
    # padded_total is the host-known, bs-derived graph_num_tokens (>= the real total
    # sum(verify_lens)); the first real_total rows pack each request's window back to
    # back, rows [real_total, padded_total) are padding. The old design sized the
    # buffer to the exact host `total` via repeat_interleave(output_size=total), which
    # forced the caller to D2H verify_lens off the forward stream first (one
    # per-step cudaStreamSynchronize -- the very sync this removes). Here real_total is
    # a DEVICE scalar (cumsum[-1]) compared against a host-sized arange, so the row->req
    # map is built with NO compute-stream sync. Padding rows carry req_id == bs (a sink
    # id the callers route to a discarded slot) and within == 0.
    verify_lens = verify_lens.to(device=device, dtype=torch.int64)
    bs = int(verify_lens.numel())
    incl = torch.cumsum(verify_lens, dim=0)  # inclusive prefix sum, device [bs]
    start = incl - verify_lens  # exclusive per-request start
    real_total = incl[-1]  # DEVICE scalar; never read to host
    row = torch.arange(padded_total, device=device, dtype=torch.int64)
    valid = row < real_total  # device mask, no sync
    # searchsorted(incl, row, right=True) is the owning request; rows >= real_total map
    # to bs (past the last request) -> routed to the sink.
    req_id = torch.searchsorted(incl, row, right=True)
    req_id = torch.where(valid, req_id, torch.full_like(req_id, bs))
    within = torch.where(
        valid, row - start[req_id.clamp(max=bs - 1)], torch.zeros_like(row)
    )
    return req_id, within, valid


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
