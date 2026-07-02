from __future__ import annotations

from typing import Tuple

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import ceil_align

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_SWA_PAGE_INDICES.get()


class DsparkWindowGather(msgspec.Struct, frozen=True):
    num_q: int
    bs: int
    context_lens: torch.Tensor  # [bs] int32 = clamp(prefix_lens, max=swa_window)
    req_pool_indices_per_request: torch.Tensor  # [bs]
    offsets: torch.Tensor  # [bs, swa_window] int64, already clamp(min=0)
    invalid: torch.Tensor  # [bs, swa_window] bool, computed PRE-clamp


class ComputeDsparkWindowGather:
    @classmethod
    def execute(cls, *args, **kwargs) -> DsparkWindowGather:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        block_size: int,
        swa_window: int,
    ) -> DsparkWindowGather:
        return compute_dspark_window_gather(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            block_size=block_size,
            swa_window=swa_window,
        )

    @classmethod
    def triton(
        cls,
        *,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        block_size: int,
        swa_window: int,
    ) -> DsparkWindowGather:
        return compute_dspark_window_gather_triton(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            block_size=block_size,
            swa_window=swa_window,
        )


class BuildDsparkSwaPageIndices:
    @classmethod
    def execute(cls, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_per_request: torch.Tensor,
        offsets: torch.Tensor,
        invalid: torch.Tensor,
        out_loc: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_dspark_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_per_request=req_pool_indices_per_request,
            offsets=offsets,
            invalid=invalid,
            out_loc=out_loc,
            context_lens=context_lens,
            block_size=block_size,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_per_request: torch.Tensor,
        offsets: torch.Tensor,
        invalid: torch.Tensor,
        out_loc: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_dspark_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_per_request=req_pool_indices_per_request,
            offsets=offsets,
            out_loc=out_loc,
            context_lens=context_lens,
            block_size=block_size,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )


def compute_dspark_window_gather(
    *,
    seq_lens_casual: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    block_size: int,
    swa_window: int,
) -> DsparkWindowGather:
    """Pre-gather index arithmetic for the NON-CAUSAL DSpark draft-block SWA window.

    Pure tensor arithmetic that turns the uniform-gamma per-token causal lengths into the
    per-request committed-window geometry consumed by ``get_dspark_swa_page_indices``: the
    per-request first-token prefix length, its clamped context length, the request index,
    the window column ``offsets`` into ``req_to_token`` (most-recent-last) and the ``invalid``
    mask for pre-start positions. ``swa_window`` is the SWA window size (SWA_WINDOW), passed
    in so this kernel stays self-contained (mirroring ``build_dspark_swa_page_indices``).
    """
    seq_lens_casual = seq_lens_casual.to(torch.int32)
    num_q = seq_lens_casual.size(0)
    assert num_q % block_size == 0, (
        f"DSpark draft block forward must be uniform-gamma: num_q={num_q} not "
        f"divisible by block_size={block_size}."
    )
    bs = num_q // block_size
    device = seq_lens_casual.device

    # Per-request prefix length (committed context before the draft block) and the
    # request's first-token row index in the uniform layout.
    first_token = torch.arange(bs, device=device, dtype=torch.int64) * block_size
    prefix_lens = (seq_lens_casual[first_token] - 1).to(torch.int32)
    context_lens = torch.clamp(prefix_lens, max=swa_window).to(torch.int32)
    req_pool_indices_per_request = req_pool_indices_repeated[first_token]

    # Window slots: the request's prefix positions [prefix - W .. prefix - 1], most
    # recent last, future-of-block-start masked -1 (mirrors the reference window).
    offsets = (
        prefix_lens.to(torch.int64).unsqueeze(1)
        - swa_window
        + torch.arange(swa_window, device=device, dtype=torch.int64).unsqueeze(0)
    )
    # invalid MUST be computed BEFORE the clamp: a pre-start window position has a negative
    # absolute offset, but the clamp(min=0) below rewrites those negatives to 0 so the
    # caller's req_to_token gather stays in-bounds. Reordering to clamp-then-(offsets < 0)
    # would make invalid all-False and silently leak the request's token-0 slot into the
    # masked window prefix.
    invalid = offsets < 0
    offsets = offsets.clamp(min=0)

    return DsparkWindowGather(
        num_q=num_q,
        bs=bs,
        context_lens=context_lens,
        req_pool_indices_per_request=req_pool_indices_per_request,
        offsets=offsets,
        invalid=invalid,
    )


@triton.jit
def _window_gather_kernel(
    seq_lens_casual_ptr,
    req_pool_rep_ptr,
    context_lens_ptr,
    req_pool_out_ptr,
    offsets_ptr,
    invalid_ptr,
    block_size,
    swa_window,
    W_BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    ft = i * block_size
    prefix = tl.load(seq_lens_casual_ptr + ft).to(tl.int64) - 1
    tl.store(context_lens_ptr + i, tl.minimum(prefix, swa_window).to(tl.int32))
    tl.store(req_pool_out_ptr + i, tl.load(req_pool_rep_ptr + ft))
    col = tl.arange(0, W_BLOCK)
    cmask = col < swa_window
    off = prefix - swa_window + col
    tl.store(invalid_ptr + i * swa_window + col, off < 0, mask=cmask)
    tl.store(offsets_ptr + i * swa_window + col, tl.maximum(off, 0), mask=cmask)


def compute_dspark_window_gather_triton(
    *,
    seq_lens_casual: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    block_size: int,
    swa_window: int,
) -> DsparkWindowGather:
    seq_lens_casual = seq_lens_casual.to(torch.int32).contiguous()
    num_q = seq_lens_casual.size(0)
    assert num_q % block_size == 0, (
        f"DSpark draft block forward must be uniform-gamma: num_q={num_q} not "
        f"divisible by block_size={block_size}."
    )
    bs = num_q // block_size
    device = seq_lens_casual.device
    req_pool_indices_repeated = req_pool_indices_repeated.to(device=device).contiguous()
    context_lens = torch.empty(bs, dtype=torch.int32, device=device)
    req_pool_out = torch.empty(bs, dtype=req_pool_indices_repeated.dtype, device=device)
    offsets = torch.empty((bs, swa_window), dtype=torch.int64, device=device)
    invalid = torch.empty((bs, swa_window), dtype=torch.bool, device=device)
    W_BLOCK = triton.next_power_of_2(swa_window)
    _window_gather_kernel[(bs,)](
        seq_lens_casual,
        req_pool_indices_repeated,
        context_lens,
        req_pool_out,
        offsets,
        invalid,
        block_size,
        swa_window,
        W_BLOCK=W_BLOCK,
    )
    return DsparkWindowGather(
        num_q=num_q,
        bs=bs,
        context_lens=context_lens,
        req_pool_indices_per_request=req_pool_out,
        offsets=offsets,
        invalid=invalid,
    )


def build_dspark_swa_page_indices(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_per_request: torch.Tensor,
    offsets: torch.Tensor,
    invalid: torch.Tensor,
    out_loc: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    swa_window: int,
    page_index_aligned_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the NON-CAUSAL full-block paged SWA index layout for the DSpark draft block.

    Paged port of the reference ``get_dspark_topk_idxs`` (model.py:744):

        matrix = cat([arange(min(window, start_pos+1)), window + arange(block_size)])
                 .view(1, 1, -1).expand(bsz, block_size, -1)

    Every one of the ``block_size`` draft queries in a request attends the SAME set of SWA
    slots: the whole committed sliding window of injected target-hidden KV plus the whole
    draft block (NON-CAUSAL, no triangular mask). This builder fuses what used to be the
    caller's torch middle -- the ``req_to_token`` window gather, the full->SWA translate
    (a ``full_to_swa_mapping`` table lookup), and the block-slot translate -- so the whole
    ``get_dspark_swa_page_indices`` collapses to window_gather + this builder with no torch
    glue in between.

    Args:
        req_to_token: ``[max_reqs, max_ctx]`` the request->token-slot table (full space).
        full_to_swa_mapping: ``[n_full]`` the full-loc -> SWA-loc translation table
            (``translate_loc_from_full_to_swa(x) == full_to_swa_mapping[x]``).
        req_pool_indices_per_request: ``[bs]`` the request's pool index (row into
            ``req_to_token``).
        offsets: ``[bs, swa_window]`` int64 window columns into ``req_to_token`` (most-recent
            last, already ``clamp(min=0)``).
        invalid: ``[bs, swa_window]`` bool pre-start mask (True where the window column is
            before the request's start). Consumed only by the torch reference (the triton
            path skips invalid columns via the context-length bound).
        out_loc: ``[num_q]`` the full-space slots this forward writes for the gamma draft
            tokens (block slots), flat, ``block_size`` per request.
        context_lens: ``[bs]`` the number of valid committed window tokens per request
            (= min(swa_window, prefix_len)).
        block_size: gamma, the number of draft-block query rows / draft tokens.
        swa_window: SWA window size (SWA_WINDOW), passed in to keep the kernel self-contained.
        page_index_aligned_size: page-index width alignment (PAGE_INDEX_ALIGNED_SIZE).

    Returns:
        ``swa_page_indices`` ``[bs * block_size, K]`` int32 (K padded to a multiple of
        ``page_index_aligned_size``; padding ``-1``) and ``swa_topk_lengths``
        ``[bs * block_size]`` int32 (= context_lens + block_size, identical across the
        ``block_size`` rows of a request). The kernel attends only the first
        ``swa_topk_lengths[q]`` entries, so the ``-1`` padding is never read.
    """
    if offsets.ndim != 2 or offsets.shape[1] != swa_window:
        raise ValueError(
            "offsets must be [bs, swa_window]; "
            f"got shape={tuple(offsets.shape)} (swa_window={swa_window})."
        )
    bs = offsets.shape[0]
    device = offsets.device
    context_lens = context_lens.to(device=device, dtype=torch.int32)

    # Window gather + full->SWA translate (was the caller's torch middle). Invalid
    # (pre-start) columns gather req_to_token slot 0 then get masked to -1; they sit at the
    # leading columns and are never picked by the left-pack below (cl counts only valid).
    window_full_locs = req_to_token[
        req_pool_indices_per_request[:, None].to(torch.int64), offsets
    ]
    window_full_locs = window_full_locs.masked_fill(invalid, 0)
    window_swa_locs = full_to_swa_mapping[window_full_locs].to(torch.int32)
    window_swa_locs = window_swa_locs.masked_fill(invalid, -1)

    block_full_locs = out_loc[: bs * block_size].view(bs, block_size)
    block_swa_locs = full_to_swa_mapping[block_full_locs].to(torch.int32)

    # The widest row holds the full window (swa_window) + the whole block, aligned up.
    target_width = ceil_align(swa_window + block_size, page_index_aligned_size)

    swa_page_indices = _compact_dspark_window_then_block(
        window_swa_locs=window_swa_locs,
        block_swa_locs=block_swa_locs,
        context_lens=context_lens,
        target_width=target_width,
        block_size=block_size,
        swa_window=swa_window,
    )

    # Replicate the request's single shared row to all block_size query rows (non-causal:
    # every query sees the same window + whole block).
    swa_page_indices = (
        swa_page_indices.view(bs, 1, target_width)
        .expand(bs, block_size, target_width)
        .reshape(bs * block_size, target_width)
        .contiguous()
    )
    swa_topk_lengths = (
        (context_lens + block_size)
        .view(bs, 1)
        .expand(bs, block_size)
        .reshape(bs * block_size)
        .contiguous()
        .to(torch.int32)
    )
    return swa_page_indices, swa_topk_lengths


def _compact_dspark_window_then_block(
    *,
    window_swa_locs: torch.Tensor,
    block_swa_locs: torch.Tensor,
    context_lens: torch.Tensor,
    target_width: int,
    block_size: int,
    swa_window: int,
) -> torch.Tensor:
    """Left-pack each request's valid window slots, then its block slots, then ``-1``.

    ``window_swa_locs`` keeps invalid (pre-start) slots as ``-1`` at the front (the
    reference fills the most-recent-last window with padding at the start when
    ``start_pos + 1 < window``). The kernel reads a length-prefix run, so the valid window
    slots must be contiguous and immediately precede the block slots: this gathers the last
    ``context_lens`` window slots into packed columns ``[0, context_lens)``, places the
    block slots at ``[context_lens, context_lens + block_size)``, and ``-1``-pads the rest.
    """
    bs = window_swa_locs.shape[0]
    device = window_swa_locs.device
    out = torch.full((bs, target_width), -1, dtype=torch.int32, device=device)

    # Left-pack the valid window suffix via gather + where, NOT boolean-mask advanced
    # indexing: out[bool_mask] = src first calls nonzero(), a D2H host sync with a
    # data-dependent shape, which is illegal inside cuda-graph capture (this builder runs
    # in the recorded graph on the draft worker). gather/where are elementwise, static
    # shape, no sync. Semantics: out[r, j] = window_swa_locs[r, (W - context_len[r]) + j]
    # for j < context_len[r], else -1 (the valid window suffix shifted left to [0, cl)).
    j = torch.arange(swa_window, device=device, dtype=torch.int32).view(1, -1)
    shift = (swa_window - context_lens.view(-1, 1)).to(torch.int32)
    src_col = (shift + j).clamp_(min=0, max=swa_window - 1).to(torch.int64)
    gathered = torch.gather(window_swa_locs, dim=1, index=src_col)
    valid = j < context_lens.view(-1, 1)
    out[:, :swa_window] = torch.where(valid, gathered, -1)

    # Block slots: static-shape integer advanced indexing (no nonzero) is capture-safe.
    block_col = context_lens.view(-1, 1) + torch.arange(
        block_size, device=device, dtype=torch.int32
    ).view(1, -1)
    block_rows = torch.arange(bs, device=device).view(-1, 1).expand(-1, block_size)
    out[block_rows, block_col] = block_swa_locs
    return out


@triton.jit
def _swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    offsets_ptr,
    out_loc_ptr,
    context_lens_ptr,
    out_ptr,
    topk_ptr,
    rt_stride,
    swa_window,
    block_size,
    target_width,
    TW_BLOCK: tl.constexpr,
):
    q = tl.program_id(0)
    i = q // block_size
    cl = tl.load(context_lens_ptr + i)
    rp = tl.load(req_pool_ptr + i).to(tl.int64)
    k = tl.arange(0, TW_BLOCK)
    kmask = k < target_width
    # out[q, k] = window suffix (left-packed) for k < cl; block slot for
    # cl <= k < cl+block_size; else -1. Every one of the block_size query rows of a
    # request shares this same row (non-causal), so grid = bs*block_size writes the
    # replication directly (no expand). The two full->SWA gathers (req_to_token window
    # gather + out_loc block gather, each translated through full_to_swa) are fused here,
    # replacing the caller's torch middle. Invalid (pre-start) window columns need no mask:
    # left-packing reads only the valid suffix (k < cl), which is always in-window.
    in_window = k < cl
    src_col = tl.minimum(tl.maximum((swa_window - cl) + k, 0), swa_window - 1)
    wmask = kmask & in_window
    off = tl.load(offsets_ptr + i * swa_window + src_col, mask=wmask, other=0).to(
        tl.int64
    )
    win_full = tl.load(req_to_token_ptr + rp * rt_stride + off, mask=wmask, other=0).to(
        tl.int64
    )
    win_swa = tl.load(full_to_swa_ptr + win_full, mask=wmask, other=-1).to(tl.int32)

    in_block = (k >= cl) & (k < cl + block_size)
    bmask = kmask & in_block
    bcol = tl.maximum(k - cl, 0)
    blk_full = tl.load(out_loc_ptr + i * block_size + bcol, mask=bmask, other=0).to(
        tl.int64
    )
    blk_swa = tl.load(full_to_swa_ptr + blk_full, mask=bmask, other=-1).to(tl.int32)

    val = tl.where(in_window, win_swa, tl.where(in_block, blk_swa, -1))
    tl.store(out_ptr + q * target_width + k, val.to(tl.int32), mask=kmask)
    tl.store(topk_ptr + q, (cl + block_size).to(tl.int32))


def build_dspark_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_per_request: torch.Tensor,
    offsets: torch.Tensor,
    out_loc: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    swa_window: int,
    page_index_aligned_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets.ndim != 2 or offsets.shape[1] != swa_window:
        raise ValueError(
            "offsets must be [bs, swa_window]; "
            f"got shape={tuple(offsets.shape)} (swa_window={swa_window})."
        )
    bs = offsets.shape[0]
    device = offsets.device
    req_pool = req_pool_indices_per_request.to(device=device).contiguous()
    offsets = offsets.to(torch.int64).contiguous()
    out_loc = out_loc[: bs * block_size].contiguous()
    context_lens = context_lens.to(device=device, dtype=torch.int32).contiguous()
    rt_stride = req_to_token.stride(0)
    target_width = ceil_align(swa_window + block_size, page_index_aligned_size)
    n_q = bs * block_size
    swa_page_indices = torch.empty(
        (n_q, target_width), dtype=torch.int32, device=device
    )
    swa_topk_lengths = torch.empty(n_q, dtype=torch.int32, device=device)
    TW_BLOCK = triton.next_power_of_2(target_width)
    _swa_page_indices_kernel[(n_q,)](
        req_to_token,
        full_to_swa_mapping,
        req_pool,
        offsets,
        out_loc,
        context_lens,
        swa_page_indices,
        swa_topk_lengths,
        rt_stride,
        swa_window,
        block_size,
        target_width,
        TW_BLOCK=TW_BLOCK,
    )
    return swa_page_indices, swa_topk_lengths
