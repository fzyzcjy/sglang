from __future__ import annotations

from typing import Tuple

import msgspec
import torch

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
        raise NotImplementedError(
            "ComputeDsparkWindowGather.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_SWA_PAGE_INDICES=torch until the triton kernel lands."
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
        window_swa_locs: torch.Tensor,
        block_swa_locs: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_dspark_swa_page_indices(
            window_swa_locs=window_swa_locs,
            block_swa_locs=block_swa_locs,
            context_lens=context_lens,
            block_size=block_size,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )

    @classmethod
    def triton(
        cls,
        *,
        window_swa_locs: torch.Tensor,
        block_swa_locs: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "BuildDsparkSwaPageIndices.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_SWA_PAGE_INDICES=torch until the triton kernel lands."
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


def build_dspark_swa_page_indices(
    *,
    window_swa_locs: torch.Tensor,
    block_swa_locs: torch.Tensor,
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
    draft block (NON-CAUSAL, no triangular mask). ``window_swa_locs`` / ``block_swa_locs``
    are already in SWA space (the caller translates via ``translate_loc_from_full_to_swa``).

    Args:
        window_swa_locs: ``[bs, swa_window]`` int32 SWA slots of the committed window,
            most-recent-last, with positions before the request's start padded ``-1``.
        block_swa_locs: ``[bs, block_size]`` int32 SWA slots written by this forward for the
            gamma draft tokens (shared by every query row of the request).
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
    if window_swa_locs.ndim != 2 or window_swa_locs.shape[1] != swa_window:
        raise ValueError(
            "window_swa_locs must be [bs, swa_window]; "
            f"got shape={tuple(window_swa_locs.shape)} (swa_window={swa_window})."
        )
    if block_swa_locs.ndim != 2 or block_swa_locs.shape[1] != block_size:
        raise ValueError(
            "block_swa_locs must be [bs, block_size]; "
            f"got shape={tuple(block_swa_locs.shape)} (block_size={block_size})."
        )
    bs = window_swa_locs.shape[0]
    device = window_swa_locs.device

    window_swa_locs = window_swa_locs.to(torch.int32)
    block_swa_locs = block_swa_locs.to(torch.int32)
    context_lens = context_lens.to(device=device, dtype=torch.int32)

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
