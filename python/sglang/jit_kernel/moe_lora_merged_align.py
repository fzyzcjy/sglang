from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_lora_merged_align",
        *args,
        cuda_files=["lora/moe_lora_merged_align_kernel.cu"],
        cuda_wrappers=[
            ("moe_lora_merged_align", f"MoeLoraMergedAlignKernel<{args}>::run"),
        ],
    )


def supports_merged_align(virtual_num_experts: int) -> bool:
    """Commit-1 kernel only implements the (64, 1024] bucket-count branch.

    The bucket count is virtual_num_experts + 1 (the +1 sentinel bucket). Other
    regimes (small-batch <=64, v2 >1024) keep the old path."""
    num_buckets = virtual_num_experts + 1
    return 64 < num_buckets <= 1024


def moe_lora_merged_align(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    shared_outer: bool,
    max_loras: int,
    block_size: int,
    local_expert_offset: int = 0,
    local_num_experts: Optional[int] = None,
    do_skip: bool = True,
    compact: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Fused replacement for (_fused_virtual_topk_ids + _align_block_size) on the
    merged-virtual-expert LoRA path.

    Reads raw topk_ids + token_lora_mapping, computes the merged virtual id
    inline (mirrors _fused_virtual_topk_ids), and aligns to block_size.

    Returns (sorted_token_ids, expert_ids, num_tokens_post_padded,
    token_lora_mask, virtual_num_experts).
    """
    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1)
    if flat_topk_ids.dtype == torch.int64:
        flat_topk_ids = flat_topk_ids.to(torch.int32)
    M, top_k = topk_ids.shape
    numel = M * top_k

    num_experts_for_weight = 1 if shared_outer else num_experts
    virtual_num_experts = num_experts_for_weight * max_loras
    ep_local = (
        (not shared_outer)
        and (local_num_experts is not None)
        and (local_num_experts < num_experts_for_weight)
    )

    # compact: histogram over only the rank's local experts (dense LOCAL ids)
    # instead of the full global virtual space. Valid only for the single-adapter
    # EP per-expert path (safe_lora shift is 0; owned ids are a contiguous window
    # remappable by a single -offset). expert_ids is restored to global in-kernel.
    compact_eff = compact and ep_local and max_loras == 1 and not shared_outer
    bucket_experts = local_num_experts if compact_eff else virtual_num_experts

    # Allocation mirrors moe_align_block_size.py (the kernel uses a +1 sentinel
    # bucket, so the padded buffers are sized with bucket_experts + 1).
    if numel < bucket_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (bucket_experts + 1) * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    # No memset: the align kernel writes cumsum[0..num_buckets] before
    # count_and_sort reads it (same as moe_align_block_size.py's empty cumsum).
    cumsum_buffer = torch.empty(
        (bucket_experts + 2,), dtype=torch.int32, device=device
    )
    token_lora_mask = torch.empty((M,), dtype=torch.bool, device=device)

    module = _jit_module(flat_topk_ids.dtype)
    module.moe_lora_merged_align(
        flat_topk_ids,
        token_lora_mapping,
        token_lora_mask,
        bucket_experts + 1,  # bucket count (the +1 sentinel)
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,  # pad_sorted_token_ids
        top_k,
        num_experts_for_weight,
        local_expert_offset,
        local_num_experts if local_num_experts is not None else 0,
        ep_local,
        shared_outer,
        do_skip,
        compact_eff,
    )

    return (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        token_lora_mask,
        virtual_num_experts,
    )
