from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_align_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_align_block_size",
        *args,
        cuda_files=["moe/moe_align_kernel.cu"],
        cuda_wrappers=[
            ("moe_align_block_size", f"MoeAlignBlockSizeKernel<{args}>::run"),
        ],
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    module = _jit_moe_align_module(topk_ids.dtype)
    # [SHAPECAP] adhoc shape-capture print (committed for trace; reverted after the run).
    _seen = moe_align_block_size.__dict__.setdefault("_shapecap_seen", set())
    _sig = (
        tuple(topk_ids.shape),
        str(topk_ids.dtype),
        int(num_experts),
        int(block_size),
    )
    if _sig not in _seen:
        _seen.add(_sig)
        print(
            f"[SHAPECAP moe_align_py] topk_ids={tuple(topk_ids.shape)}/{topk_ids.dtype} "
            f"numel={topk_ids.numel()} num_experts={num_experts} block_size={block_size} "
            f"sorted_token_ids={tuple(sorted_token_ids.shape)} "
            f"expert_ids={tuple(expert_ids.shape)} "
            f"cumsum_buffer={tuple(cumsum_buffer.shape)} pad={pad_sorted_token_ids}",
            flush=True,
        )
    module.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )
