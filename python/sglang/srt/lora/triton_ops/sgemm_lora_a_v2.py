"""Single-adapter specialized v2 of the LoRA-A shrink GEMM.

The production decode shape is a skinny [S, K] x [K, N] shrink with N = stack_num*rank
<= 64 and K up to 2048. The original ``_sgemm_lora_a_kernel`` launches grid
(num_s_tiles, bs) = (4, 1) for S=64 -- only ~4 CTAs on a 148-SM GB200, so it is badly
parallelism-starved and runs ~6-8 us despite < 0.1 us of actual memory traffic.

This v2 raises the CTA count with aggressive split-K along the long K dim: each of
SPLIT_K programs reduces a 1/SPLIT_K slice of K and atomic-adds its fp32 partial into
the output, giving grid (num_s_tiles, SPLIT_K) = up to ~128 CTAs. It specializes for the
uniform single-adapter batch (one merged segment, weight slot 0, rank known) but keeps
the permutation gather so a shuffled token order is still correct.

The fp32 output is consumed directly by the LoRA-B expand kernels, which cast x on-load
(``x_tile.to(w_tile.dtype)``), so the fp32 accumulator costs no extra cast kernel.
"""

import functools

import torch
import triton
import triton.language as tl

from sglang.srt.lora.triton_ops.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _sgemm_lora_a_v2_kernel(
    x,
    weights,
    output,
    N,  # stack_num * rank, <= BLOCK_N (covered in one N tile)
    K,  # input_dim
    x_stride_0,
    x_stride_1,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    seg_len,
    sorted_token_ids,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    pid_s = tl.program_id(axis=0)
    pid_sk = tl.program_id(axis=1)

    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N)
    k_offset = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    # Single adapter: merged segment starts at row 0, weight slot 0.
    s_physical = _resolve_token_positions(
        sorted_token_ids, 0, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = weights + (k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        # k_offset already carries the global column (pid_sk*BLOCK_K + arange); compare
        # against the shrinking remainder so the tail iteration masks out-of-range cols.
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < k_remaining),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < k_remaining) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * SPLIT_K * x_stride_1
        w_ptrs += BLOCK_K * SPLIT_K * w_stride_2

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    if SPLIT_K == 1:
        tl.store(output_ptr, partial_sum.to(output.dtype.element_ty), mask=output_mask)
    else:
        tl.atomic_add(
            output_ptr,
            partial_sum.to(output.dtype.element_ty),
            mask=output_mask,
            sem="relaxed",
        )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _select_config(K: int, num_s_tiles: int, num_sms: int) -> dict:
    """Pick (BLOCK_K, SPLIT_K, num_warps, num_stages) to roughly fill the SMs with
    split-K programs. SPLIT_K is capped by the K-tile count so no program is empty."""
    block_k = 64 if K >= 256 else triton.next_power_of_2(K)
    num_k_tiles = triton.cdiv(K, block_k)
    target = max(1, 2 * num_sms // max(num_s_tiles, 1))
    split_k = max(1, min(target, num_k_tiles))
    return {
        "BLOCK_K": block_k,
        "SPLIT_K": split_k,
        "num_warps": 2 if split_k <= 4 else 4,
        "num_stages": 3,
    }


def sgemm_lora_a_v2_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    stack_num: int = 1,
    out_alloc_stream=None,
    config: dict | None = None,
) -> torch.Tensor:
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3
    assert weights.shape[0] == 1, "v2 specializes for the single-adapter batch"

    S = x.shape[0]
    N = weights.shape[-2]
    K = weights.shape[-1]
    assert x.shape[-1] == K
    BLOCK_N = triton.next_power_of_2(N)
    assert N <= 64 and BLOCK_N <= 64, "v2 covers N (stack*rank) in one tile"

    BLOCK_S = 16
    num_s_tiles = triton.cdiv(batch_info.max_len, BLOCK_S)
    cfg = config or _select_config(K, num_s_tiles, _num_sms(x.device.index))
    split_k = cfg["SPLIT_K"]

    if split_k > 1:
        output = torch.zeros((S, N), device=x.device, dtype=torch.float32)
    elif out_alloc_stream is not None:
        with torch.cuda.stream(out_alloc_stream):
            output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    else:
        output = torch.empty((S, N), device=x.device, dtype=x.dtype)

    grid = (num_s_tiles, split_k)
    sorted_by_adapter = batch_info.permutation is not None
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _sgemm_lora_a_v2_kernel[grid](
        x,
        weights,
        output,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        int(batch_info.max_len),
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_N,
        cfg["BLOCK_K"],
        split_k,
        ENABLE_PDL=enable_pdl,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **pdl_kwargs,
    )
    return output
