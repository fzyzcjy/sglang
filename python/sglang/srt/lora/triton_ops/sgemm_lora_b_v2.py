"""Single-adapter specialized v2 of the LoRA-B expand-add GEMM.

The production decode shape is a skinny [S, R] x [R, N] expand with R=16 and N either
2048 (o_proj / shared_down) or 62080 (lm_head), added into base_output. The original
``_sgemm_lora_b_kernel`` uses BLOCK_N=256 -> grid (4, 8, 1) = only 32 CTAs for N=2048,
badly parallelism-starved on a 148-SM GB200.

This v2 specializes for the single-adapter batch and raises occupancy with a finer output
tile (more N programs), dropping the per-segment weight/rank/seg indirection. It keeps the
exact arithmetic of the original kernel -- same fp32 dot, same cast, same ``atomic_add``
into base_output -- so its output is BITWISE IDENTICAL to the old kernel (the atomic adds
are per-exclusive-tile, order-independent). That lets the guardrail assert bitwise
equality. The permutation gather is kept so a shuffled token order stays correct; the
input x may be the fp32 split-K LoRA-A accumulator, cast to the weight dtype on-load.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.lora.triton_ops.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _sgemm_lora_b_v2_kernel(
    x,
    weights,
    output,
    N,  # output_dim
    K,  # rank
    x_stride_0,
    x_stride_1,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    seg_len,
    sorted_token_ids,
    scalings,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    pid_s = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    if pid_s * BLOCK_S >= seg_len:
        return

    # Single adapter: weight slot 0, so scaling = scalings[0] (loaded device-side to stay
    # cuda-graph safe; a host .item() would sync during capture).
    scaling = tl.load(scalings)
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    s_physical = _resolve_token_positions(
        sorted_token_ids, 0, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = weights + (k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    n_mask = n_offset[None, :] < N
    output_mask = (s_offset[:, None] < seg_len) & n_mask
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )

    x_tile = tl.load(
        x_ptrs,
        mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K),
        other=0.0,
    )
    w_tile = tl.load(w_ptrs, mask=(k_offset[:, None] < K) & n_mask, other=0.0)
    partial_sum = tl.dot(x_tile.to(w_tile.dtype), w_tile) * scaling

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Exact same write as the old kernel (cast then atomic_add into base_output) so the
    # output is bitwise identical; atomic per-exclusive-tile is order-independent here.
    tl.atomic_add(
        output_ptr,
        partial_sum.to(output.dtype.element_ty),
        mask=output_mask,
        sem="relaxed",
    )


def _select_config_b(N: int) -> dict:
    """Finer N tiles for the small N=2048 expand (raise 32 -> ~128 CTAs); keep a wide
    tile for the huge lm_head N=62080 which already has thousands of programs."""
    if N >= 16384:
        return {"BLOCK_N": 256, "num_warps": 8, "num_stages": 3}
    return {"BLOCK_N": 64, "num_warps": 4, "num_stages": 3}


def sgemm_lora_b_v2_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor = None,
    config: dict | None = None,
) -> torch.Tensor:
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3
    assert weights.shape[0] == 1, "v2 specializes for the single-adapter batch"

    S = x.shape[0]
    N = weights.shape[-2]
    R = weights.shape[-1]
    assert x.shape[-1] == R

    cfg = config or _select_config_b(N)
    BLOCK_S = 16
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_R = triton.next_power_of_2(R)

    if base_output is None:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    grid = (triton.cdiv(batch_info.max_len, BLOCK_S), triton.cdiv(N, BLOCK_N))
    sorted_by_adapter = batch_info.permutation is not None
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _sgemm_lora_b_v2_kernel[grid](
        x,
        weights,
        output,
        N,
        R,
        x.stride(0),
        x.stride(1),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        int(batch_info.max_len),
        batch_info.permutation,
        batch_info.scalings,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_N,
        BLOCK_R,
        ENABLE_PDL=enable_pdl,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **pdl_kwargs,
    )
    return output
