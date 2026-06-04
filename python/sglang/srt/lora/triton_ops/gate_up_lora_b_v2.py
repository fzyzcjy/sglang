"""Single-adapter specialized v2 of the gate_up LoRA-B expand-add GEMM.

Packs the gate and up expand into one launch (axis-1 selects the slice), like the original
``_gate_up_lora_b_kernel``, but specializes for the uniform single-adapter decode batch:
drops the per-segment weight/rank/scaling indirection and raises occupancy with a finer
output tile. The production shape is tiny (output_dim=128, R=16) so the original grid is
only (8, 2, 1) = 16 CTAs; a smaller BLOCK_OUT lifts the CTA count toward the SM count.

Keeps load+add+store (the tile is exclusive within the launch) and the permutation gather.
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
def _gate_up_lora_b_v2_kernel(
    x,
    weights,
    output,
    K,  # rank
    output_dim,
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
    pid = tl.program_id(axis=0)
    gate_up_id = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(output_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    scaling = tl.load(scalings)
    n_start = gate_up_id * output_dim

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    s_physical = _resolve_token_positions(
        sorted_token_ids, 0, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = (
        x
        + gate_up_id * K * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    n_mask = n_offset[None, :] < output_dim
    x_tile = tl.load(
        x_ptrs,
        mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K),
        other=0.0,
    )
    w_tile = tl.load(w_ptrs, mask=(k_offset[:, None] < K) & n_mask, other=0.0)
    partial_sum = tl.dot(x_tile.to(w_tile.dtype), w_tile) * scaling

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    output_ptr = (
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & n_mask
    partial_sum += tl.load(output_ptr, mask=output_mask).to(tl.float32)
    tl.store(output_ptr, partial_sum.to(output.dtype.element_ty), mask=output_mask)


def gate_up_lora_b_v2_fwd(
    x: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_dim: int,
    base_output: torch.Tensor = None,
    config: dict | None = None,
) -> torch.Tensor:
    s = x.shape[0]
    r = gate_up_lora_b.shape[-1]
    assert x.shape[1] == 2 * r
    assert gate_up_lora_b.shape[0] == 1, "v2 specializes for the single-adapter batch"

    cfg = config or {"BLOCK_OUT": 32, "num_warps": 4, "num_stages": 3}
    BLOCK_S = 16
    BLOCK_OUT = cfg["BLOCK_OUT"]
    BLOCK_R = triton.next_power_of_2(r)

    if base_output is None:
        output = torch.zeros((s, 2 * output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(output_dim, BLOCK_OUT),
        2,
    )
    sorted_by_adapter = batch_info.permutation is not None
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _gate_up_lora_b_v2_kernel[grid](
        x,
        gate_up_lora_b,
        output,
        r,
        output_dim,
        x.stride(0),
        x.stride(1),
        gate_up_lora_b.stride(1),
        gate_up_lora_b.stride(2),
        output.stride(0),
        output.stride(1),
        int(batch_info.max_len),
        batch_info.permutation,
        batch_info.scalings,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_OUT,
        BLOCK_R,
        ENABLE_PDL=enable_pdl,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **pdl_kwargs,
    )
    return output
