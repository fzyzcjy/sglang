from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_kimi_k2_moe_fused_gate_module() -> Module:
    return load_jit(
        "kimi_k2_moe_fused_gate",
        cuda_files=["moe/kimi_k2_moe_fused_gate.cuh"],
        cuda_wrappers=[
            ("kimi_k2_moe_fused_gate", "KimiK2MoEFusedGateKernel::run"),
        ],
    )


def kimi_k2_moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float | None = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kimi K2 MoE fused gate (num_expert_group=1, DeepSeek noaux_tc routing).

    Supports num_experts in {256, 384} and topk <= 8. input and bias must both
    be float32 CUDA tensors. Returns (output_weights, expert_indices).
    """
    assert input.dtype == torch.float32, "input must be float32"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert input.ndim == 2, "input must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert input.size(1) == bias.size(0), "input and bias must have same num_experts"

    num_rows = input.size(0)
    device = input.device

    output = torch.empty(num_rows, topk, dtype=torch.float32, device=device)
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=device)

    module = _jit_kimi_k2_moe_fused_gate_module()
    module.kimi_k2_moe_fused_gate(
        input,
        bias,
        output,
        indices,
        topk,
        renormalize,
        float(routed_scaling_factor) if routed_scaling_factor is not None else 1.0,
        apply_routed_scaling_factor_on_output,
    )

    return output, indices
