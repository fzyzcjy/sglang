import pytest
import torch

from sglang.jit_kernel.kimi_k2_moe_fused_gate import kimi_k2_moe_fused_gate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")


def _reference(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
):
    """Torch reference for the kimi K2 fused gate (num_expert_group=1 noaux_tc)."""
    scores = torch.sigmoid(input.float())
    biased = scores + bias.float().unsqueeze(0)
    _, idx = torch.topk(biased, k=topk, dim=-1, sorted=True)
    weights = torch.gather(scores, 1, idx)
    if renormalize:
        s = weights.sum(dim=-1, keepdim=True)
        rcp = torch.where(s > 0, 1.0 / s, torch.ones_like(s))
        if apply_routed_scaling_factor_on_output:
            rcp = rcp * routed_scaling_factor
        weights = weights * rcp
    return weights, idx.to(torch.int32)


def _canonicalize(weights: torch.Tensor, indices: torch.Tensor):
    """Sort each row by expert id so order differences don't fail the compare."""
    order = torch.argsort(indices, dim=-1)
    return torch.gather(weights, 1, order), torch.gather(indices, 1, order)


@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("num_rows", [1, 17, 512, 513, 4096])  # cross small/large kernel
@pytest.mark.parametrize("topk", [1, 8])
@pytest.mark.parametrize("renormalize", [True, False])
def test_kimi_k2_moe_fused_gate_matches_reference(
    num_experts, num_rows, topk, renormalize
):
    """JIT kimi K2 fused gate matches the torch reference across small/large token paths."""
    torch.manual_seed(0)
    routed_scaling_factor = 2.5
    apply_scaling = True

    input = torch.randn(num_rows, num_experts, dtype=torch.float32, device="cuda")
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")

    out, idx = kimi_k2_moe_fused_gate(
        input,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_scaling,
    )
    ref_out, ref_idx = _reference(
        input, bias, topk, renormalize, routed_scaling_factor, apply_scaling
    )

    out, idx = _canonicalize(out, idx)
    ref_out, ref_idx = _canonicalize(ref_out, ref_idx)

    torch.testing.assert_close(idx, ref_idx)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


def test_kimi_k2_moe_fused_gate_unsupported_num_experts():
    """num_experts outside {256, 384} raises instead of silently mis-routing."""
    input = torch.randn(4, 128, dtype=torch.float32, device="cuda")
    bias = torch.randn(128, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception):
        kimi_k2_moe_fused_gate(input, bias, topk=4, renormalize=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
