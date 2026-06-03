import pytest
import torch

from sglang.jit_kernel.kimi_k2_moe_fused_gate import (
    kimi_k2_moe_fused_gate as jit_kimi_k2_moe_fused_gate,
)
from sglang.srt.layers.moe.topk import kimi_k2_biased_topk_impl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")

# (num_experts, topk, routed_scaling_factor)
_CONFIGS = [
    (384, 6, 2.872),  # Kimi K2
    (256, 8, 1.0),  # MiMo V2.5
]

# Cross both the small-token (<=512 rows) and large-token kernels.
_SEQ_LENGTHS = list(range(1, 10)) + [16, 64, 128, 512, 513, 1024, 4096, 16384]


@pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
@pytest.mark.parametrize("config", _CONFIGS, ids=["kimi384", "mimo256"])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_jit_kimi_matches_biased_topk_reference(
    seq_length, config, apply_routed_scaling_factor_on_output
):
    """JIT kimi K2 fused gate weights match the kimi_k2_biased_topk_impl reference."""
    num_experts, topk, routed_scaling_factor = config
    renormalize = True

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=torch.float32, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    output, indices = jit_kimi_k2_moe_fused_gate(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    ref_output, ref_indices = kimi_k2_biased_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # Weights drive the MoE output; compare them after sorting (selection order differs).
    assert torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-2,
        atol=1e-3,
    ), (
        f"Weight mismatch at seq_length {seq_length}, num_experts {num_experts}, "
        f"topk {topk}, apply_scaling {apply_routed_scaling_factor_on_output}"
    )


@pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
@pytest.mark.parametrize("config", _CONFIGS, ids=["kimi384", "mimo256"])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_jit_kimi_matches_aot_kernel(
    seq_length, config, apply_routed_scaling_factor_on_output
):
    """JIT kimi K2 kernel reproduces the AOT sgl_kernel output (device code is verbatim)."""
    aot = pytest.importorskip("sgl_kernel")
    aot_kimi_k2_moe_fused_gate = getattr(aot, "kimi_k2_moe_fused_gate", None)
    if aot_kimi_k2_moe_fused_gate is None:
        pytest.skip("sgl_kernel.kimi_k2_moe_fused_gate not available")

    num_experts, topk, routed_scaling_factor = config
    renormalize = True

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    jit_out, jit_idx = jit_kimi_k2_moe_fused_gate(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    try:
        aot_out, aot_idx = aot_kimi_k2_moe_fused_gate(
            tensor,
            bias,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )
    except RuntimeError as e:
        # Older AOT builds only support 384 experts; nothing to compare against then.
        pytest.skip(f"AOT kernel rejected this config: {e}")

    # Verbatim device kernels -> identical results. Sort to ignore selection order.
    torch.testing.assert_close(jit_out.sort(dim=-1)[0], aot_out.sort(dim=-1)[0])
    torch.testing.assert_close(jit_idx.sort(dim=-1)[0], aot_idx.sort(dim=-1)[0])


def test_jit_kimi_output_shapes_and_renorm():
    """Output shapes/dtypes are correct and renormalized weights sum to 1 per token."""
    num_experts, topk = 384, 6
    seq_length = 1024

    torch.manual_seed(42)
    tensor = torch.rand((seq_length, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    output, indices = jit_kimi_k2_moe_fused_gate(
        tensor, bias, topk=topk, renormalize=True
    )

    assert output.shape == (seq_length, topk)
    assert indices.shape == (seq_length, topk)
    assert output.dtype == torch.float32
    assert indices.dtype == torch.int32

    weight_sums = output.sum(dim=-1)
    assert torch.allclose(
        weight_sums, torch.ones_like(weight_sums), rtol=1e-3, atol=1e-4
    )


def test_jit_kimi_unsupported_num_experts():
    """num_experts outside {256, 384} raises instead of silently mis-routing."""
    tensor = torch.rand((4, 128), dtype=torch.float32, device="cuda")
    bias = torch.rand(128, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception):
        jit_kimi_k2_moe_fused_gate(tensor, bias, topk=4, renormalize=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
