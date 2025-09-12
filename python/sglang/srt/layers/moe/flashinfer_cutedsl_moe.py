from typing import Any, Dict, Optional, Union

import torch
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
from sgl_kernel.gemm import (
    scaled_fp4_grouped_quant,
    silu_and_mul_scaled_fp4_grouped_quant,
)

from sglang.srt.debug_utils.dumper import dumper


def get_cute_dtype(input: torch.Tensor) -> str:
    if input.dtype == torch.bfloat16:
        return "bfloat16"
    elif input.dtype == torch.float16:
        return "float16"
    elif input.dtype == torch.float32:
        return "float32"
    else:
        raise ValueError(f"Unsupported cute dtype {input.dtype}")


def flashinfer_cutedsl_moe_masked(
    hidden_states: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    input_global_scale: torch.Tensor,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alpha,
    w2: torch.Tensor,
    a2_global_scale: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alpha,
    masked_m: torch.Tensor,
    layer_id: int,
    dispatch_output_bf16,
):
    """
    Perform masked Mixture-of-Experts computation with FlashInfer's CuteDSL
    kernels.

    Args:
        hidden_states: Either of the following case
            * torch.Tensor: [num_experts, m, k], bf16
            * tuple[torch.Tensor, torch.Tensor]: [num_experts, m, k // 2], uint8, [num_experts, m, k // 16], float8_e4m3fn
        input_global_scale (torch.Tensor): (l,)
        w1 (torch.Tensor): fp4 weights, [l, 2 * n, k // 2], uint8
        w1_blockscale (torch.Tensor): blockscale factors, e4m3,
        w1_alpha (torch.Tensor): (l,)
        w2 (torch.Tensor): fp4 weights, [l, k, n // 2], uint8
        a2_global_scale (torch.Tensor): (l,)
        w2_blockscale (torch.Tensor): blockscale factors, e4m3,
        w2_alpha (torch.Tensor): (l,)
        masked_m (torch.Tensor): Masked dimension indices

    Notes:
        - Assumes max(masked_m) == m.
    """

    # dumper.dump("moe__hidden_states_a", hidden_states[0], layer_id=layer_id)
    # dumper.dump("moe__hidden_states_b", hidden_states[1], layer_id=layer_id)
    # dumper.dump("moe__masked_m", masked_m, layer_id=layer_id)

    # === Assertions on dtypes ===
    assert w1.dtype == torch.uint8, f"w1 must be uint8 (fp4 packed), got {w1.dtype}"
    assert (
        w1_blockscale.dtype == torch.float8_e4m3fn
    ), f"w1_blockscale must be float8_e4m3fn, got {w1_blockscale.dtype}"
    assert (
        w1_alpha.dtype == torch.float32
    ), f"w1_alpha must be float32, got {w1_alpha.dtype}"
    assert w2.dtype == torch.uint8, f"w2 must be uint8 (fp4 packed), got {w2.dtype}"
    assert (
        a2_global_scale.dtype == torch.float32
    ), f"a2_global_scale must be float32, got {a2_global_scale.dtype}"
    assert (
        w2_blockscale.dtype == torch.float8_e4m3fn
    ), f"w2_blockscale must be float8_e4m3fn, got {w2_blockscale.dtype}"
    assert (
        w2_alpha.dtype == torch.float32
    ), f"w2_alpha must be float32, got {w2_alpha.dtype}"

    # === Assertions on shapes ===
    n = w2.shape[-1] * 2  # intermediate dimension

    # def get_tensor_info(x):
    #     min = x.float().min() if x.numel() > 0 else None
    #     max = x.float().max() if x.numel() > 0 else None
    #     mean = x.float().mean() if x.numel() > 0 else None
    #     return f"shape={x.shape} dtype={x.dtype} device={x.device} stride={x.stride()} min={min} max={max} mean={mean}"
    # print(
    #     f"[{torch.distributed.get_rank()}, {layer_id=}] hi call moe "
    #     f"{get_tensor_info(hidden_states[0])=} "
    #     f"{get_tensor_info(hidden_states[1])=} "
    # )

    if isinstance(hidden_states, tuple):
        # temp skip this check
        # assert input_global_scale is None, "input_global_scale is needed when input needs quant"

        a_q = hidden_states[0].view(torch.uint8)
        a_q_sf = hidden_states[1].view(torch.float8_e4m3fn)
        m, k_by_2, num_experts = a_q.shape
        k = k_by_2 * 2
    else:
        num_experts, m, k = hidden_states.shape

        assert (
                input_global_scale.dtype == torch.float32
        ), f"input_global_scale must be float32, got {input_global_scale.dtype}"
        assert input_global_scale.shape == (
            num_experts,
        ), f"input_global_scale must be (l,), got {input_global_scale.shape}"

        a_q, a_q_sf = scaled_fp4_grouped_quant(
            hidden_states,
            input_global_scale,
            masked_m,
        )

    a_q_slow, a_q_sf_slow = scaled_fp4_grouped_quant(
        dispatch_output_bf16.hidden_states_fp8,
        input_global_scale.repeat(num_experts),
        dispatch_output_bf16.masked_m,
    )

    # HACK: use bf16 outputs
    a_q, a_q_sf = a_q_slow, a_q_sf_slow

    assert w1.shape[-2] == 2 * n, f"w1 last-2 dim must be 2*n, got {w1.shape}"
    assert (
        w1.shape[-1] * 2 == k
    ), f"w1 last dim * 2 must equal k, got {w1.shape[-1]} vs k={k}"
    assert w2.shape[-2:] == (
        k,
        n // 2,
    ), f"w2 shape mismatch, got {w2.shape[-2:]}, expected {(k, n//2)}"
    assert w1_alpha.shape == (
        num_experts,
    ), f"w1_alpha must be (l,), got {w1_alpha.shape}"
    assert a2_global_scale.shape == (
        num_experts,
    ), f"a2_global_scale must be (l,), got {a2_global_scale.shape}"
    assert w2_alpha.shape == (
        num_experts,
    ), f"w2_alpha must be (l,), got {w2_alpha.shape}"

    # TODO(kaixih@nvidia): dtype should be based on inputs.
    gateup_output = torch.empty(
        (num_experts, m, n * 2), dtype=torch.bfloat16, device=a_q.device
    )
    gateup_output = gateup_output.permute(1, 2, 0)  # requirement of kernel
    sf_vec_size = 16
    assert a_q_sf.dtype == torch.float8_e4m3fn
    assert a_q.dtype == torch.uint8
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"

    # Gemm1
    grouped_gemm_nt_masked(
        (a_q, a_q_sf),
        (w1.permute(1, 2, 0), w1_blockscale),
        gateup_output,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w1_alpha.view(1, 1, num_experts),
        alpha_dtype=get_cute_dtype(w1_alpha),
    )  # in logical [m, n, l]

    # dumper.dump("moe__gateup_output", gateup_output, layer_id=layer_id)

    # SILU and quantization
    diq, diq_sf = silu_and_mul_scaled_fp4_grouped_quant(
        gateup_output.permute(2, 0, 1),
        a2_global_scale,
        masked_m,
    )

    # Gemm2
    # out = torch.empty(
    # NOTE HACK
    out = torch.zeros(
        (num_experts, m, k), dtype=torch.bfloat16, device=a_q.device
    )
    out_lmk = out
    out = out.permute(1, 2, 0)  # requirement of kernel
    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1, 2, 0), w2_blockscale),
        out,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w2_alpha.view(1, 1, num_experts),
        alpha_dtype=get_cute_dtype(w2_alpha),
    )  # in logical [m, k, l]

    # dumper.dump("moe__out", out, layer_id=layer_id)
    # dumper.dump("moe__any_isnan_out", torch.any(torch.isnan(out)), layer_id=layer_id)

    # if any(
    #     torch.any(torch.isnan(
    #         out_lmk[local_expert_idx, :masked_m[local_expert_idx]]
    #     )).cpu().item()
    #     for local_expert_idx in range(len(masked_m))
    # ):
    #     print(
    #         f"[{torch.distributed.get_rank()}] hi flashinfer_cutedsl_moe_masked find nan! thus extra dump!"
    #         # f"{hidden_states=} "
    #         # f"{masked_m=} "
    #         # f"{gateup_output=} {gateup_output.sum()=} "
    #         # f"{out_lmk=} {out_lmk.sum()=} "
    #         ,
    #         flush=True
    #     )
    #
    #     dumper.dump("moe__hidden_states_a", hidden_states[0], layer_id=layer_id)
    #     dumper.dump("moe__hidden_states_b", hidden_states[1], layer_id=layer_id)
    #     dumper.dump("moe__masked_m", masked_m, layer_id=layer_id)
    #     dumper.dump("moe__gateup_output", gateup_output, layer_id=layer_id)
    #     dumper.dump("moe__out_lmk", out_lmk, layer_id=layer_id)
    #     dumper.dump("moe__any_isnan_out", torch.any(torch.isnan(out_lmk)), layer_id=layer_id)

    return out.permute(2, 0, 1)
