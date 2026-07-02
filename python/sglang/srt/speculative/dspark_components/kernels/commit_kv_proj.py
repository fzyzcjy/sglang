from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_COMMIT_KV_PROJ.get()

_STACKED_WEIGHT_CACHE: dict[int, _StackedWkvWeight] = {}


class CommitKvProj:
    @classmethod
    def execute(cls, *args, **kwargs) -> list[torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        main_x: torch.Tensor,
        wkv_linears: list[torch.nn.Module],
    ) -> list[torch.Tensor]:
        return commit_kv_proj(main_x=main_x, wkv_linears=wkv_linears)

    @classmethod
    def triton(
        cls,
        *,
        main_x: torch.Tensor,
        wkv_linears: list[torch.nn.Module],
    ) -> list[torch.Tensor]:
        return commit_kv_proj_fused(main_x=main_x, wkv_linears=wkv_linears)


def commit_kv_proj(
    *,
    main_x: torch.Tensor,
    wkv_linears: list[torch.nn.Module],
) -> list[torch.Tensor]:
    return [linear(main_x)[0] for linear in wkv_linears]


def commit_kv_proj_fused(
    *,
    main_x: torch.Tensor,
    wkv_linears: list[torch.nn.Module],
) -> list[torch.Tensor]:
    # One GEMM over the stacked per-stage wkv weights replaces the num_stages
    # quant + GEMM dispatch chains; the [N, head_dim] contiguous slices feed the
    # per-stage pool writer. Not a triton kernel per se, but it is the
    # launch-count-optimized impl behind the same toggle.
    #
    # Preferred path: stack the QUANTIZED blockwise-fp8 weights + scales along the
    # output dim (valid when every stage's out_dim is a whole number of scale
    # blocks) and call the same w8a8_block_fp8_linear the per-stage forward uses --
    # one input quant + one deep_gemm, per-element math identical to the per-stage
    # reference (each 128-row output block keeps its own scale; the input quant of
    # the shared main_x is deterministic, so quantizing once == quantizing thrice).
    # Fallback (unquantized / non-block layouts): one bf16 GEMM over dequantized
    # stacked weights -- the weights are bit-exact copies of the quantized values,
    # but the GEMM numerics differ slightly from deep_gemm.
    #
    # deep_gemm wants the weight scale mn-major (sf.stride(-2) == 1); _stacked_wkv_weight
    # rebuilds the cat'd scale into that layout so the fused GEMM runs on Blackwell as
    # well as Hopper (verified numerically equal to the per-stage path on both).
    num_stages = len(wkv_linears)
    stacked = _stacked_wkv_weight(wkv_linears=wkv_linears)

    if stacked.fp8_scale is not None:
        quant_method = wkv_linears[0].quant_method
        kv_all = quant_method.w8a8_block_fp8_linear(
            input=main_x,
            weight=stacked.weight,
            block_size=quant_method.quant_config.weight_block_size,
            weight_scale=stacked.fp8_scale,
            input_scale=None,
            bias=None,
        )
    else:
        kv_all = torch.nn.functional.linear(main_x, stacked.weight)

    head_dim = kv_all.shape[-1] // num_stages
    return [
        kv_all[:, i * head_dim : (i + 1) * head_dim].contiguous()
        for i in range(num_stages)
    ]


class _StackedWkvWeight(msgspec.Struct):
    weight: torch.Tensor
    fp8_scale: Optional[torch.Tensor]


def _stacked_wkv_weight(*, wkv_linears: list[torch.nn.Module]) -> _StackedWkvWeight:
    key = id(wkv_linears[0])
    cached = _STACKED_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = _build_stacked_wkv_weight(wkv_linears=wkv_linears)
        _STACKED_WEIGHT_CACHE[key] = cached
    return cached


def _build_stacked_wkv_weight(
    *, wkv_linears: list[torch.nn.Module]
) -> _StackedWkvWeight:
    first = wkv_linears[0]
    quant_method = first.quant_method
    block_quant = hasattr(quant_method, "block_quant") and quant_method.block_quant
    if block_quant and hasattr(quant_method, "w8a8_block_fp8_linear"):
        block_out = quant_method.quant_config.weight_block_size[0]
        if all(
            linear.weight.dtype == torch.float8_e4m3fn
            and linear.weight.shape[0] % block_out == 0
            for linear in wkv_linears
        ):
            weight = torch.cat([linear.weight for linear in wkv_linears], dim=0)
            if wkv_linears[0].weight_scale_inv.dtype == torch.int32:
                # Blackwell deepgemm path: per-stage scales are UE8M0-packed,
                # mn-major TMA-aligned tensors (stride(-2) == 1). torch.cat of
                # those silently yields a ROW-major tensor that fails deepgemm's
                # `sf.stride(-2) == 1` layout assert at the first forward. Stack
                # at the un-packed fp32 [N/128, K/128] block grid instead, then
                # re-apply the packing transform for the stacked mn (numerically
                # exact: each 128-row block keeps its own scale, and the inverse
                # self-checks round-trip consistency at startup).
                from sglang.srt.layers.quantization.fp8_utils import (
                    inverse_transform_scale_ue8m0,
                    transform_scale_ue8m0,
                )

                sf_fp32 = torch.cat(
                    [
                        inverse_transform_scale_ue8m0(
                            linear.weight_scale_inv, mn=linear.weight.shape[0]
                        )
                        for linear in wkv_linears
                    ],
                    dim=0,
                )
                scale = transform_scale_ue8m0(sf_fp32, mn=weight.shape[0])
                return _StackedWkvWeight(weight=weight, fp8_scale=scale)
            scale = torch.cat(
                [linear.weight_scale_inv for linear in wkv_linears], dim=0
            )
            # deep_gemm wants the weight scale mn-major (sf.stride(-2) == 1); a plain
            # contiguous cat is row-major (stride(-2) == in_blocks). Rebuild the last
            # two dims column-major so stride(-2) == 1 without changing values.
            if scale.dim() >= 2 and scale.stride(-2) != 1:
                scale = scale.transpose(-2, -1).contiguous().transpose(-2, -1)
            return _StackedWkvWeight(weight=weight, fp8_scale=scale)
    weight = torch.cat(
        [_dequant_linear_weight(linear) for linear in wkv_linears], dim=0
    )
    return _StackedWkvWeight(weight=weight, fp8_scale=None)


def _dequant_linear_weight(linear: torch.nn.Module) -> torch.Tensor:
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(torch.bfloat16)
    assert weight.dtype == torch.float8_e4m3fn, (
        f"unsupported wkv weight dtype {weight.dtype} for the fused commit kv proj; "
        f"set SGLANG_DSPARK_KERNEL_COMMIT_KV_PROJ=torch"
    )
    block = 128
    scale = linear.weight_scale_inv
    out_dim, in_dim = weight.shape
    expected_scale_shape = (
        (out_dim + block - 1) // block,
        (in_dim + block - 1) // block,
    )
    assert tuple(scale.shape) == expected_scale_shape, (
        f"wkv weight_scale_inv shape {tuple(scale.shape)} does not match the "
        f"128x128 block grid {expected_scale_shape} for weight {tuple(weight.shape)}; "
        f"set SGLANG_DSPARK_KERNEL_COMMIT_KV_PROJ=torch"
    )
    scale_full = scale.repeat_interleave(block, dim=0)[:out_dim]
    scale_full = scale_full.repeat_interleave(block, dim=1)[:, :in_dim]
    return (weight.to(torch.float32) * scale_full.to(torch.float32)).to(torch.bfloat16)
