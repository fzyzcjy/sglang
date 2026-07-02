from __future__ import annotations

import torch

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_COMMIT_KV_PROJ.get()

_STACKED_WEIGHT_CACHE: dict[int, torch.Tensor] = {}


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
    # Faithful original chain: one quantized-linear forward per stage. Every stage
    # projects the SAME main_x, so on the fp8 path this re-quantizes the input and
    # pays a full linear dispatch chain per stage.
    return [linear(main_x)[0] for linear in wkv_linears]


def commit_kv_proj_fused(
    *,
    main_x: torch.Tensor,
    wkv_linears: list[torch.nn.Module],
) -> list[torch.Tensor]:
    # One bf16 GEMM over the stacked (dequantized) per-stage wkv weights replaces
    # the num_stages quant + fp8 GEMM dispatch chains; the [N, head_dim] contiguous
    # slices feed the per-stage pool writer. Not a triton kernel per se (the GEMM is
    # cuBLAS), but it is the launch-count-optimized impl behind the same toggle.
    # NOTE vs the fp8 reference: the dequantized bf16 weights are bit-exact copies
    # of the quantized values, and skipping the per-step input quantization only
    # removes a rounding step, but the GEMM numerics differ slightly from deep_gemm.
    weight = _stacked_wkv_weight(wkv_linears=wkv_linears)
    kv_all = torch.nn.functional.linear(main_x, weight)
    num_stages = len(wkv_linears)
    head_dim = kv_all.shape[-1] // num_stages
    return [
        kv_all[:, i * head_dim : (i + 1) * head_dim].contiguous()
        for i in range(num_stages)
    ]


def _stacked_wkv_weight(*, wkv_linears: list[torch.nn.Module]) -> torch.Tensor:
    # Built once per model instance (weights are static post-load); keyed by the
    # first linear's identity, mirroring the chain-verify buffer cache pattern.
    key = id(wkv_linears[0])
    cached = _STACKED_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = torch.cat(
            [_dequant_linear_weight(linear) for linear in wkv_linears], dim=0
        )
        _STACKED_WEIGHT_CACHE[key] = cached
    return cached


def _dequant_linear_weight(linear: torch.nn.Module) -> torch.Tensor:
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(torch.bfloat16)
    assert weight.dtype == torch.float8_e4m3fn, (
        f"unsupported wkv weight dtype {weight.dtype} for the fused commit kv proj; "
        f"set SGLANG_DSPARK_KERNEL_COMMIT_KV_PROJ=torch"
    )
    # Blockwise fp8 scales sit on the standard 128x128 grid; deriving the block
    # from the scale shape is ambiguous when a dim is not an exact multiple.
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
