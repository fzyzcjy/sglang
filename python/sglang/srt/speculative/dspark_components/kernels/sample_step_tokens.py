from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_SAMPLE_STEP_TOKENS.get()

_BLOCK_V = 1024
# Triton forbids reading a bare module-global from inside @jit; wrap the sentinel as a
# tl.constexpr so the argmax kernels can reference it directly (host code never reads it).
_IDX_SENTINEL = tl.constexpr(2147483647)


class SampleStepTokens:
    @classmethod
    def execute(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(
                step_logits=step_logits,
                temperatures=temperatures,
                greedy_mask=greedy_mask,
                exp_noise=exp_noise,
            )
        return cls.triton(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )

    @classmethod
    def torch(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        return sample_step_tokens(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )

    @classmethod
    def triton(
        cls,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        exp_noise: torch.Tensor,
    ) -> torch.Tensor:
        return sample_step_tokens_triton(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )


def sample_step_tokens(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> torch.Tensor:
    # Byte-identical to the pre-kernel fast_sampling path, but consumes the caller's
    # injected exp_noise instead of drawing it (the draw stays in the caller to keep the
    # torch RNG stream unchanged). Gumbel-max: argmax(softmax(s) / Exp(1)) ~ Categorical.
    # greedy rows get noise 1 so argmax(probs / 1) == argmax(logits).
    probs = torch.softmax(step_logits.float() / temperatures[:, None], dim=-1)
    noise = torch.where(greedy_mask[:, None], 1.0, exp_noise)
    return probs.div_(noise).argmax(dim=-1)


@triton.jit
def _online_partial_kernel(
    logits_ptr,
    temperatures_ptr,
    greedy_mask_ptr,
    exp_noise_ptr,
    tile_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    V,
    stride_row,
    n_tiles,
    BLOCK_V: tl.constexpr,
):
    # Single vocab pass per tile (fuses the old row_max + argmax partials): emit this
    # tile's max of s = logits/T, plus its best ratio-space key + smallest-index argmax
    # measured RELATIVE to the tile max. The combine kernel rescales tile keys to the
    # global row max, so a two-pass global-max exp is never needed.
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    logits = tl.load(
        logits_ptr + row * stride_row + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    temperature = tl.load(temperatures_ptr + row)
    s = logits / temperature
    tile_max = tl.max(s, axis=0)
    greedy = tl.load(greedy_mask_ptr + row) != 0
    noise = tl.load(exp_noise_ptr + row * V + offs, mask=mask, other=1.0)
    denom = tl.where(greedy, 1.0, noise)
    # Ratio-space key relative to the tile max: exp underflow -> 0 excludes far-tail tokens
    # exactly like fp32 softmax; the normalization constant Z is dropped (argmax-invariant).
    key = tl.exp(s - tile_max) / denom
    key = tl.where(mask, key, -1.0)
    tile_best = tl.max(key, axis=0)
    # Tie-break to the smaller index (match torch.argmax): min over indices at the max.
    idx = tl.where(key == tile_best, offs, _IDX_SENTINEL)
    tl.store(tile_max_ptr + row * n_tiles + tile, tile_max)
    tl.store(partial_key_ptr + row * n_tiles + tile, tile_best)
    tl.store(partial_idx_ptr + row * n_tiles + tile, tl.min(idx, axis=0))


@triton.jit
def _online_combine_kernel(
    tile_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    next_tokens_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    # Rescale each tile's best key from its tile max to the global row max
    # (exp(tile_max - global_max)), recovering exp(s - global_max)/denom bit-for-bit up to
    # fp rounding, then take the global argmax with smallest-index tie-break across tiles.
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    tile_max = tl.load(
        tile_max_ptr + row * n_tiles + offs, mask=mask, other=float("-inf")
    )
    keys = tl.load(partial_key_ptr + row * n_tiles + offs, mask=mask, other=-1.0)
    idxs = tl.load(
        partial_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL
    )
    global_max = tl.max(tile_max, axis=0)
    rescaled = keys * tl.exp(tile_max - global_max)
    rescaled = tl.where(mask, rescaled, -1.0)
    best = tl.max(rescaled, axis=0)
    cand = tl.where(rescaled == best, idxs, _IDX_SENTINEL)
    tl.store(next_tokens_ptr + row, tl.min(cand, axis=0).to(tl.int64))


def sample_step_tokens_triton(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> torch.Tensor:
    # Ratio-space online argmax in two kernels (fused from the old four): a per-tile pass
    # that emits (tile_max, best key rel tile_max, best idx) and a combine that rescales to
    # the global row max and takes the global argmax. Never materializes probs / key over the
    # full vocab and parallelizes along vocab (bs=1 still fills the SMs), replacing the
    # batch-dim softmax + argmax reductions that idle the SMs at bs=1. Reads step_logits at
    # its own row stride so the caller's cropped view (full[..., :vocab] of the gathered
    # padded logits) is consumed in place -- no .contiguous() crop copy per step.
    bs, V = step_logits.shape
    device = step_logits.device
    assert step_logits.stride(1) == 1, "step_logits rows must be contiguous"
    stride_row = step_logits.stride(0)
    temperatures = temperatures.to(torch.float32).contiguous()
    greedy_mask = greedy_mask.to(torch.int32).contiguous()
    exp_noise = exp_noise.to(torch.float32).contiguous()

    n_tiles = triton.cdiv(V, _BLOCK_V)
    block_tiles = triton.next_power_of_2(n_tiles)

    tile_max = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_key = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_idx = torch.empty((bs, n_tiles), dtype=torch.int32, device=device)
    next_tokens = torch.empty((bs,), dtype=torch.int64, device=device)

    tile_grid = (bs, n_tiles)
    row_grid = (bs,)

    _online_partial_kernel[tile_grid](
        step_logits,
        temperatures,
        greedy_mask,
        exp_noise,
        tile_max,
        partial_key,
        partial_idx,
        V,
        stride_row,
        n_tiles,
        BLOCK_V=_BLOCK_V,
    )
    _online_combine_kernel[row_grid](
        tile_max, partial_key, partial_idx, next_tokens, n_tiles, BLOCK_TILES=block_tiles
    )
    return next_tokens
