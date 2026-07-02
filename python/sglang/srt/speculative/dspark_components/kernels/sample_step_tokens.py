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
def _row_max_partial_kernel(
    logits_ptr,
    temperatures_ptr,
    partial_max_ptr,
    V,
    n_tiles,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    logits = tl.load(logits_ptr + row * V + offs, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    temperature = tl.load(temperatures_ptr + row)
    s = logits / temperature
    tl.store(partial_max_ptr + row * n_tiles + tile, tl.max(s, axis=0))


@triton.jit
def _row_max_combine_kernel(
    partial_max_ptr,
    row_max_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    partial = tl.load(
        partial_max_ptr + row * n_tiles + offs, mask=mask, other=float("-inf")
    )
    tl.store(row_max_ptr + row, tl.max(partial, axis=0))


@triton.jit
def _argmax_partial_kernel(
    logits_ptr,
    temperatures_ptr,
    greedy_mask_ptr,
    exp_noise_ptr,
    row_max_ptr,
    partial_key_ptr,
    partial_idx_ptr,
    V,
    n_tiles,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    logits = tl.load(logits_ptr + row * V + offs, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    temperature = tl.load(temperatures_ptr + row)
    row_max = tl.load(row_max_ptr + row)
    s = logits / temperature
    greedy = tl.load(greedy_mask_ptr + row) != 0
    noise = tl.load(exp_noise_ptr + row * V + offs, mask=mask, other=1.0)
    denom = tl.where(greedy, 1.0, noise)
    # Ratio-space key: exp underflow -> 0 excludes far-tail tokens exactly like fp32
    # softmax; the normalization constant Z is dropped since it does not move argmax.
    key = tl.exp(s - row_max) / denom
    key = tl.where(mask, key, -1.0)
    local_max = tl.max(key, axis=0)
    # Tie-break to the smaller index (match torch.argmax): min over indices at the max.
    idx = tl.where(key == local_max, offs, _IDX_SENTINEL)
    tl.store(partial_key_ptr + row * n_tiles + tile, local_max)
    tl.store(partial_idx_ptr + row * n_tiles + tile, tl.min(idx, axis=0))


@triton.jit
def _argmax_combine_kernel(
    partial_key_ptr,
    partial_idx_ptr,
    next_tokens_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    keys = tl.load(partial_key_ptr + row * n_tiles + offs, mask=mask, other=-1.0)
    idxs = tl.load(
        partial_idx_ptr + row * n_tiles + offs, mask=mask, other=_IDX_SENTINEL
    )
    global_max = tl.max(keys, axis=0)
    # Combine tiles: smallest index among tiles hitting the global max key. Each tile's
    # partial_idx is already its own smallest max index, so this is the global argmax.
    cand = tl.where(keys == global_max, idxs, _IDX_SENTINEL)
    tl.store(next_tokens_ptr + row, tl.min(cand, axis=0).to(tl.int64))


def sample_step_tokens_triton(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> torch.Tensor:
    # Ratio-space two-stage argmax. Never materializes probs / key over the full vocab
    # and parallelizes both reductions along vocab (bs=1 still fills the SMs), replacing
    # the batch-dim softmax + argmax reductions that idle the SMs at bs=1.
    bs, V = step_logits.shape
    device = step_logits.device
    step_logits = step_logits.contiguous()
    temperatures = temperatures.to(torch.float32).contiguous()
    greedy_mask = greedy_mask.to(torch.int32).contiguous()
    exp_noise = exp_noise.to(torch.float32).contiguous()

    n_tiles = triton.cdiv(V, _BLOCK_V)
    block_tiles = triton.next_power_of_2(n_tiles)

    row_max = torch.empty((bs,), dtype=torch.float32, device=device)
    partial_max = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_key = torch.empty((bs, n_tiles), dtype=torch.float32, device=device)
    partial_idx = torch.empty((bs, n_tiles), dtype=torch.int32, device=device)
    next_tokens = torch.empty((bs,), dtype=torch.int64, device=device)

    tile_grid = (bs, n_tiles)
    row_grid = (bs,)

    _row_max_partial_kernel[tile_grid](
        step_logits, temperatures, partial_max, V, n_tiles, BLOCK_V=_BLOCK_V
    )
    _row_max_combine_kernel[row_grid](
        partial_max, row_max, n_tiles, BLOCK_TILES=block_tiles
    )
    _argmax_partial_kernel[tile_grid](
        step_logits,
        temperatures,
        greedy_mask,
        exp_noise,
        row_max,
        partial_key,
        partial_idx,
        V,
        n_tiles,
        BLOCK_V=_BLOCK_V,
    )
    _argmax_combine_kernel[row_grid](
        partial_key, partial_idx, next_tokens, n_tiles, BLOCK_TILES=block_tiles
    )
    return next_tokens
