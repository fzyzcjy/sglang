from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_BLOCK_SEQ_LENS_CASUAL.get()


class BuildBlockSeqLensCasual:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        seq_lens: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_block_seq_lens_casual(
            seq_lens=seq_lens,
            block_size=block_size,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        seq_lens: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_block_seq_lens_casual_triton(
            seq_lens=seq_lens,
            block_size=block_size,
            device=device,
        )


def build_block_seq_lens_casual(
    *,
    seq_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    # The per-token causal length for a uniform-gamma draft block: request r's gamma
    # tokens have causal lengths prefix_r + 1 .. prefix_r + gamma (the non-causal index
    # builder only reads the first-token prefix per request, but the layout must match
    # expand_prefill_casually's [prefix+1 .. prefix+gamma] ordering).
    prefix = seq_lens.to(torch.int32)
    steps = torch.arange(1, block_size + 1, device=device, dtype=torch.int32)
    return (prefix[:, None] + steps[None, :]).reshape(-1)


@triton.jit
def _block_seq_lens_casual_kernel(
    seq_lens_ptr,
    out_ptr,
    block_size,
    n_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    row = offs // block_size
    col = offs % block_size
    prefix = tl.load(seq_lens_ptr + row, mask=mask, other=0)
    tl.store(out_ptr + offs, (prefix + col + 1).to(tl.int32), mask=mask)


def build_block_seq_lens_casual_triton(
    *,
    seq_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    # Single fused kernel: out[r*G + j] = seq_lens[r] + (j + 1), replacing the
    # arange + broadcast-add + reshape torch chain with one launch.
    seq_lens = seq_lens.to(device=device, dtype=torch.int64).contiguous()
    n_rows = seq_lens.shape[0]
    n_out = n_rows * block_size
    out = torch.empty(n_out, dtype=torch.int32, device=device)
    BLOCK = 256
    grid = (triton.cdiv(n_out, BLOCK),)
    _block_seq_lens_casual_kernel[grid](seq_lens, out, block_size, n_out, BLOCK=BLOCK)
    return out
