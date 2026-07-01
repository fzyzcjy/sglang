from __future__ import annotations

import torch

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
        raise NotImplementedError(
            "BuildBlockSeqLensCasual.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_BLOCK_SEQ_LENS_CASUAL=torch until the triton kernel lands."
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
