from __future__ import annotations

import torch

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_PADDED_TO_BUCKET.get()


class PaddedToBucket:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        return pad_verify_lens_to_bucket(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            num_draft_tokens=num_draft_tokens,
        )

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "PaddedToBucket.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_PADDED_TO_BUCKET=torch until the triton kernel lands."
        )


def pad_verify_lens_to_bucket(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    num_draft_tokens: int,
) -> torch.Tensor:
    padded_bs = graph_num_tokens // num_draft_tokens
    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: graph_num_tokens "
        f"{graph_num_tokens} cannot hold this batch's requests"
    )
    device = verify_lens.device
    num_pad_reqs = padded_bs - bs
    padded = verify_lens.to(torch.int32)
    if num_pad_reqs > 0:
        pad_block = torch.full(
            (num_pad_reqs,), num_draft_tokens, dtype=torch.int32, device=device
        )
        padded = torch.cat([padded, pad_block])
    else:
        padded = padded.clone()
    # leftover = graph_num_tokens - sum(padded); a DEVICE scalar folded into the
    # last (synthetic, or last real when padded_bs == bs) request. Kept on device
    # so the pad introduces no compute-stream sync.
    leftover = graph_num_tokens - padded.to(torch.int64).sum()
    padded[-1] = (padded[-1].to(torch.int64) + leftover).to(torch.int32)
    return padded
