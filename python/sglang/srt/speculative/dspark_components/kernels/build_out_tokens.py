from __future__ import annotations

import torch

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_BUILD_OUT_TOKENS.get()


class BuildOutTokens:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        verify_num_draft_tokens: int,
        gamma: int,
    ) -> torch.Tensor:
        return build_out_tokens(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=verify_num_draft_tokens,
            gamma=gamma,
        )

    @classmethod
    def triton(
        cls,
        *,
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        verify_num_draft_tokens: int,
        gamma: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "BuildOutTokens.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_BUILD_OUT_TOKENS=torch until the triton kernel lands."
        )


def build_out_tokens(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    bs = draft_tokens.shape[0]
    out_tokens = torch.empty(
        (bs, verify_num_draft_tokens),
        dtype=torch.int64,
        device=draft_tokens.device,
    )
    out_tokens[:, :gamma].copy_(draft_tokens)
    out_tokens[:, gamma].fill_(0)
    out_tokens.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
    return out_tokens
