from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    CapCorrectLen,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_ACCEPT_GREEDY.get()


class AcceptGreedy:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_layout: Optional[RaggedVerifyLayout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_greedy(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_layout=cutoff_layout,
        )

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        verify_num_draft_tokens: int,
        cutoff_layout: Optional[RaggedVerifyLayout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "AcceptGreedy.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_ACCEPT_GREEDY=torch until the triton kernel lands."
        )


def accept_greedy(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_layout: Optional[RaggedVerifyLayout] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_layout is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, layout=cutoff_layout
        )
        row_ids = torch.arange(bs, device=target_predict.device)
        bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
    return correct_len, bonus, cap_trim_lens
