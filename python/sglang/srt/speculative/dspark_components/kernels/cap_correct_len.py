from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_CAP_CORRECT_LEN.get()


class CapCorrectLen:
    @classmethod
    def execute(cls, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        correct_len: torch.Tensor,
        layout: RaggedVerifyLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cap_correct_len(
            correct_len=correct_len,
            layout=layout,
        )

    @classmethod
    def triton(
        cls,
        *,
        correct_len: torch.Tensor,
        layout: RaggedVerifyLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "CapCorrectLen.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_CAP_CORRECT_LEN=torch until the triton kernel lands."
        )


def cap_correct_len(
    *,
    correct_len: torch.Tensor,
    layout: RaggedVerifyLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Cutoff-only cap: commit at most ell_r = verify_len - 1 correct drafts per
    # request. Capping accept is lossless -- fewer correctly-verified drafts are
    # committed and the bonus (recomputed by callers at the capped index) is
    # still the target's true next token at the cap.
    #
    # cap_trim_lens = correct_len - capped (>= 0) is the per-request count of
    # target-correct drafts the confidence cap dropped. Only the CAP_ACCEPT mode
    # (full bs*(gamma+1) window) makes this observable: there correct_len is the
    # true full-block accept length, so the delta measures the accept-length the
    # confidence schedule left on the table. COMPACT only computes 1+ell_r tokens
    # so correct_len <= ell_r already and the delta is 0; STATIC has no cap.
    ell_r = (layout.verify_lens.to(device=correct_len.device) - 1).to(correct_len.dtype)
    capped = torch.minimum(correct_len, ell_r)
    cap_trim_lens = correct_len - capped
    return capped, cap_trim_lens
