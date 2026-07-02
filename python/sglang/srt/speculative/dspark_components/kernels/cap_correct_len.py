from __future__ import annotations

import torch
import triton
import triton.language as tl

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
        return cap_correct_len_triton(
            correct_len=correct_len,
            layout=layout,
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


@triton.jit
def _cap_correct_len_kernel(
    correct_len_ptr,
    verify_lens_ptr,
    capped_ptr,
    trim_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    cl = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int64)
    vl = tl.load(verify_lens_ptr + offs, mask=mask, other=0).to(tl.int64)
    ell = vl - 1
    capped = tl.minimum(cl, ell)
    trim = cl - capped
    tl.store(capped_ptr + offs, capped, mask=mask)
    tl.store(trim_ptr + offs, trim, mask=mask)


def cap_correct_len_triton(
    *,
    correct_len: torch.Tensor,
    layout: RaggedVerifyLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fuse (verify_lens - 1) + minimum + subtract into one launch. Outputs keep
    # correct_len's dtype (int32 on the sampling path, int64 on the greedy path).
    device = correct_len.device
    correct_len = correct_len.contiguous()
    verify_lens = layout.verify_lens.to(device=device).contiguous()
    n = correct_len.shape[0]
    capped = torch.empty_like(correct_len)
    trim = torch.empty_like(correct_len)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _cap_correct_len_kernel[grid](
        correct_len, verify_lens, capped, trim, n, BLOCK=BLOCK
    )
    return capped, trim
