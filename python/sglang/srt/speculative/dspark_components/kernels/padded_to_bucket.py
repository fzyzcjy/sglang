from __future__ import annotations

import torch
import triton
import triton.language as tl

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
        return pad_verify_lens_to_bucket_triton(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            num_draft_tokens=num_draft_tokens,
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


@triton.jit
def _padded_to_bucket_kernel(
    verify_lens_ptr,
    out_ptr,
    bs,
    padded_bs,
    num_draft_tokens,
    graph_num_tokens,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    valid = idx < padded_bs
    is_real = idx < bs
    vl = tl.load(verify_lens_ptr + idx, mask=is_real, other=0).to(tl.int64)
    base = tl.where(is_real, vl, num_draft_tokens)
    base = tl.where(valid, base, 0)
    # leftover (graph_num_tokens - sum of the padded lens) folds into the last slot,
    # replacing the cat + device-scalar sum + in-place add torch chain.
    leftover = graph_num_tokens - tl.sum(base)
    is_last = idx == (padded_bs - 1)
    final = base + tl.where(is_last, leftover, 0)
    tl.store(out_ptr + idx, final.to(tl.int32), mask=valid)


def pad_verify_lens_to_bucket_triton(
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
    verify_lens = verify_lens.to(torch.int32).contiguous()
    out = torch.empty(padded_bs, dtype=torch.int32, device=device)
    BLOCK = triton.next_power_of_2(max(padded_bs, 1))
    _padded_to_bucket_kernel[(1,)](
        verify_lens,
        out,
        bs,
        padded_bs,
        num_draft_tokens,
        graph_num_tokens,
        BLOCK=BLOCK,
    )
    return out
