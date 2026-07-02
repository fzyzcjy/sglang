from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_SOFTMAX_TEMP.get()


class SoftmaxTemp:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        return softmax_temp(
            logits=logits,
            temperatures=temperatures,
            rows_per_request=rows_per_request,
        )

    @classmethod
    def triton(
        cls,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        rows_per_request: int,
    ) -> torch.Tensor:
        return softmax_temp_triton(
            logits=logits,
            temperatures=temperatures,
            rows_per_request=rows_per_request,
        )


def softmax_temp(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    # Reference chain: float cast -> per-request temperature divide -> row softmax.
    # ``logits`` is [num_rows, vocab] with num_rows = bs * rows_per_request (the row
    # block of request r covers rows [r*rows_per_request, (r+1)*rows_per_request));
    # ``temperatures`` is [bs] (or [bs, 1]) float. Replaces the un-fused
    # repeat_interleave + div + softmax launch chain at the accept call sites.
    num_rows = logits.shape[0]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(bs).to(torch.float32), rows_per_request, dim=0
    )
    scaled = logits.to(torch.float32) / temp_per_row[:, None]
    return torch.softmax(scaled, dim=-1)


@triton.jit
def _softmax_temp_kernel(
    logits_ptr,
    temp_ptr,
    out_ptr,
    vocab,
    rows_per_request,
    logits_row_stride,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_row_stride
    out_base = out_ptr + row.to(tl.int64) * vocab

    row_max = -float("inf")
    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    sum_exp = 0.0
    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        e = tl.exp(x - row_max)
        e = tl.where(vmask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        e = tl.exp(x - row_max)
        tl.store(out_base + offs, e / sum_exp, mask=vmask)


def softmax_temp_triton(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    num_rows, vocab = logits.shape[0], logits.shape[-1]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temperatures = temperatures.reshape(bs).to(torch.float32).contiguous()
    out = torch.empty((num_rows, vocab), dtype=torch.float32, device=logits.device)
    BLOCK_V = 4096
    _softmax_temp_kernel[(num_rows,)](
        logits,
        temperatures,
        out,
        vocab,
        rows_per_request,
        logits.stride(0),
        BLOCK_V=BLOCK_V,
    )
    return out
