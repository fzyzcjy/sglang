from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.kernels.compact_layout import (
    compact_row_index,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_SCATTER.get()


class ScatterCompactToStrided:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        compact: torch.Tensor,
        layout: RaggedVerifyLayout,
        fill_value: float,
        verify_num_draft_tokens: int,
    ) -> torch.Tensor:
        return scatter_compact_to_strided(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=verify_num_draft_tokens,
        )

    @classmethod
    def triton(
        cls,
        *,
        compact: torch.Tensor,
        layout: RaggedVerifyLayout,
        fill_value: float,
        verify_num_draft_tokens: int,
    ) -> torch.Tensor:
        return scatter_compact_to_strided_triton(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=verify_num_draft_tokens,
        )


def scatter_compact_to_strided(
    *,
    compact: torch.Tensor,
    layout: RaggedVerifyLayout,
    fill_value: float,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    # compact->strided scatter (infra section 12.E). compact is [graph_num_tokens, dim];
    # scatter to the [bs*(gamma+1), dim] strided layout accept/inject expect (request i
    # owns rows [i*stride, i*stride+verify_len_i); intra-request pad = fill_value, kept
    # out of commit by the accept cap -> lossless). The compact padding tail (valid False)
    # routes to a throwaway sink row that the [:bs*stride] return drops, so the output is
    # shape/semantics-identical to the old real-N scatter -- accept/inject stay unchanged.
    stride = verify_num_draft_tokens
    bs = layout.verify_lens.shape[0]
    dim = compact.shape[1]
    device = compact.device
    strided = torch.full(
        (bs * stride + 1, dim), fill_value, dtype=compact.dtype, device=device
    )
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    sink = bs * stride
    strided_pos = torch.where(
        valid,
        req_id.clamp(max=bs - 1) * stride + within,
        torch.full_like(within, sink),
    )
    strided.index_copy_(0, strided_pos, compact)
    return strided[: bs * stride]


@triton.jit
def _scatter_compact_to_strided_kernel(
    compact_ptr,
    verify_lens_ptr,
    start_ptr,
    out_ptr,
    stride,
    dim,
    fill_value,
    BLOCK_D: tl.constexpr,
):
    o = tl.program_id(0).to(tl.int64)
    dblk = tl.program_id(1)
    i = o // stride
    w = o % stride
    vl_i = tl.load(verify_lens_ptr + i)
    start_i = tl.load(start_ptr + i)
    d = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = d < dim
    # Inverse of the scatter: out[i*stride + w] = compact[start_i + w] iff w < verify_len_i
    # (request i packs its window contiguously at compact rows [start_i, start_i+vl_i));
    # else fill_value. Gather (one read per out row) -> no scatter races.
    in_range = w < vl_i
    src = tl.where(in_range, start_i + w, 0)
    val = tl.load(compact_ptr + src * dim + d, mask=dmask & in_range, other=0)
    val = tl.where(in_range, val, fill_value)
    tl.store(out_ptr + o * dim + d, val, mask=dmask)


def scatter_compact_to_strided_triton(
    *,
    compact: torch.Tensor,
    layout: RaggedVerifyLayout,
    fill_value: float,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    stride = verify_num_draft_tokens
    bs = layout.verify_lens.shape[0]
    dim = compact.shape[1]
    device = compact.device
    compact = compact.contiguous()
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int64).contiguous()
    start = (torch.cumsum(verify_lens, dim=0) - verify_lens).contiguous()
    n_out = bs * stride
    out = torch.empty((n_out, dim), dtype=compact.dtype, device=device)
    BLOCK_D = 1024
    grid = (n_out, triton.cdiv(dim, BLOCK_D))
    _scatter_compact_to_strided_kernel[grid](
        compact,
        verify_lens,
        start,
        out,
        stride,
        dim,
        fill_value,
        BLOCK_D=BLOCK_D,
    )
    return out
