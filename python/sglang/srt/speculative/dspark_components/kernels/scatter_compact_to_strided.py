from __future__ import annotations

import torch

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
        raise NotImplementedError(
            "ScatterCompactToStrided.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_SCATTER=torch until the triton kernel lands."
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
