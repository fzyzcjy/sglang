from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_COMPACT_LAYOUT.get()


class CompactRowIndex:
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
        verify_lens: torch.Tensor,
        padded_total: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compact_row_index(
            verify_lens=verify_lens,
            padded_total=padded_total,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        padded_total: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "CompactRowIndex.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_COMPACT_LAYOUT=torch until the triton kernel lands."
        )


class CompactVerifyIds:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        layout: RaggedVerifyLayout,
        device: str,
    ) -> torch.Tensor:
        return compact_verify_ids(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        layout: RaggedVerifyLayout,
        device: str,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CompactVerifyIds.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_COMPACT_LAYOUT=torch until the triton kernel lands."
        )


def compact_verify_ids(
    *,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    layout: RaggedVerifyLayout,
    device: str,
) -> torch.Tensor:
    # Pack [anchor, s_0..s_{ell_r-1}] per request into a compact graph_num_tokens-row
    # tensor; padding tail (valid False) is zeroed. anchor = draft_block_ids[:, 0].
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    bs = layout.verify_lens.shape[0]
    safe_req = req_id.clamp(max=bs - 1)  # sink req_id == bs on padding rows
    anchors = draft_block_ids[:, 0]
    # within==0 -> anchor; else draft_tokens[:, within-1] (clamp masked at 0).
    drafts = draft_tokens[safe_req, (within - 1).clamp_min(0)]
    verify_ids = torch.where(within == 0, anchors[safe_req], drafts)
    verify_ids = torch.where(valid, verify_ids, torch.zeros_like(verify_ids))
    return verify_ids.to(torch.int64)


def compact_row_index(
    *,
    verify_lens: torch.Tensor,
    padded_total: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (req_id, within, valid) for each row of a FIXED padded_total-row compact buffer.
    #
    # padded_total is the host-known, bs-derived graph_num_tokens (>= the real total
    # sum(verify_lens)); the first real_total rows pack each request's window back to
    # back, rows [real_total, padded_total) are padding. The old design sized the
    # buffer to the exact host `total` via repeat_interleave(output_size=total), which
    # forced the caller to D2H verify_lens off the forward stream first (one
    # per-step cudaStreamSynchronize -- the very sync this removes). Here real_total is
    # a DEVICE scalar (cumsum[-1]) compared against a host-sized arange, so the row->req
    # map is built with NO compute-stream sync. Padding rows carry req_id == bs (a sink
    # id the callers route to a discarded slot) and within == 0.
    verify_lens = verify_lens.to(device=device, dtype=torch.int64)
    bs = int(verify_lens.numel())
    incl = torch.cumsum(verify_lens, dim=0)  # inclusive prefix sum, device [bs]
    start = incl - verify_lens  # exclusive per-request start
    real_total = incl[-1]  # DEVICE scalar; never read to host
    row = torch.arange(padded_total, device=device, dtype=torch.int64)
    valid = row < real_total  # device mask, no sync
    # searchsorted(incl, row, right=True) is the owning request; rows >= real_total map
    # to bs (past the last request) -> routed to the sink.
    req_id = torch.searchsorted(incl, row, right=True)
    req_id = torch.where(valid, req_id, torch.full_like(req_id, bs))
    within = torch.where(
        valid, row - start[req_id.clamp(max=bs - 1)], torch.zeros_like(row)
    )
    return req_id, within, valid
