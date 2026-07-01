from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dspark_components.dspark_info import RaggedVerifyWindow
from sglang.srt.speculative.dspark_components.kernels.compact_layout import (
    compact_row_index,
    compact_verify_ids,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_RAGGED_WINDOW.get()


class BuildRaggedVerifyWindow:
    @classmethod
    def execute(cls, *args, **kwargs) -> RaggedVerifyWindow:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        return build_ragged_verify_window(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )

    @classmethod
    def triton(
        cls,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        raise NotImplementedError(
            "BuildRaggedVerifyWindow.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_RAGGED_WINDOW=torch until the triton kernel lands."
        )


def build_ragged_verify_window(
    *,
    batch: ScheduleBatch,
    layout: RaggedVerifyLayout,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    # Compact (real-N) verify window, sync-free: sized to the host-known padded
    # graph_num_tokens (no verify_lens D2H; see compact_row_index). Request r contributes
    # its [anchor, s_0..s_{ell_r-1}] = verify_len_r tokens; cache slots are over-allocated
    # to the full gamma+1 block (plan section 6). Padding rows (valid False) get position
    # 0 / slot 0 -- the runner's tail-zero contract (no stale KV slot / vocab id read).
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens

    req_id, within, valid = compact_row_index(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    safe_req = req_id.clamp(max=bs - 1)  # sink req_id == bs on padding rows
    positions = torch.where(
        valid,
        prefix_lens.to(torch.int64)[safe_req] + within,
        torch.zeros_like(within),
    )
    # Cache locs are written compactly (same ordering as req_id/within) into a bs*stride
    # buffer; pad to graph_num_tokens and zero padding rows (also clears the
    # [real_total, bs*stride) torch.empty tail).
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    verify_cache_loc = torch.nn.functional.pad(
        real_cache_loc, (0, padded_total - real_cache_loc.shape[0])
    )
    verify_cache_loc = torch.where(
        valid, verify_cache_loc, torch.zeros_like(verify_cache_loc)
    )

    verify_ids = compact_verify_ids(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )

    # Host verify seq_lens is NOT built here (keeps the window sync-free); the
    # dense-backend host pre-add lives in the worker's _run_ragged_target_verify.
    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
    )
