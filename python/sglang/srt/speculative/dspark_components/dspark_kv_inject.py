from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dspark_components.kernels.commit_inject_layout import (
    BuildCommitInjectLayout,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func


class TargetHiddenKvInjector:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        model_runner,
        device,
        verify_num_draft_tokens: int,
        block_pos_offsets: torch.Tensor,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.model_runner = model_runner
        self.device = device
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self._block_pos_offsets = block_pos_offsets

    def inject_target_hidden(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        # Capability dispatch by pool type (no model-identity branch). The single
        # MLA latent pool (V4 SWA-latent) exposes the radix-fused SWA writer +
        # full->swa translate; the MHA k/v pool (dense qwen3 / gemma4) exposes the
        # split-k/v ``set_kv_buffer`` family. Both write the projected target hidden
        # as the draft KV that the draft block attends.
        if target_hidden is None or target_hidden.numel() == 0:
            return
        device = self.model_runner.device
        cache_loc = cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)
        positions = positions.to(device=device, dtype=torch.int64, non_blocking=True)
        target_hidden = target_hidden.to(device=device, non_blocking=True)
        # Under DP attention the eager prefill's target hidden is MLP-sync padded past
        # the real extend tokens, but the KV injection is keyed on the real
        # positions/cache_loc; drop the trailing pad rows to match. No-op when
        # unpadded (non-DP, or the decode-commit path where sizes already agree).
        n_real = positions.shape[0]
        if target_hidden.shape[0] > n_real:
            target_hidden = target_hidden[:n_real]
        if cache_loc_2d is not None:
            cache_loc_2d = cache_loc_2d.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if commit_lens is not None:
            commit_lens = commit_lens.to(
                device=device, dtype=torch.int32, non_blocking=True
            )

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            self._inject_mla(
                pool=pool,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )
            return

        # Dense MHA: the model owns the per-layer kv_proj + pool write (mirrors the
        # MLA branch delegating to write_target_hidden_kv).
        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                target_hidden=target_hidden,
                pool=pool,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )

    def _inject_mla(
        self,
        *,
        pool,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
    ) -> None:
        # MLA single-latent SWA injection (V4 draft). The worker owns the full->SWA
        # slot translation and the per-row absolute positions; the model owns the
        # per-stage projection + pool write (``write_target_hidden_kv`` iterates its
        # own ``self.stages``, projecting the target hidden through each ``wkv`` into
        # the RAW latent the radix-fused-norm-rope writer expects -- the kernel does
        # norm + rope + fp8 pack inside, pinned to ``_fused_norm_rope``, NOT ``_fused``).
        #
        # There is no MLA ``set_kv_buffer_prefix_valid`` (MHA-only). To keep this write
        # sync-free we do NOT masked-select the committed rows: that gather's output shape
        # depends on the mask, forcing a D2H count readback every verify step. Instead we
        # translate the FULL fixed-shape ``cache_loc`` and mark the non-committed rows
        # (columns ``>= commit_len[r]`` of the verify window) with ``swa_loc = -1`` via
        # ``torch.where``; the fused-norm-rope writer kernel skips ``out_loc < 0``, so only
        # columns ``0..commit_len[r]-1`` per request are written -- the same committed set
        # as the MHA prefix-valid write and the SoT reference (per-position target hidden,
        # not only the bonus). Prefill (no commit_lens) writes the whole window.
        # ``translate_loc_from_full_to_swa`` is a post-alloc integer lookup (slots already
        # allocated by assign_extend_cache_locs / _alloc_verify_window), so it stays
        # fixed-shape and sync-free even over the full window.
        swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
        if commit_lens is not None and cache_loc_2d is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
            committed_mask = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            swa_loc = torch.where(committed_mask, swa_loc, torch.full_like(swa_loc, -1))

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

    def inject_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        hidden_strided: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> None:
        # Inject the committed tokens' target hidden into the draft KV using the
        # strided layout. The full gamma+1 cache slots are over-allocated per
        # request (plan section 6); the 2d prefix-valid write commits exactly
        # commit_lens_r slots (= 1 + correct_len_r <= verify_len_r), so the ragged
        # commit lengths are honoured.
        stride = self.verify_num_draft_tokens
        prefix_lens = batch.seq_lens
        hidden = hidden_strided.view(bs, stride, -1)

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            # MLA fast path: one kernel builds the masked SWA slots + rope positions
            # straight from req_to_token / full_to_swa (folding the
            # assign_extend_cache_locs launch, the translate gather, the commit-mask
            # arange/lt/where chain and the positions add), then the model does
            # projection + pool write exactly as _inject_mla's tail.
            if hidden_strided.numel() == 0:
                return
            inject_layout = BuildCommitInjectLayout.execute(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                prefix_lens=prefix_lens,
                block_pos_offsets=self._block_pos_offsets[:stride],
                full_to_swa_mapping=pool.full_to_swa_index_mapping,
                commit_lens=commit_lens,
                stride=stride,
            )
            with torch.inference_mode():
                self.draft_model.write_target_hidden_kv(
                    main_hidden=hidden.reshape(-1, hidden.shape[-1]),
                    swa_loc=inject_layout.swa_loc,
                    positions=inject_layout.positions,
                    pool=pool,
                )
            return

        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        verify_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + stride,
            batch_size=bs,
            draft_token_num=stride,
            device=self.device,
        )
        verify_cache_loc_2d = verify_cache_loc.view(bs, stride)
        self.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_cache_loc,
            cache_loc_2d=verify_cache_loc_2d,
            positions=positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )
