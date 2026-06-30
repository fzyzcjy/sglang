from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
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

    def _inject_target_hidden_to_draft_kv(
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
            self._inject_target_hidden_to_draft_kv_mla(
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

    def _inject_target_hidden_to_draft_kv_mla(
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
        # There is no MLA ``set_kv_buffer_prefix_valid`` (that is MHA-only), so the
        # commit is flattened to a per-row flat ``swa_loc``: prefill writes the whole
        # per-request window slots (one latent per prefill token); decode-commit writes
        # exactly the accepted bonus slot per request (``commit_len[r]-1`` within the
        # verify window), so the target hidden, the swa slot, and the position are all
        # gathered to one row per request and stay aligned for the per-stage write.
        # ``translate_loc_from_full_to_swa`` is an alloc-time state lookup -- the slots
        # are already allocated here (assign_extend_cache_locs / _alloc_verify_window
        # ran first) and padded rows map to -1, so we translate only AFTER allocation
        # and never read a default-0 SWA slot.
        if commit_lens is not None and cache_loc_2d is not None:
            bs = cache_loc_2d.shape[0]
            row_ids = torch.arange(bs, device=cache_loc.device)
            commit_idx = (commit_lens.to(torch.long) - 1).clamp_min(0)
            write_full_loc = cache_loc_2d[row_ids, commit_idx]
            write_positions = positions.view(bs, -1)[row_ids, commit_idx]
            write_hidden = target_hidden.view(bs, -1, target_hidden.shape[-1])[
                row_ids, commit_idx
            ]
        else:
            write_full_loc = cache_loc
            write_positions = positions
            write_hidden = target_hidden
        swa_loc = pool.translate_loc_from_full_to_swa(write_full_loc).to(torch.int32)

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=write_hidden,
                swa_loc=swa_loc,
                positions=write_positions,
                pool=pool,
            )

    def _inject_ragged_hidden_to_draft_kv(
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
        hidden = hidden_strided.view(bs, stride, -1)
        self._inject_target_hidden_to_draft_kv(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_cache_loc,
            cache_loc_2d=verify_cache_loc_2d,
            positions=positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )
