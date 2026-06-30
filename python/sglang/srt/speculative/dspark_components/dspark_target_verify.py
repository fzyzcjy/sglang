from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dspark_components.dspark_info import (
    RaggedVerifyWindow,
    TargetVerifyResult,
    VerifyWindow,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    apply_logits_adjustments_strided,
    build_ragged_verify_window,
    scatter_compact_to_strided,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class TargetVerifyExecutor:
    def __init__(
        self,
        *,
        target_worker,
        verify_num_draft_tokens: int,
        model_runner,
        kv_injector: TargetHiddenKvInjector,
    ) -> None:
        self.target_worker = target_worker
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self.model_runner = model_runner
        self.kv_injector = kv_injector
        # The target attn backend is constructed after the worker (the scheduler runs
        # init_attention_backends later), so the verify-prep self-add capability is
        # resolved lazily on first verify and cached.
        self._verify_backend_self_adds_seq_lens_cache: Optional[bool] = None

    def _run_target_verify_mode_non_compact(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_ids_2d: torch.Tensor,
        verify_window: VerifyWindow,
        sampling_info,
    ) -> TargetVerifyResult:
        verify_w = self.verify_num_draft_tokens
        positions_2d = verify_window.positions_2d
        verify_cache_loc = verify_window.verify_cache_loc

        verify_input = DFlashVerifyInput(
            draft_token=verify_ids_2d.reshape(-1),
            positions=positions_2d.reshape(-1),
            draft_token_num=verify_w,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch.out_cache_loc = verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        # Verify-prep host seq-len pre-add (B5), backend-polymorphic. The dense
        # FlashInfer / Triton verify backends add the verify window once on the GPU
        # `seq_lens` path, so the worker pre-adds it to the host `seq_lens_cpu`. The
        # V4-family verify backend builds metadata that self-adds the window to both
        # `seq_lens` and `seq_lens_cpu`, so the worker must leave the host length at
        # the prefix here to avoid a double-add; the predicate is resolved from the
        # target attn backend's capability, not the model identity.
        if not self._verify_backend_self_adds_seq_lens():
            if seq_lens_cpu_backup is not None:
                batch.seq_lens_cpu = seq_lens_cpu_backup + verify_w
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            elif draft_input.reserved_seq_lens_cpu is not None:
                batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
                batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=verify_w,
            )

        return TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _commit_verify_hidden(
        self,
        *,
        batch: ScheduleBatch,
        layout: Optional[RaggedVerifyLayout],
        hidden_strided: Optional[torch.Tensor],
        verify_window: VerifyWindow,
        logits_output,
        commit_lens: torch.Tensor,
        bs: int,
        run_compact: bool,
    ) -> None:
        # Commit the accepted tokens' target hidden into the draft KV pool. Every
        # draft (dense + V4) writes the committed prefix back into its real pool here;
        # the MHA-vs-MLA write is dispatched by pool capability inside
        # ``inject_target_hidden``. ``commit_lens`` (= 1 + correct_len)
        # bounds the per-request committed window, and for the MLA single-latent pool
        # selects the accepted bonus slot per request.
        if run_compact:
            self.kv_injector.inject_ragged(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                commit_lens=commit_lens,
                bs=bs,
            )
            return
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        hidden = hidden.view(bs, self.verify_num_draft_tokens, -1)
        self.kv_injector.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_window.verify_cache_loc,
            cache_loc_2d=verify_window.verify_cache_loc_2d,
            positions=verify_window.positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )

    def _run_ragged_target_verify(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        ragged_window: RaggedVerifyWindow,
        sampling_info,
    ) -> TargetVerifyResult:
        # Compact verify forward. The merged decode runner reads
        # spec_info.ragged_verify_layout to select the token-keyed graph and the
        # merged attention backend builds ragged metadata from it
        # (generate_attn_arg_prefill's ragged branch). seq_lens stays at the
        # prefix; the backend adds layout.verify_lens itself.
        #
        # WAR/RAW (invariant 3): the layout's device buffers are built on the
        # forward stream before this prepare_for_verify, so the runner snapshots
        # them inside load_batch ahead of read_done.record(); the generic overlap
        # barrier (is_dflash_or_dspark(), already DSPARK-gated) then
        # serializes the next schedule-stream write against this read. No new
        # event is introduced.
        verify_input = DFlashVerifyInput(
            draft_token=ragged_window.verify_ids,
            positions=ragged_window.positions,
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=layout,
        )
        batch.out_cache_loc = ragged_window.verify_cache_loc
        # prepare_for_verify reads seq_lens off `batch` in place (no param), so
        # snapshot it, overwrite with the verify length for the forward, restore after.
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        # Sync-free verify, keyed on host-mirror availability (not a self-add flag:
        # trtllm_mha is GPU-only yet not self-adding). GPU-only backends
        # (needs_cpu_seq_lens=False) leave seq_lens_cpu None and build kv metadata
        # from GPU seq_lens + device layout.verify_lens; dense backends publish it,
        # so add the real per-req compact window on the host.
        if seq_lens_cpu_backup is not None:
            batch.seq_lens_cpu = seq_lens_cpu_backup + torch.tensor(
                layout.verify_lens_cpu, dtype=seq_lens_cpu_backup.dtype
            )
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        return TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _run_target_verify_mode_compact(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        sampling_info,
    ) -> tuple[TargetVerifyResult, torch.Tensor]:
        # Real-N (full) verify: run the compact total-token forward, then scatter
        # the compact logits and hidden_states back to the bs*(gamma+1) strided
        # layout the accept path + result processor expect. Padded strided rows
        # are filled with 0 and never enter the commit decision (accept is capped
        # to ell_r). Returns the result with strided logits/hidden in place, plus
        # the strided hidden for the ragged KV injection.
        ragged_window = build_ragged_verify_window(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )
        target_verify = self._run_ragged_target_verify(
            batch=batch,
            layout=layout,
            ragged_window=ragged_window,
            sampling_info=sampling_info,
        )
        logits_output = target_verify.logits_output

        compact_logits = logits_output.next_token_logits
        strided_logits = scatter_compact_to_strided(
            compact=compact_logits,
            layout=layout,
            fill_value=0.0,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        apply_logits_adjustments_strided(
            next_token_logits=strided_logits,
            sampling_info=sampling_info,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        logits_output.next_token_logits = strided_logits

        compact_hidden = logits_output.hidden_states
        if compact_hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        hidden_strided = scatter_compact_to_strided(
            compact=compact_hidden,
            layout=layout,
            fill_value=0.0,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        logits_output.hidden_states = hidden_strided
        return target_verify, hidden_strided

    def _verify_backend_self_adds_seq_lens(self) -> bool:
        # True when the target attn backend self-adds the verify window to both
        # seq_lens and seq_lens_cpu (V4), so the worker must NOT pre-add on the host
        # (would double-add). Dense backends add only on the GPU path, so the worker
        # pre-adds the host side. Detected by capability, cached.
        if self._verify_backend_self_adds_seq_lens_cache is None:
            backend = self.target_worker.model_runner.attn_backend
            self._verify_backend_self_adds_seq_lens_cache = hasattr(
                backend, "make_forward_metadata_from_raw_verify"
            )
        return self._verify_backend_self_adds_seq_lens_cache
