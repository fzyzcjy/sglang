from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_draft import (
    resolve_greedy_mask,
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_info import (
    DraftBlockResult,
    DraftForwardResult,
    DraftProposal,
    VerifyWindow,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context


class DraftBlockProposer:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        gamma: int,
        mask_token_id: int,
        draft_block_spec_info,
        dp_moe_sync: bool = False,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.gamma = gamma
        self._mask_token_id = mask_token_id
        self._draft_block_spec_info = draft_block_spec_info
        self._draft_sampler = None
        # dsv4 (MoE) draft under dp attention: the draft forward runs the shared
        # MoE-DP gather, so its manually-built ForwardBatch must carry DP metadata
        # (global_num_tokens = per-rank bs * gamma). The dense draft runs replicated
        # in the attention-TP context and must NOT get this (it would trigger an
        # unwanted MLP sync in a size-1 group), so it stays False.
        self._dp_moe_sync = dp_moe_sync

    def _base_logits_context(self):
        """Neutralize compute_base_logits' vocab all-gather under dsv4 (MoE) DP.

        gather_and_crop_vocab all-gathers over the global TP group, but under
        --enable-dp-lm-head + attn_tp==1 the lm_head is full-vocab per rank and each DP
        rank holds different tokens, so a global gather is both wrong and deadlocks:
        idle DP groups never call compute_base_logits, so busy ranks block forever on the
        collective. Patch _TP to the size-1 attn-TP group so the gather is a per-rank
        no-op (mirrors the dense draft, whose whole propose runs in this context). The
        draft MODEL forward stays outside this context -- its MoE gather needs the real
        global TP group and is matched by the idle group's run_idle_participation."""
        if self._dp_moe_sync:
            return draft_tp_context(get_attention_tp_group())
        return nullcontext()

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        target_model,
        sampling_info,
    ) -> DraftProposal:
        # Single orchestration for every draft (dense + V4): run the draft block forward
        # on the real pool, then let the MODEL produce its base logits and the worker
        # reshape to ``[bs, gamma, vocab]`` for the serial Markov block. Base-logit
        # provenance is no longer a worker concern: every draft model owns
        # ``compute_base_logits(raw_hidden)`` (dense: weight-dtype matmul; dsv4: hc_head
        # collapse -> norm -> fp32 F.linear), and the worker calls it once. The
        # ``[bs, gamma, vocab]`` reshape is MANDATORY: markov ``sample_block`` reads
        # ``shape[:2]`` as ``(bs, proposal_len)`` and indexes ``[:, step, :]``.
        embed_module = target_model.get_input_embeddings()
        fwd = self._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=embed_module,
        )
        draft_block_ids = fwd.draft_block_ids

        draft_sampler = self._draft_sampler
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        if draft_sampler is not None and fwd.can_run_graph and all_greedy:
            # Captured greedy proposal: compute_base_logits + Markov argmax already ran
            # in the draft cuda graph and wrote draft_sampler.out. Read it instead of
            # the eager matmul + serial loop. corrected_logits unused on the greedy
            # accept path, so None; greedy_mask/temperatures are cheap.
            if sampling_info is None:
                temperatures = torch.ones(bs, dtype=torch.float32, device=device)
            else:
                temperatures = (
                    sampling_info.temperatures.view(-1)
                    .to(torch.float32)
                    .clamp_min(1e-5)
                )
            draft_block = DraftBlockResult(
                draft_tokens=draft_sampler.out[: bs * self.gamma].view(bs, self.gamma),
                corrected_logits=None,
                greedy_mask=resolve_greedy_mask(
                    bs=bs, sampling_info=sampling_info, device=device
                ),
                temperatures=temperatures,
            )
        else:
            with self._base_logits_context():
                base_logits = self.draft_model.compute_base_logits(fwd.raw_hidden).view(
                    bs, self.gamma, -1
                )
            draft_block = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=draft_block_ids[:, 0],
                draft_hidden=fwd.draft_hidden_3d,
                sampling_info=sampling_info,
                markov_head=self.draft_model.markov_head,
                device=device,
            )
        return DraftProposal(
            draft_block_ids=draft_block_ids,
            draft_block=draft_block,
            draft_hidden=fwd.draft_hidden_3d,
        )

    def run_idle_participation(self, batch: ScheduleBatch) -> None:
        """dsv4 (MoE) draft under DP attention: run a 0-token draft forward so an idle
        attention-DP group joins the draft's dp_gather (busy ranks gather bs*gamma
        tokens across DP). No-op unless dp_moe_sync. Scale global_num_tokens by gamma
        exactly like the busy draft; the idle rank's own entry is already 0, so it
        contributes 0 rows to the gather. Output discarded."""
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        device = self.draft_model_runner.device
        empty_long = torch.empty((0,), dtype=torch.int64, device=device)
        idle_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=empty_long,
            req_pool_indices=empty_long,
            seq_lens=empty_long,
            out_cache_loc=empty_long,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            positions=empty_long,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_dp_moe_sync_metadata(idle_batch, batch)
        with torch.inference_mode():
            self.draft_model_runner.forward(idle_batch)

    def _run_forward(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        embed_module,
    ) -> DraftForwardResult:
        gamma = self.gamma
        prefix_lens = batch.seq_lens
        positions_2d = verify_window.positions_2d
        verify_cache_loc_2d = verify_window.verify_cache_loc_2d

        draft_block_ids = torch.full(
            (bs, gamma), int(self._mask_token_id), dtype=torch.long, device=device
        )
        draft_block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        draft_positions = positions_2d[:, :gamma].reshape(-1)
        draft_cache_loc = verify_cache_loc_2d[:, :gamma].reshape(-1)

        # Embedding ownership is a capability, not a model identity: a draft model
        # that hc-expands its own embedding exposes ``forward_embed`` and takes the
        # flat ``input_ids`` only (V4 mHC needs the [N, hc, d] expand the worker can't
        # build); dense backbones take the precomputed flat ``input_embeds`` from the
        # target embedding, byte-identical to before.
        draft_owns_embed = hasattr(self.draft_model, "forward_embed")
        draft_input_embeds: Optional[torch.Tensor] = None
        if not draft_owns_embed:
            noise_embedding = embed_module(draft_block_ids)
            draft_input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        if batch.seq_lens_cpu is not None:
            draft_seq_lens_cpu = batch.seq_lens_cpu + gamma
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_input.reserved_seq_lens_cpu is not None:
            draft_seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
        else:
            raise RuntimeError("DSpark decode expected batch.seq_lens_cpu, got None")

        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=draft_block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=draft_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=draft_seq_lens_cpu,
            positions=draft_positions,
            input_embeds=draft_input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_dp_moe_sync_metadata(draft_forward_batch, batch)
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_forward_batch)
        logits_output = draft_out.logits_output
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            raise RuntimeError("DSpark draft model returned no hidden states.")
        # raw_hidden is the model's un-reshaped backbone hidden (dense 2-D [bs*gamma, d];
        # dsv4 3-D [bs*gamma, hc, d]); compute_base_logits consumes it as-is (the dsv4
        # hc-collapse needs the [N, hc, d] layout, NOT this view). draft_hidden_3d is the
        # dense markov / dense confidence input; dsv4 ignores it.
        draft_hidden_3d = raw_hidden.view(bs, gamma, -1)
        return DraftForwardResult(
            draft_block_ids=draft_block_ids,
            raw_hidden=raw_hidden,
            draft_hidden_3d=draft_hidden_3d,
            can_run_graph=draft_out.can_run_graph,
        )

    def _fill_dp_moe_sync_metadata(
        self, forward_batch: ForwardBatch, batch: ScheduleBatch
    ) -> None:
        """dsv4 (MoE) draft under DP attention: give the hand-built draft ForwardBatch
        the DP MLP-sync metadata that ``ForwardBatch.init_new`` would derive (its "For
        MLP sync" block), so the forward enters ``prepare_mlp_sync_batch`` and the
        shared MoE-DP gather sizes its buffer correctly. The scheduler's all-gathered
        per-rank bs is scaled by the draft-block spec_info's declared coefficient
        (``draft_token_num`` == gamma, matching the bs*gamma draft tokens per rank --
        NOT the graph MAX_LEN uniform value), mirroring EAGLE's init_new path.
        prepare_mlp_sync_batch fills dp_padding_mode / global_dp_buffer_len from these.
        No-op unless dp_moe_sync (the dense draft runs replicated in the attn-TP
        context and must never get DP metadata)."""
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        gnt, gnt_logprob = (
            self._draft_block_spec_info.get_spec_adjusted_global_num_tokens(batch)
        )
        device = self.draft_model_runner.device
        forward_batch.global_num_tokens_cpu = gnt
        forward_batch.global_num_tokens_for_logprob_cpu = gnt_logprob
        forward_batch.global_num_tokens_gpu = torch.tensor(gnt, dtype=torch.int64).to(
            device, non_blocking=True
        )
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            gnt_logprob, dtype=torch.int64
        ).to(device, non_blocking=True)
        forward_batch.can_run_dp_cuda_graph = batch.can_run_dp_cuda_graph
