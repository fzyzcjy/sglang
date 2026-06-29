import logging
from typing import Optional

import msgspec
import torch

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    apply_dflash_verify_logits_adjustments,
    build_dflash_verify_target_probs,
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_input_v2,
)
from sglang.srt.speculative.dspark_utils import (
    dspark_gamma_from_num_draft_tokens,
    parse_dspark_draft_config,
)
from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func

logger = logging.getLogger(__name__)


class _VerifyWindow(msgspec.Struct, frozen=True):
    positions_2d: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_cache_loc_2d: torch.Tensor


class _TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool


class DSparkWorkerV2(BaseSpecWorker):
    """DSpark dense speculative decoding worker (spec-v2, static verify).

    Drives both overlap and non-overlap scheduling like DFlash, but the draft is
    a semi-AR Markov block: a ``gamma``-slot draft block produces ``s_0..s_{γ-1}``
    by a serial Markov loop (eager; not capturable), and the target verifies the
    ``gamma+1`` window ``[anchor, s_0..s_{γ-1}]``. Lossless: greedy verify is
    argmax-match (DFlash rule), sampling verify is rejection sampling via the
    chain kernel fed the real Markov draft distribution. Model-agnostic over the
    draft backbone (Qwen3 / Gemma4 dense). Supports tp_size > 1: the base logits
    are TP all-gathered to a full, per-rank-identical vocab, after which the bare
    serial Markov sampling and accept coin auto-align across ranks via the same
    "same seed + SPMD lockstep RNG" the main sampler relies on.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        # Draft runner (separate KV cache + attention backend), shared with DFlash.
        bundle = build_draft_tp_worker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_model_config=target_worker.model_runner.model_config,
            algo_label="DSPARK",
        )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model

        dspark_config = parse_dspark_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark draft requires markov_rank > 0; got "
                f"markov_rank={dspark_config.markov_rank}."
            )
        # speculative_num_draft_tokens is the verify window (= gamma + 1). The draft
        # block is exactly gamma slots (plan §2). Both must stay consistent.
        if server_args.speculative_num_draft_tokens is None:
            gamma = int(dspark_config.resolve_gamma(default=None) or 0)
            if gamma < 1:
                raise ValueError(
                    "DSpark could not resolve gamma from the draft config and "
                    "speculative_num_draft_tokens is unset."
                )
            self.gamma = gamma
        else:
            self.gamma = dspark_gamma_from_num_draft_tokens(
                int(server_args.speculative_num_draft_tokens)
            )
            config_gamma = dspark_config.resolve_gamma(default=None)
            if config_gamma is not None and int(config_gamma) != self.gamma:
                logger.warning(
                    "DSpark gamma mismatch: using gamma=%s (from "
                    "speculative_num_draft_tokens=%s) but draft config block_size=%s.",
                    self.gamma,
                    server_args.speculative_num_draft_tokens,
                    config_gamma,
                )
        self.verify_num_draft_tokens = self.gamma + 1
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens

        if dspark_config.mask_token_id is None:
            raise ValueError(
                "DSpark requires mask_token_id to be set in the draft model config."
            )
        self._mask_token_id = int(dspark_config.mask_token_id)
        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if self._mask_token_id >= vocab_size:
            raise ValueError(
                f"DSpark mask_token_id={self._mask_token_id} is outside the target "
                f"vocab size {vocab_size}."
            )

        if self.tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, mask_token_id=%s, "
                "markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.gamma), device=self.device
        )

        # Optional confidence relay. The head and its relay are inert (no
        # buffers, no events, no compute) when the draft model lacks a
        # confidence head, so the a+b lossless decode path is unchanged.
        self._confidence_head = getattr(self.draft_model, "confidence_head", None)
        self._confidence_buf: Optional[torch.Tensor] = None
        self._confidence_cpu_pinned: Optional[torch.Tensor] = None
        self._confidence_ready = None
        self._confidence_d2h_stream = None
        if self._confidence_head is not None and self.tp_rank == 0:
            logger.info(
                "DSpark confidence head enabled (with_markov=%s); confidence is "
                "relayed on an independent stream/event and is advisory only.",
                getattr(self._confidence_head, "with_markov", True),
            )

    @property
    def carries_confidence(self) -> bool:
        return self._confidence_head is not None

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        # The serial Markov loop is eager (it cannot be captured); only the draft
        # backbone forward may be graph-captured. No in-graph draft sampler.
        capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        self._draft_worker.init_cuda_graphs(
            capture_decode_cuda_graph=capture_decode_cuda_graph
        )

    def clear_cache_pool(self):
        pass

    def _inject_target_hidden_to_draft_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
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

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(target_hidden)
            pool = self.draft_model_runner.token_to_kv_pool
            for layer in self.draft_model.layers:
                attn = layer.self_attn
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(positions, k)
                v = attn.apply_v_norm(v)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                if cache_loc_2d is not None and commit_lens is not None:
                    pool.set_kv_buffer_prefix_valid(
                        attn.attn,
                        cache_loc_2d,
                        commit_lens,
                        k,
                        v,
                        attn.attn.k_scale,
                        attn.attn.v_scale,
                    )
                else:
                    pool.set_kv_buffer(
                        attn.attn,
                        cache_loc,
                        k,
                        v,
                        attn.attn.k_scale,
                        attn.attn.v_scale,
                    )

    def _make_next_draft_input(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
    ) -> DFlashDraftInputV2:
        return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)

    def _compute_base_logits(
        self, *, draft_hidden: torch.Tensor, lm_head
    ) -> torch.Tensor:
        # ParallelLMHead.forward is intentionally disabled; compute base logits
        # directly from the head weight (the local vocab shard under TP), then
        # TP all-gather to full vocab so every rank holds an identical, full-vocab
        # tensor, mirroring LogitsProcessor._get_logits (logits_processor.py:858).
        weight = lm_head.weight
        hidden = draft_hidden
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        local_logits = torch.matmul(hidden, weight.T)
        full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
        org_vocab_size = int(lm_head.org_vocab_size)
        return full_logits[..., :org_vocab_size]

    def _sample_draft_block(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        draft_hidden: torch.Tensor,
        sampling_info,
    ):
        markov_head = self.draft_model.markov_head
        is_greedy = sampling_info is None or sampling_info.is_all_greedy

        if is_greedy:

            def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                return torch.argmax(step_logits, dim=-1)

            temperatures = None
        else:
            temperatures = (
                sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
            )

            def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                return torch.multinomial(probs, num_samples=1).squeeze(-1)

        draft_tokens, corrected_logits = markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor_tokens,
            hidden_states=draft_hidden,
            sampler=sampler,
        )
        return draft_tokens, corrected_logits, is_greedy, temperatures

    def _build_markov_embed_stack(
        self,
        *,
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # Per-step prev tokens fed to the Markov head during the serial loop:
        # step 0 sees the anchor, step i (>0) sees the previously sampled token,
        # i.e. prev_seq = [anchor, s_0, ..., s_{gamma-2}] (the chapter's
        # off-by-one). markov_embed[:, i] = markov_w1(prev_seq[:, i]).
        markov_head = self.draft_model.markov_head
        prev_seq = torch.cat(
            [anchor_tokens.view(-1, 1), draft_tokens[:, : self.gamma - 1]], dim=1
        )
        return markov_head.get_prev_embeddings(prev_seq)

    def _compute_confidence(
        self,
        *,
        draft_hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # Dense DSpark: the confidence head consumes the same post-norm draft
        # hidden that feeds base_logits (DeepSpec qwen3 modeling feeds the
        # post-norm output_hidden to both lm_head and the confidence head). For
        # with_markov heads it also takes the per-step markov_embed stack.
        confidence_head = self._confidence_head
        assert confidence_head is not None
        if confidence_head.with_markov:
            markov_embed_stack = self._build_markov_embed_stack(
                anchor_tokens=anchor_tokens, draft_tokens=draft_tokens
            )
        else:
            markov_embed_stack = None
        confidence_raw = confidence_head(draft_hidden, markov_embed_stack)
        # STS calibration is identity in the MVP (no reference); the head logit
        # is mapped to (0, 1) by sigmoid. Losslessness does not depend on the
        # calibration quality, only on the scheduler being non-anticipating.
        confidence = torch.sigmoid(confidence_raw.float())
        assert bool(
            ((confidence > 0) & (confidence < 1)).all()
        ), "DSpark confidence must lie in the open interval (0, 1)."
        return confidence

    def _ensure_confidence_relay_buffers(self, *, confidence: torch.Tensor) -> None:
        if self._confidence_buf is not None:
            return
        device_module = torch.get_device_module(self.device)
        req_pool_size = int(self.model_runner.req_to_token_pool.req_to_token.shape[0])
        self._confidence_buf = torch.empty(
            (req_pool_size, self.gamma),
            dtype=confidence.dtype,
            device=self.device,
        )
        self._confidence_cpu_pinned = torch.empty(
            (req_pool_size, self.gamma),
            dtype=confidence.dtype,
            pin_memory=True,
        )
        self._confidence_ready = device_module.Event()
        self._confidence_d2h_stream = device_module.Stream()

    def _stash_confidence(
        self,
        *,
        req_pool_indices: torch.Tensor,
        confidence: torch.Tensor,
    ) -> None:
        # Write confidence into the relay buffer on the forward stream, then
        # record the independent confidence_ready event AFTER the write. The
        # D2H copy is gated on this event (not on publish_ready, which is
        # recorded earlier mid-worker), and runs on its own stream so the
        # critical path issues no synchronize().
        self._ensure_confidence_relay_buffers(confidence=confidence)
        self._confidence_buf[req_pool_indices] = confidence
        self._confidence_ready.record()

    def pull_confidence_history(self) -> Optional[torch.Tensor]:
        # Non-blocking: only kick the D2H copy when the forward-stream write has
        # already completed (event.query()); otherwise reuse the stale CPU copy.
        # Never synchronize on the critical path.
        if self._confidence_ready is None or self._confidence_cpu_pinned is None:
            return None
        if self._confidence_ready.query():
            device_module = torch.get_device_module(self.device)
            with device_module.stream(self._confidence_d2h_stream):
                self._confidence_cpu_pinned.copy_(
                    self._confidence_buf, non_blocking=True
                )
        return self._confidence_cpu_pinned

    def _accept_greedy(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
    ):
        bs = candidates.shape[0]
        target_predict = torch.argmax(target_logits, dim=-1).view(
            bs, self.verify_num_draft_tokens
        )
        correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        return correct_len, bonus

    def _accept_sampling(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
    ):
        bs = candidates.shape[0]
        device = candidates.device
        gamma = self.gamma
        target_probs = build_dflash_verify_target_probs(
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            draft_token_num=self.verify_num_draft_tokens,
            bs=bs,
            max_top_k=draft_input.max_top_k,
            uniform_top_k_value=draft_input.uniform_top_k_value,
        )
        (
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            predicts,
            accept_index,
            accept_token_num,
        ) = _get_or_create_chain_verify_buffers(
            bs=bs,
            draft_token_num=self.verify_num_draft_tokens,
            device=device,
        )
        uniform_samples = torch.rand((bs, gamma), dtype=torch.float32, device=device)
        uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=device)
        candidates_i64 = candidates.to(torch.int64)
        chain_speculative_sampling_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates_i64,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
        correct_len = accept_token_num
        row_ids = torch.arange(bs, dtype=torch.long, device=device)
        accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
        bonus = predicts[accept_pos].to(torch.int64)
        return correct_len, bonus

    def _build_out_tokens(
        self,
        *,
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
    ) -> torch.Tensor:
        bs = draft_tokens.shape[0]
        out_tokens = torch.empty(
            (bs, self.verify_num_draft_tokens),
            dtype=torch.int64,
            device=draft_tokens.device,
        )
        out_tokens[:, : self.gamma].copy_(draft_tokens)
        out_tokens[:, self.gamma].fill_(0)
        out_tokens.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
        return out_tokens

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_prefill(batch, on_publish)

        return self._forward_decode(batch, on_publish)

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(batch)
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        device = next_token_ids.device
        ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, device=device
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        self._inject_target_hidden_to_draft_kv(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
        )
        logits_output.hidden_states = None

        batch_output.next_draft_input = self._make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = self._make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _alloc_verify_window(
        self,
        *,
        batch: ScheduleBatch,
        bs: int,
        device: str,
    ) -> _VerifyWindow:
        prefix_lens = batch.seq_lens
        verify_w = self.verify_num_draft_tokens
        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        verify_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + verify_w,
            batch_size=bs,
            draft_token_num=verify_w,
            device=device,
        )
        verify_cache_loc_2d = verify_cache_loc.view(bs, verify_w)
        return _VerifyWindow(
            positions_2d=positions_2d,
            verify_cache_loc=verify_cache_loc,
            verify_cache_loc_2d=verify_cache_loc_2d,
        )

    def _run_draft_block_forward(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: _VerifyWindow,
        bs: int,
        device: str,
        embed_module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        noise_embedding = embed_module(draft_block_ids)
        draft_input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        if batch.seq_lens_cpu is not None:
            draft_seq_lens_cpu = batch.seq_lens_cpu + gamma
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_input.reserved_seq_lens_cpu is not None:
            draft_seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
        else:
            draft_seq_lens_cpu = prefix_lens.to("cpu", dtype=torch.int32)
            draft_seq_lens_sum = int(prefix_lens.sum().item())

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
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_forward_batch)
        draft_hidden = draft_out.logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DSpark draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, gamma, -1)
        return draft_block_ids, draft_hidden

    def _run_target_verify(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_ids_2d: torch.Tensor,
        verify_window: _VerifyWindow,
        sampling_info,
    ) -> _TargetVerifyResult:
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

        return _TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_decode(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            return self._decode_idle_result(on_publish=on_publish)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DSpark requires the target model to expose `lm_head` with `weight`."
            )

        verify_window = self._alloc_verify_window(batch=batch, bs=bs, device=device)

        draft_block_ids, draft_hidden = self._run_draft_block_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=embed_module,
        )

        base_logits = self._compute_base_logits(
            draft_hidden=draft_hidden, lm_head=lm_head
        )
        sampling_info = batch.sampling_info
        draft_tokens, corrected_logits, is_greedy, temperatures = (
            self._sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=draft_block_ids[:, 0],
                draft_hidden=draft_hidden,
                sampling_info=sampling_info,
            )
        )

        if self._confidence_head is not None:
            confidence = self._compute_confidence(
                draft_hidden=draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
            )
            self._stash_confidence(
                req_pool_indices=batch.req_pool_indices,
                confidence=confidence,
            )

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        target_verify = self._run_target_verify(
            batch=batch,
            draft_input=draft_input,
            verify_ids_2d=verify_ids_2d,
            verify_window=verify_window,
            sampling_info=sampling_info,
        )
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        if is_greedy:
            correct_len, bonus = self._accept_greedy(
                candidates=verify_ids_2d,
                target_logits=logits_output.next_token_logits,
            )
        else:
            draft_probs = torch.softmax(
                corrected_logits.float() / temperatures[:, None, None], dim=-1
            )
            correct_len, bonus = self._accept_sampling(
                candidates=verify_ids_2d,
                target_logits=logits_output.next_token_logits,
                draft_probs=draft_probs,
                sampling_info=sampling_info,
                draft_input=draft_input,
            )

        commit_lens = correct_len.to(torch.int32) + 1
        out_tokens = self._build_out_tokens(
            draft_tokens=draft_tokens, correct_len=correct_len, bonus=bonus
        )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        hidden = hidden.view(bs, self.verify_num_draft_tokens, -1)
        self._inject_target_hidden_to_draft_kv(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_window.verify_cache_loc,
            cache_loc_2d=verify_window.verify_cache_loc_2d,
            positions=verify_window.positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )
        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=new_seq_lens,
        )
