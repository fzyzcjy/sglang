import logging
import os
from typing import Optional

import msgspec
import torch

from sglang.srt.distributed import get_tp_group
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
from sglang.srt.speculative.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
)
from sglang.srt.speculative.dspark_sps_table import (
    SpsCostTable,
    load_sps_table_from_path,
)
from sglang.srt.speculative.dspark_utils import (
    dspark_gamma_from_num_draft_tokens,
    parse_dspark_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func

logger = logging.getLogger(__name__)


class _VerifyWindow(msgspec.Struct, frozen=True):
    positions_2d: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_cache_loc_2d: torch.Tensor


class _RaggedVerifyWindow(msgspec.Struct, frozen=True):
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_ids: torch.Tensor
    seq_lens_cpu: torch.Tensor


class _TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool


class _DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: torch.Tensor
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor


class _DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: _DraftBlockResult
    draft_hidden: Optional[torch.Tensor]


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

        # Capability-driven draft polymorphism (no model-identity branches). A
        # draft model that exposes ``forward_spec`` owns its whole draft block
        # (embed -> stages -> Markov head finish) and its own KV ring, so the
        # worker delegates the draft forward, the target-hidden injection, and the
        # confidence to the model instead of orchestrating them per-layer. Dense
        # backbones (qwen3 / gemma4) expose none of these, so all flags are False
        # and every dense path below is byte-identical.
        self._draft_owns_block_forward = hasattr(self.draft_model, "forward_spec")
        self._draft_owns_kv_injection = hasattr(
            self.draft_model, "inject_target_hidden"
        )
        self._draft_owns_confidence = hasattr(self.draft_model, "last_confidence")
        # The target attn backend is constructed after the worker (the scheduler runs
        # init_attention_backends later), so the verify-prep self-add capability is
        # resolved lazily on first verify and cached.
        self._verify_backend_self_adds_seq_lens_cache: Optional[bool] = None
        if self._draft_owns_block_forward and hasattr(
            self.draft_model, "attach_shared_modules"
        ):
            target_model = self.target_worker.model_runner.model
            self.draft_model.attach_shared_modules(
                embed_tokens=self._resolve_target_embed_tokens(target_model),
                lm_head=target_model.lm_head,
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

        # Ragged-verify mode and the confidence prefix scheduler. The scheduler is
        # inert (None) unless the mode is cap-accept/compact AND a confidence head is
        # present, so the static-mode / no-head path is byte-identical to the static
        # uniform-gamma worker. cap-accept runs the full bs*(gamma+1) window and
        # only caps accept at per-request ell_r; compact (real-N) is not wired here.
        self._ragged_verify_mode = read_ragged_verify_mode()
        self._verify_scheduler: Optional[ConfidencePrefixScheduler] = None
        if (
            self._ragged_verify_mode is not RaggedVerifyMode.STATIC
            and self._confidence_head is not None
        ):
            self._verify_scheduler = ConfidencePrefixScheduler(
                sps_table=self._build_sps_cost_table(),
                cfg=DSparkScheduleConfig(gamma=self.gamma),
            )
            if self.tp_rank == 0:
                logger.info(
                    "DSpark ragged-verify scheduler enabled (mode=%s).",
                    self._ragged_verify_mode.value,
                )

    def _build_sps_cost_table(self) -> SpsCostTable:
        # Load a pre-profiled table when a path is given, else a flat constant-SPS
        # table (budget = verify-all-up-to-max). Flat is the inert default for
        # cap-accept, which has zero throughput gain; the GPU profiler hook lands
        # with the compact real-N path.
        sps_table_path = os.environ.get("SGLANG_DSPARK_SPS_TABLE_PATH")
        if sps_table_path:
            return load_sps_table_from_path(sps_table_path)
        max_batch_tokens = max(
            1,
            int(self.server_args.max_running_requests or 1)
            * self.verify_num_draft_tokens,
        )
        return SpsCostTable(
            sample_batch_tokens=[1],
            sample_steps_per_sec=[1.0],
            max_batch_tokens=max_batch_tokens,
        )

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

    def _resolve_target_embed_tokens(self, target_model):
        # The V4 draft forward reuses the target's input embedding module. Some
        # target models expose it via ``get_input_embeddings`` on the outer module;
        # others (DeepSeek-V4) expose it on the inner ``model`` submodule. Resolve
        # whichever the target provides so the attach works without a model-identity
        # branch.
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

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
        start_pos: int = 0,
        is_prefill: bool = False,
    ) -> None:
        # Capability dispatch (B4): a draft model that owns its KV ring takes the
        # whole injection (project -> per-stage proj -> norm -> rope -> ring store)
        # via ``inject_target_hidden``; the worker only hands it the captured target
        # hidden plus the sliding-window position / commit signal. Dense backbones
        # lack ``inject_target_hidden`` so they fall through to the per-layer pool
        # loop below, which is byte-identical to before.
        if self._draft_owns_kv_injection:
            self._inject_target_hidden_via_model(
                target_hidden=target_hidden,
                start_pos=start_pos,
                is_prefill=is_prefill,
            )
            return
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

    def _inject_target_hidden_via_model(
        self,
        *,
        target_hidden: torch.Tensor,
        start_pos: int,
        is_prefill: bool,
    ) -> None:
        if target_hidden is None or target_hidden.numel() == 0:
            return
        target_hidden = target_hidden.to(
            device=self.model_runner.device, non_blocking=True
        )
        with torch.inference_mode():
            self.draft_model.inject_target_hidden(
                main_hidden=target_hidden,
                start_pos=start_pos,
                is_prefill=is_prefill,
            )

    def _make_next_draft_input(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        draft_input = make_draft_input_v2(
            bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens
        )
        # Dense relays only bonus_tokens + new_seq_lens (the legacy Eagle-shaped
        # hidden_states stays the empty placeholder, byte-identical). A draft model
        # that owns its block forward needs the next step's anchor target hidden, so
        # the V4 path threads it through the otherwise-unused relay slot.
        if hidden_states is not None:
            draft_input.hidden_states = hidden_states
        return draft_input

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

    def _resolve_greedy_mask(self, *, bs: int, sampling_info) -> torch.Tensor:
        # Per-request greedy mask (review M4). A row is greedy iff top_k <= 1,
        # which mirrors SamplingBatchInfo.is_all_greedy (= all rows greedy). In a
        # mixed batch the previous batch-level branch forced greedy rows onto the
        # rejection-sampling accept path; the per-row mask lets greedy rows use
        # argmax-match accept and sampling rows use the chain kernel.
        if sampling_info is None:
            return torch.ones(bs, dtype=torch.bool, device=self.device)
        return (sampling_info.top_ks <= 1).view(-1)

    def _sample_draft_block(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        draft_hidden: torch.Tensor,
        sampling_info,
    ) -> _DraftBlockResult:
        markov_head = self.draft_model.markov_head
        bs = base_logits.shape[0]
        greedy_mask = self._resolve_greedy_mask(bs=bs, sampling_info=sampling_info)
        any_sampling = bool((~greedy_mask).any())

        if sampling_info is None:
            temperatures = torch.ones(bs, dtype=torch.float32, device=self.device)
        else:
            temperatures = (
                sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
            )

        if not any_sampling:
            # All-greedy batch: argmax only. Crucially this must NOT draw random
            # numbers (no torch.multinomial), otherwise the RNG stream diverges
            # from the off/cutoff/a+b path and breaks byte-identical losslessness.
            def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                return torch.argmax(step_logits, dim=-1)

        else:

            def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                # Per-row mixed sampling: greedy rows take argmax, sampling rows
                # draw from the temperature-scaled softmax. torch.where selects per
                # row so a mixed batch keeps each request's own draft distribution.
                # With at least one sampling row this matches the all-sampling RNG
                # draw count (one multinomial per step), so the all-sampling path
                # stays byte-identical.
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

        draft_tokens, corrected_logits = markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor_tokens,
            hidden_states=draft_hidden,
            sampler=sampler,
        )
        return _DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=corrected_logits,
            greedy_mask=greedy_mask,
            temperatures=temperatures,
        )

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
        # sigmoid is mathematically in (0, 1) but saturates to exactly 0.0 / 1.0 at
        # fp32 precision for large-magnitude logits, so the sanity check uses the
        # closed interval (matching the V4 draft). The value is advisory only;
        # losslessness does not depend on it.
        assert bool(
            ((confidence >= 0) & (confidence <= 1)).all()
        ), "DSpark confidence must lie in [0, 1]."
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

    def _maybe_schedule_ragged_layout(
        self,
        *,
        req_pool_indices: torch.Tensor,
        device: torch.device,
    ) -> Optional[RaggedVerifyLayout]:
        # Gate: STATIC -> None (uniform path). CAP_ACCEPT/COMPACT build a ragged
        # layout from n-2-frozen verify_lens; COMPACT additionally token-keys the
        # graph and scatter-packs the verify window.
        if self._ragged_verify_mode is RaggedVerifyMode.STATIC:
            return None
        verify_lens = self._schedule_verify_lens(
            req_pool_indices=req_pool_indices, device=device
        )
        if verify_lens is None:
            return None
        verify_lens_cpu = verify_lens.to("cpu").tolist()
        grid = self._verify_layout_grid(verify_lens_cpu=verify_lens_cpu)
        return RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=device,
            grid=grid,
        )

    def _schedule_verify_lens(
        self,
        *,
        req_pool_indices: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        # Shared cutoff/full schedule: derive per-request verify_lens (= 1 + ell_r)
        # from the n-2-frozen confidence history, broadcast from rank 0 across the
        # TP group for cross-rank shape consistency. Returns None (-> uniform full
        # block) whenever the confidence history is not yet available.
        if self._verify_scheduler is None:
            return None
        confidence_history = self.pull_confidence_history()
        if confidence_history is None:
            return None

        req_pool_indices_cpu = req_pool_indices.to("cpu", dtype=torch.int64)
        confidence = confidence_history[req_pool_indices_cpu].to(self.device)
        survival_probs = torch.cumprod(confidence.to(torch.float32), dim=1)

        self._verify_scheduler.update_budget_from_history(
            history_survival_probs=survival_probs
        )
        verify_lens = self._verify_scheduler.compute_verify_lens(
            survival_probs=survival_probs
        ).to(device=device, dtype=torch.int32)

        if self.server_args.tp_size > 1:
            # GroupCoordinator.broadcast maps the local src rank to a global rank
            # via self.ranks[src]; passing src=0 to dist.broadcast directly would
            # use a GLOBAL rank and broadcast from the wrong source when the TP
            # group's ranks[0] != 0 (review A6). Use the group wrapper so rank 0 is
            # the TP-local source. DP-attention follow-up must switch to
            # get_attention_tp_group().broadcast (mirror sampler.py); c-v1 is
            # TP-only.
            get_tp_group().broadcast(verify_lens, src=0)

        return verify_lens

    def _verify_layout_grid(self, *, verify_lens_cpu: list[int]) -> list[int]:
        # COMPACT aligns the grid to the decode runner's token buckets so
        # graph_num_tokens lands on a captured tier; else [total] suffices.
        total = sum(verify_lens_cpu)
        if self._ragged_verify_mode is not RaggedVerifyMode.COMPACT:
            return [total]
        capture_num_tokens = self._ragged_capture_num_tokens()
        if capture_num_tokens is None:
            return [total]
        return capture_num_tokens

    def _ragged_capture_num_tokens(self) -> Optional[list[int]]:
        # The decode runner owns the token-keyed capture grid (Plan B). Read it so
        # graph_num_tokens = round_up_grid(total, capture_num_tokens) selects a
        # captured graph. Returns None when no token-keyed graph exists (e.g.
        # cuda-graph disabled or an eager runner without ragged capture), in which
        # case full runs eager on exactly `total`. The runner type is polymorphic
        # (graph runner vs eager runner), so ragged_verify_mode may be absent.
        runner = self.model_runner.decode_cuda_graph_runner
        if runner is None or not getattr(runner, "ragged_verify_mode", False):
            return None
        return runner.capture_num_tokens

    def _cap_correct_len(
        self,
        *,
        correct_len: torch.Tensor,
        layout: RaggedVerifyLayout,
    ) -> torch.Tensor:
        # Cutoff-only cap: commit at most ell_r = verify_len - 1 correct drafts per
        # request. Capping accept is lossless -- fewer correctly-verified drafts are
        # committed and the bonus (recomputed by callers at the capped index) is
        # still the target's true next token at the cap.
        ell_r = (layout.verify_lens.to(device=correct_len.device) - 1).to(
            correct_len.dtype
        )
        return torch.minimum(correct_len, ell_r)

    def _accept_greedy(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        cutoff_layout: Optional[RaggedVerifyLayout] = None,
    ):
        bs = candidates.shape[0]
        target_predict = torch.argmax(target_logits, dim=-1).view(
            bs, self.verify_num_draft_tokens
        )
        correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        if cutoff_layout is not None:
            correct_len = self._cap_correct_len(
                correct_len=correct_len, layout=cutoff_layout
            )
            row_ids = torch.arange(bs, device=target_predict.device)
            bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
        return correct_len, bonus

    def _accept_sampling(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        cutoff_layout: Optional[RaggedVerifyLayout] = None,
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
        if cutoff_layout is not None:
            correct_len = self._cap_correct_len(
                correct_len=correct_len, layout=cutoff_layout
            )
        row_ids = torch.arange(bs, dtype=torch.long, device=device)
        accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
        bonus = predicts[accept_pos].to(torch.int64)
        return correct_len, bonus

    def _accept_draft_tokens(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_block: _DraftBlockResult,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        cutoff_layout: Optional[RaggedVerifyLayout] = None,
    ):
        # Per-request accept (greedy argmax-match vs rejection sampling), dispatched
        # by batch composition. Both rules are lossless.
        greedy_mask = draft_block.greedy_mask
        # All-greedy fast path. is_all_greedy is host-side, so the branch is sync-free.
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        if all_greedy:
            return self._accept_greedy(
                candidates=candidates,
                target_logits=target_logits,
                cutoff_layout=cutoff_layout,
            )
        draft_probs = torch.softmax(
            draft_block.corrected_logits.float()
            / draft_block.temperatures[:, None, None],
            dim=-1,
        )
        # All-sampling fast path: no greedy rows -> only the chain kernel (host-side, sync-free).
        if not sampling_info.is_any_greedy:
            return self._accept_sampling(
                candidates=candidates,
                target_logits=target_logits,
                draft_probs=draft_probs,
                sampling_info=sampling_info,
                draft_input=draft_input,
                cutoff_layout=cutoff_layout,
            )
        # Mixed: run both rules and select per row by greedy_mask.
        greedy_len, greedy_bonus = self._accept_greedy(
            candidates=candidates,
            target_logits=target_logits,
            cutoff_layout=cutoff_layout,
        )
        sampling_len, sampling_bonus = self._accept_sampling(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            cutoff_layout=cutoff_layout,
        )
        correct_len = torch.where(
            greedy_mask, greedy_len.to(sampling_len.dtype), sampling_len
        )
        bonus = torch.where(greedy_mask, greedy_bonus, sampling_bonus)
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
            start_pos=0,
            is_prefill=True,
        )
        next_main_hidden = None
        if self._draft_owns_block_forward:
            next_main_hidden = self._select_prefill_last_hidden(
                hidden=logits_output.hidden_states, extend_lens=batch.extend_lens
            )
        logits_output.hidden_states = None

        batch_output.next_draft_input = self._make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
            hidden_states=next_main_hidden,
        )
        return batch_output

    def _select_prefill_last_hidden(
        self,
        *,
        hidden: torch.Tensor,
        extend_lens: list,
    ) -> torch.Tensor:
        # The first V4 decode anchors on each request's last prefill token, so its
        # target hidden is gathered at the per-request extend boundary from the flat
        # extend-token aux hidden.
        device = hidden.device
        ends = torch.tensor(extend_lens, dtype=torch.int64, device=device).cumsum(0) - 1
        return hidden[ends]

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

    def _propose_draft_block(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: _VerifyWindow,
        bs: int,
        device: str,
        target_model,
        lm_head,
        sampling_info,
    ) -> _DraftProposal:
        # Capability dispatch (B3). A draft model that owns its block forward runs
        # the whole draft block (embed -> stages -> Markov head finish, plus its own
        # confidence tap) inside ``forward_spec`` and reads from its own KV ring, so
        # the worker delegates the proposal to it. Dense backbones run the existing
        # paged-pool draft forward -> base-logits -> serial-Markov-sample sequence,
        # byte-identical to before.
        if self._draft_owns_block_forward:
            return self._propose_draft_block_via_model(
                batch=batch,
                draft_input=draft_input,
                bs=bs,
                device=device,
                sampling_info=sampling_info,
            )

        embed_module = target_model.get_input_embeddings()
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
        draft_block = self._sample_draft_block(
            base_logits=base_logits,
            anchor_tokens=draft_block_ids[:, 0],
            draft_hidden=draft_hidden,
            sampling_info=sampling_info,
        )
        return _DraftProposal(
            draft_block_ids=draft_block_ids,
            draft_block=draft_block,
            draft_hidden=draft_hidden,
        )

    def _propose_draft_block_via_model(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        bs: int,
        device: str,
        sampling_info,
    ) -> _DraftProposal:
        # V4-style self-contained draft: ``forward_spec`` consumes the anchor token
        # plus the captured target hidden, runs the draft stages off its own KV ring,
        # and finishes with the serial Markov head. The worker still owns the per-row
        # mixed sampler (greedy rows argmax, sampling rows temperature-softmax) so the
        # draft distribution and the RNG draw count match the dense path; it passes
        # that sampler in and reconstructs the accept-path ``_DraftBlockResult`` from
        # ``forward_spec``'s outputs.
        gamma = self.gamma
        anchor_tokens = draft_input.bonus_tokens.view(-1).to(
            device=device, dtype=torch.long
        )
        main_hidden = draft_input.hidden_states.to(device=device, non_blocking=True)
        start_pos = self._draft_start_pos(batch=batch)

        greedy_mask = self._resolve_greedy_mask(bs=bs, sampling_info=sampling_info)
        if sampling_info is None:
            temperatures = torch.ones(bs, dtype=torch.float32, device=device)
        else:
            temperatures = (
                sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
            )
        sampler = self._build_block_sampler(
            greedy_mask=greedy_mask, temperatures=temperatures
        )

        with torch.inference_mode():
            spec_out = self.draft_model.forward_spec(
                anchor_tokens,
                main_hidden,
                start_pos=start_pos,
                sampler=sampler,
            )
        if spec_out is None:
            raise RuntimeError(
                "DSpark V4 forward_spec returned None during decode; the draft "
                "block must run when start_pos > 0."
            )
        output_ids, corrected_logits = spec_out
        draft_block_ids = output_ids[:, :gamma].contiguous()
        draft_tokens = output_ids[:, 1:].contiguous()
        draft_block = _DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=corrected_logits,
            greedy_mask=greedy_mask,
            temperatures=temperatures,
        )
        return _DraftProposal(
            draft_block_ids=draft_block_ids,
            draft_block=draft_block,
            draft_hidden=None,
        )

    def _draft_start_pos(self, *, batch: ScheduleBatch) -> int:
        # The V4 draft KV ring is indexed by a scalar window position. The worker's
        # batched ragged seq_lens collapse to a single anchor position only when the
        # batch is position-uniform; the GPU integration that lifts this to per-row
        # ring offsets is owned by the dsv4 backend chapter (GPU-unvalidated here).
        return int(batch.seq_lens.max().item())

    def _build_block_sampler(
        self,
        *,
        greedy_mask: torch.Tensor,
        temperatures: torch.Tensor,
    ):
        any_sampling = bool((~greedy_mask).any())
        if not any_sampling:

            def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                return torch.argmax(step_logits, dim=-1)

        else:

            def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

        return sampler

    def _relay_confidence(
        self,
        *,
        req_pool_indices: torch.Tensor,
        draft_hidden: Optional[torch.Tensor],
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        # Confidence dispatch. A draft model that computes its own confidence at the
        # correct tap (V4: post-hc_head PRE-norm) exposes it via ``last_confidence``;
        # the worker relays that value. Dense backbones expose no such tap, so the
        # worker computes confidence from the post-norm ``draft_hidden`` exactly as
        # before (byte-identical).
        model_confidence = (
            self.draft_model.last_confidence() if self._draft_owns_confidence else None
        )
        if model_confidence is not None:
            confidence = model_confidence
        else:
            assert draft_hidden is not None
            confidence = self._compute_confidence(
                draft_hidden=draft_hidden,
                anchor_tokens=anchor_tokens,
                draft_tokens=draft_tokens,
            )
        self._stash_confidence(
            req_pool_indices=req_pool_indices,
            confidence=confidence,
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

        return _TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _commit_verify_hidden(
        self,
        *,
        batch: ScheduleBatch,
        layout: Optional[RaggedVerifyLayout],
        hidden_strided: Optional[torch.Tensor],
        verify_window: _VerifyWindow,
        logits_output,
        correct_len: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
        run_compact: bool,
        new_seq_lens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Commit the accepted tokens' target hidden into the draft KV. A draft model
        # that owns its KV ring (V4) takes the single committed slot per request
        # (the accepted bonus position's target hidden) into its ring and the worker
        # threads that hidden forward for the next step's ``forward_spec`` anchor.
        # Dense backbones write the whole committed prefix back into the paged pool
        # exactly as before (byte-identical) and carry no hidden across iterations.
        if self._draft_owns_kv_injection:
            hidden = logits_output.hidden_states
            if hidden is None:
                raise RuntimeError(
                    "DSpark verify requires target hidden states, got None."
                )
            committed_hidden = self._select_committed_hidden(
                hidden=hidden, correct_len=correct_len, bs=bs
            )
            self._inject_target_hidden_to_draft_kv(
                target_hidden=committed_hidden,
                cache_loc=verify_window.verify_cache_loc,
                positions=verify_window.positions_2d.reshape(-1),
                start_pos=int(new_seq_lens.max().item()) - 1,
                is_prefill=False,
            )
            return committed_hidden

        if run_compact:
            self._inject_ragged_hidden_to_draft_kv(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                commit_lens=commit_lens,
                bs=bs,
            )
        else:
            hidden = logits_output.hidden_states
            if hidden is None:
                raise RuntimeError(
                    "DSpark verify requires target hidden states, got None."
                )
            hidden = hidden.view(bs, self.verify_num_draft_tokens, -1)
            self._inject_target_hidden_to_draft_kv(
                target_hidden=hidden.reshape(-1, hidden.shape[-1]),
                cache_loc=verify_window.verify_cache_loc,
                cache_loc_2d=verify_window.verify_cache_loc_2d,
                positions=verify_window.positions_2d.reshape(-1),
                commit_lens=commit_lens,
            )
        return None

    def _select_committed_hidden(
        self,
        *,
        hidden: torch.Tensor,
        correct_len: torch.Tensor,
        bs: int,
    ) -> torch.Tensor:
        # The accepted bonus position per request is at index ``correct_len`` in the
        # gamma+1 verify window; its target hidden is the next step's anchor context
        # and the single slot written into the V4 draft ring.
        hidden = hidden.view(bs, self.verify_num_draft_tokens, -1)
        row_ids = torch.arange(bs, device=hidden.device)
        return hidden[row_ids, correct_len.to(torch.long)]

    def _build_ragged_verify_window(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
    ) -> _RaggedVerifyWindow:
        # Compact (real-N) verify window. Request r contributes only its scheduled
        # prefix [anchor, s_0..s_{ell_r-1}] = verify_len_r tokens, packed back to
        # back into a `total`-token compact layout keyed by layout.extend_start_loc.
        # Cache slots are over-allocated to the full gamma+1 block (plan section 6);
        # the compact window writes only the first verify_len_r reserved slots.
        prefix_lens = batch.seq_lens
        verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
        verify_lens_cpu = layout.verify_lens_cpu

        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            prefix_lens.to(torch.int32),
            verify_lens,
            layout.total_verify_tokens,
        )
        verify_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
            batch_size=bs,
            draft_token_num=self.verify_num_draft_tokens,
            device=device,
        )[: layout.total_verify_tokens]

        verify_ids = self._compact_verify_ids(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            verify_lens_cpu=verify_lens_cpu,
            total=layout.total_verify_tokens,
            device=device,
        )

        if batch.seq_lens_cpu is not None:
            seq_lens_cpu = batch.seq_lens_cpu + torch.tensor(
                verify_lens_cpu, dtype=batch.seq_lens_cpu.dtype
            )
        else:
            seq_lens_cpu = (prefix_lens.to("cpu", dtype=torch.int32)) + torch.tensor(
                verify_lens_cpu, dtype=torch.int32
            )

        return _RaggedVerifyWindow(
            positions=positions,
            verify_cache_loc=verify_cache_loc,
            verify_ids=verify_ids,
            seq_lens_cpu=seq_lens_cpu,
        )

    def _compact_verify_ids(
        self,
        *,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        verify_lens_cpu: list[int],
        total: int,
        device: str,
    ) -> torch.Tensor:
        # Pack [anchor, s_0..s_{ell_r-1}] per request into a compact total-token
        # 1d tensor. anchor = draft_block_ids[:, 0]; s_k = draft_tokens[:, k].
        verify_ids = torch.empty((total,), dtype=torch.int64, device=device)
        offset = 0
        anchors = draft_block_ids[:, 0]
        for r, verify_len in enumerate(verify_lens_cpu):
            verify_ids[offset] = anchors[r]
            ell_r = verify_len - 1
            if ell_r > 0:
                verify_ids[offset + 1 : offset + verify_len] = draft_tokens[r, :ell_r]
            offset += verify_len
        return verify_ids

    def _run_ragged_target_verify(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        ragged_window: _RaggedVerifyWindow,
        sampling_info,
    ) -> _TargetVerifyResult:
        # Compact verify forward. The merged decode runner reads
        # spec_info.ragged_verify_layout to select the token-keyed graph and the
        # merged attention backend builds ragged metadata from it
        # (generate_attn_arg_prefill's ragged branch). seq_lens stays at the
        # prefix; the backend adds layout.verify_lens itself.
        #
        # WAR/RAW (invariant 3): the layout's device buffers are built on the
        # forward stream before this prepare_for_verify, so the runner snapshots
        # them inside load_batch ahead of read_done.record(); the generic overlap
        # barrier (supports_overalloc_war_verify(), already DSPARK-gated) then
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
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        batch.seq_lens_cpu = ragged_window.seq_lens_cpu
        batch.seq_lens_sum = int(ragged_window.seq_lens_cpu.sum())

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

        return _TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _scatter_compact_to_strided(
        self,
        *,
        compact: torch.Tensor,
        layout: RaggedVerifyLayout,
        bs: int,
        fill_value: float,
    ) -> torch.Tensor:
        # compact->strided scatter (infra section 12.E). compact is [total, dim]
        # (one row per verified token, packed by layout.extend_start_loc). The
        # accept path + result processor expect a [bs*(gamma+1), dim] strided
        # tensor where request i owns rows [i*(gamma+1), i*(gamma+1)+verify_len_i).
        # Padded strided rows are filled with `fill_value`; the accept cap to ell_r
        # keeps them out of the commit decision (lossless).
        stride = self.verify_num_draft_tokens
        dim = compact.shape[1]
        strided = torch.full(
            (bs * stride, dim),
            fill_value,
            dtype=compact.dtype,
            device=compact.device,
        )
        offset = 0
        for r, verify_len in enumerate(layout.verify_lens_cpu):
            dst = r * stride
            strided[dst : dst + verify_len] = compact[offset : offset + verify_len]
            offset += verify_len
        return strided

    def _apply_logits_adjustments_strided(
        self,
        *,
        next_token_logits: torch.Tensor,
        sampling_info,
    ) -> None:
        if sampling_info is None:
            return
        apply_dflash_verify_logits_adjustments(
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            draft_token_num=self.verify_num_draft_tokens,
        )

    def _verify_full(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        sampling_info,
    ) -> tuple[_TargetVerifyResult, torch.Tensor]:
        # Real-N (full) verify: run the compact total-token forward, then scatter
        # the compact logits and hidden_states back to the bs*(gamma+1) strided
        # layout the accept path + result processor expect. Padded strided rows
        # are filled with 0 and never enter the commit decision (accept is capped
        # to ell_r). Returns the result with strided logits/hidden in place, plus
        # the strided hidden for the ragged KV injection.
        ragged_window = self._build_ragged_verify_window(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
        )
        target_verify = self._run_ragged_target_verify(
            batch=batch,
            layout=layout,
            ragged_window=ragged_window,
            sampling_info=sampling_info,
        )
        logits_output = target_verify.logits_output

        compact_logits = logits_output.next_token_logits
        strided_logits = self._scatter_compact_to_strided(
            compact=compact_logits, layout=layout, bs=bs, fill_value=0.0
        )
        self._apply_logits_adjustments_strided(
            next_token_logits=strided_logits, sampling_info=sampling_info
        )
        logits_output.next_token_logits = strided_logits

        compact_hidden = logits_output.hidden_states
        if compact_hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        hidden_strided = self._scatter_compact_to_strided(
            compact=compact_hidden, layout=layout, bs=bs, fill_value=0.0
        )
        logits_output.hidden_states = hidden_strided
        return target_verify, hidden_strided

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
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DSpark requires the target model to expose `lm_head` with `weight`."
            )

        verify_window = self._alloc_verify_window(batch=batch, bs=bs, device=device)

        sampling_info = batch.sampling_info
        proposal = self._propose_draft_block(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            target_model=target_model,
            lm_head=lm_head,
            sampling_info=sampling_info,
        )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens

        if self._confidence_head is not None:
            self._relay_confidence(
                req_pool_indices=batch.req_pool_indices,
                draft_hidden=proposal.draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
            )

        layout = self._maybe_schedule_ragged_layout(
            req_pool_indices=batch.req_pool_indices, device=device
        )
        run_compact = (
            self._ragged_verify_mode is RaggedVerifyMode.COMPACT and layout is not None
        )

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        if run_compact:
            target_verify, hidden_strided = self._verify_full(
                batch=batch,
                layout=layout,
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=bs,
                device=device,
                sampling_info=sampling_info,
            )
        else:
            target_verify = self._run_target_verify(
                batch=batch,
                draft_input=draft_input,
                verify_ids_2d=verify_ids_2d,
                verify_window=verify_window,
                sampling_info=sampling_info,
            )
            hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        correct_len, bonus = self._accept_draft_tokens(
            candidates=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            cutoff_layout=layout,
        )

        commit_lens = correct_len.to(torch.int32) + 1
        out_tokens = self._build_out_tokens(
            draft_tokens=draft_tokens, correct_len=correct_len, bonus=bonus
        )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        next_main_hidden = self._commit_verify_hidden(
            batch=batch,
            layout=layout,
            hidden_strided=hidden_strided,
            verify_window=verify_window,
            logits_output=logits_output,
            correct_len=correct_len,
            commit_lens=commit_lens,
            bs=bs,
            run_compact=run_compact,
            new_seq_lens=new_seq_lens,
        )
        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
            hidden_states=next_main_hidden,
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
