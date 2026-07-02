import logging
from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    draft_is_deepseek_v4,
    make_draft_block_spec_info,
)
from sglang.srt.speculative.dspark_components.dspark_accept import (
    accept_draft_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_confidence_metrics import (
    ConfidenceMetricsProbe,
)
from sglang.srt.speculative.dspark_components.dspark_decision_dump import (
    DsparkDecisionDumper,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DsparkDraftSampler,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_draft_proposer import (
    DraftBlockProposer,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_sts_recorder import (
    StsDataRecorder,
)
from sglang.srt.speculative.dspark_components.dspark_target_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_utils import (
    dspark_gamma_from_num_draft_tokens,
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    alloc_verify_window,
)
from sglang.srt.speculative.dspark_components.dspark_verify_planner import (
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.kernels.build_out_tokens import (
    BuildOutTokens,
)
from sglang.srt.speculative.dspark_components.kernels.finalize_accept_lens import (
    FinalizeAcceptLens,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)

_STS_COLLECT_FLUSH_EVERY: int = 256


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

        # DP-attention path selection. A dense (qwen/gemma) draft runs replicated
        # inside the attention-TP context (EAGLE3-style): the patched _TP makes its
        # RowParallelLinear reduce and base-logits all-gather size-1 no-ops, so it is
        # per-DP-rank tp=attn_tp and never joins a cross-DP collective. A DeepSeek-V4
        # (MoE) draft cannot use this (draft_tp_context does not patch attn_dp_size, so
        # the MoE gather stays on) and instead runs full-DP: it keeps the full DP
        # context and its _run_ffn goes through the shared parent MoE-DP gather, fed by
        # the draft-side DP metadata (dp_moe_sync below). Target forwards always stay
        # OUTSIDE the dense attention-TP context.
        self._draft_is_moe = draft_is_deepseek_v4(server_args=server_args)
        self._draft_dp_context_enabled = (
            server_args.enable_dp_attention and not self._draft_is_moe
        )
        attn_tp_size = server_args.tp_size // max(server_args.dp_size, 1)
        if server_args.enable_dp_attention and self._draft_is_moe and attn_tp_size > 1:
            # MoE draft under DP runs full-DP pure-TP-MoE (a2a="none"): its _run_ffn
            # goes through the shared parent MoE-DP gather (dp_moe_sync). That path is
            # only correct when attn_tp == 1 (dp_size == tp_size); with attn_tp > 1 the
            # unequal per-rank verify batch corrupts the attn-TP all-reduce (design doc
            # 9.1). MXFP4 experts have no DeepEP runner, so DeepEP-EP is not an option.
            # Fail fast on the known-broken attn_tp > 1 case instead of silent garbage.
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        # Draft runner (separate KV cache + attention backend), shared with DFlash.
        with self._draft_context():
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
        # Built in init_cuda_graphs when the greedy proposal folds into the graph.
        self._draft_sampler = None

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

        # Capability-driven draft polymorphism (no model-identity branches). Both
        # dense (qwen3 / gemma4) and V4 drafts run the single orchestration: the worker
        # drives ``draft_model_runner.forward`` on a real paged / SWA-latent pool, calls
        # the MODEL's ``compute_base_logits`` (the model owns the matmul / hc-collapse),
        # then the shared serial Markov head. The remaining capability seam is the
        # confidence tap (the model's ``compute_confidence`` hook), resolved at use site.
        # Every DSpark draft owns ``compute_base_logits`` over the target's shared
        # lm_head, attached here via ``attach_shared_modules`` (the same live target
        # object per call so the TP local-vocab shard + org_vocab_size stay consistent).
        # Dense ignores embed_tokens (it carries its own); V4 also hc-expands its own
        # embedding from it. Validate the head once HERE so a missing / weightless
        # lm_head fails at worker init, not deep inside the model's compute_base_logits.
        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DSpark requires the target model to expose `lm_head` with `weight`."
            )
        self.draft_model.attach_shared_modules(
            embed_tokens=self._resolve_target_embed_tokens(target_model),
            lm_head=lm_head,
        )

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=self.tp_rank,
            server_args=self.server_args,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        if (
            server_args.enable_dp_attention
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and not server_args.disable_cuda_graph
        ):
            # Dense-draft (qwen/gemma) compact verify under DP + cuda graph is not yet
            # supported: it deadlocks, and every fix attempted so far hits a separate
            # backend-layer gap. Fail fast at startup with actionable guidance instead of
            # a 300s watchdog hang. Concretely, the troubles hit (see the dp-attn journal
            # 2026-07-02 for full py-spy traces):
            #  1. On a step with both busy and idle DP groups, the busy target verify
            #     replays the token-keyed COMPACT graph while the idle group's dummy
            #     verify (a layout-less DFlashVerifyInput) replays the bs-keyed graph.
            #     These are two distinct captured CUDA graphs whose baked dp_gather
            #     collectives can never rendezvous -> the target-verify collective
            #     deadlocks (2 ranks wedged in a matmul, 2 parked at recv_requests).
            #  2. Making the idle group also take the token-keyed graph fails: the
            #     token-keyed graph's dp_gather requires every rank to contribute the
            #     padded tier, but an idle group has 0 real tokens and cannot be shaped
            #     into that compact-packed tier geometry (a 0-request RaggedVerifyLayout
            #     is rejected; a global-bs padded layout then mismatches at buffer fill,
            #     `_foreach_copy size 8 vs 0`).
            #  3. Forcing the whole step eager on a mixed busy/idle step instead hits the
            #     trtllm_mha backend's incomplete eager compact-verify path
            #     (`assert max_q_len is not None`) -- compact was designed to build its
            #     verify metadata inside the graph capture, so its eager path is a stub
            #     when a graph exists.
            # A real fix needs backend work (let an idle group join the tier-padded
            # token-keyed graph, or complete trtllm_mha's eager compact metadata as the
            # mixed-step fallback). Until then: eager (--disable-cuda-graph) is verified
            # lossless (gsm8k 0.94, acc_len ~5.4). The dsv4 (MoE) draft is unaffected --
            # its draft AND verify are full-DP and their idle participation already
            # matches the busy geometry, so it keeps cuda graph under DP.
            raise ValueError(
                "DSpark dense-draft compact verify under --enable-dp-attention does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        self._kv_injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            model_runner=self.model_runner,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
        )
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            dp_moe_sync=self._draft_is_moe and server_args.enable_dp_attention,
        )
        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
        )

        # Offline STS data-collection tap (read-only, off unless
        # SGLANG_DSPARK_STS_COLLECT_PATH is set). Built lazily on first record.
        self._sts_recorder: Optional[StsDataRecorder] = None

        self._confidence_probe = ConfidenceMetricsProbe(
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.tp_rank,
        )

        self._decision_dumper = DsparkDecisionDumper(
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.tp_rank,
        )

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
        return self._verify_planner.carries_confidence

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

    def _draft_context(self):
        """Patch the TP group to the attention-TP group while running the dense draft
        under DP attention (construction / backend + graph init / draft forward), so
        the draft is per-DP-rank tp=attn_tp. No-op otherwise."""
        if self._draft_dp_context_enabled:
            return draft_tp_context(get_attention_tp_group())
        return nullcontext()

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
        with self._draft_context():
            self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        # Greedy proposal folds into the draft cuda graph via the draft_sampler hook;
        # sampling batches fall back to eager. Under DP attention the draft graph is
        # captured inside the attention-TP context (per-DP-rank tp=attn_tp).
        capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with self._draft_context():
            if capture_decode_cuda_graph:
                # Must run before capture so the draft graph folds the sampler in.
                self._draft_sampler = self._maybe_build_draft_sampler()
                self.draft_model_runner.draft_sampler = self._draft_sampler
                self._proposer._draft_sampler = self._draft_sampler
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )

    def _maybe_build_draft_sampler(self):
        def _eager(reason):
            if self.tp_rank == 0:
                logger.info(
                    "DSpark draft greedy proposal kept eager (reason=%s).", reason
                )
            return None

        if self.gamma <= 0:
            return _eager("gamma<=0")
        if not hasattr(self.draft_model, "compute_base_logits"):
            return _eager("no compute_base_logits")
        if getattr(self.draft_model, "markov_head", None) is None:
            return _eager("no markov head")
        if self.tp_rank == 0:
            logger.info(
                "DSpark draft greedy proposal folded into the draft cuda graph."
            )
        return DsparkDraftSampler(
            model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(self.server_args.cuda_graph_config.decode.bs),
            device=self.device,
            # Fold confidence into the same graph when the planner carries a head:
            # the dsv4 hook's _x_post_hc tap is only same-graph fresh (see sampler doc).
            confidence_fn=(
                self._verify_planner.compute_confidence_tensor
                if self._verify_planner.carries_confidence
                else None
            ),
        )

    def clear_cache_pool(self):
        pass

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
            # Break the online SPS profiler's consecutive-decode timing pair
            # (under non-overlap no scheduler prepare hook sees this prefill).
            self._verify_planner.note_non_decode_step()
            return self._forward_prefill(batch, on_publish)

        return self._forward_decode(batch, on_publish)

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            # Global-extend step with a locally idle attention-DP group (routed here
            # because is_extend_in_batch is the global max). Run a target idle forward
            # (coefficient 1, matching the busy extend, spec_info left unscaled) to join
            # the target dp_gather, then return an empty result.
            if self.server_args.enable_dp_attention:
                batch.capture_hidden_mode = CaptureHiddenMode.FULL
                self.target_worker.forward_batch_generation(batch)
            return self._decode_idle_result(on_publish=on_publish)

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
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
        )
        logits_output.hidden_states = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

    def _run_idle_verify_participation(self, batch: ScheduleBatch) -> None:
        """Under DP attention an idle attention-DP group must still run the target
        verify forward so it joins the target's dp_gather collective; otherwise busy
        groups hang. The dummy DFlashVerifyInput carries draft_token_num ==
        verify_num_draft_tokens so get_spec_adjusted_global_num_tokens scales
        global_num_tokens by the same factor as a busy verify -> matching dp buffer
        length across ranks. The forward output is discarded (no real requests)."""
        verify_input = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.int64, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch.out_cache_loc = torch.empty((0,), dtype=torch.int64, device=self.device)
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
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
            if self.server_args.enable_dp_attention:
                # dsv4 (MoE) draft gathers across DP, so the idle group must join the
                # draft dp_gather too (propose then verify, matching the busy order).
                if self._draft_is_moe:
                    self._proposer.run_idle_participation(batch)
                self._run_idle_verify_participation(batch)
            return self._decode_idle_result(on_publish=on_publish)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        target_model = self.target_worker.model_runner.model

        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )

        sampling_info = batch.sampling_info
        with self._draft_context():
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                target_model=target_model,
                sampling_info=sampling_info,
            )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens

        # Current-step confidence (lag 0, device): the sort source for verify_lens and
        # the value published into the relay for a future step's budget. No device ring
        # is stashed anymore -- the two-steps-prior budget K flows through the host
        # relay + carry (prepare hook in overlap, compute_budget_sync otherwise).
        # On the captured greedy path the proposal already carries the in-graph
        # confidence (same-graph _x_post_hc freshness); eager batches compute it here.
        confidence = proposal.confidence
        if confidence is None:
            confidence = self._verify_planner.compute_confidence_tensor(
                draft_hidden=proposal.draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
            )

        verify_token_budget = self._resolve_verify_token_budget(
            batch=batch,
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
        )

        # dsv4 (MoE) draft under DP: the compact verify's token-keyed cuda graph holds a
        # cross-DP MoE gather, so every rank must select the same graph tier. Feed the
        # planner the DP-global max bs (already all-gathered on batch.global_num_tokens,
        # per-rank decode bs) so the tier floors to it uniformly. None off this path.
        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            else None
        )
        layout = self._verify_planner.schedule_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=verify_token_budget,
            global_num_reqs=global_num_reqs,
        )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        if run_compact:
            target_verify, hidden_strided = self._verify_executor.run_compact(
                batch=batch,
                layout=layout,
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=bs,
                device=device,
                sampling_info=sampling_info,
            )
        else:
            target_verify = self._verify_executor.run_non_compact(
                batch=batch,
                draft_input=draft_input,
                verify_ids_2d=verify_ids_2d,
                verify_window=verify_window,
                sampling_info=sampling_info,
            )
            hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        correct_len, bonus, cap_trim_lens = accept_draft_tokens(
            candidates=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            cutoff_layout=layout,
        )

        finalized = FinalizeAcceptLens.execute(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )
        commit_lens = finalized.commit_lens
        new_seq_lens = finalized.new_seq_lens
        out_tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        if on_publish is not None:
            if confidence is not None:
                # Publish this step's confidence + its prefix_len stamp into the relay
                # alongside new_seq_lens (B2: the stamp travels with the confidence so a
                # future step's freshness guard compares against the right seq_len). The
                # relay no-ops the confidence kwargs unless needs_confidence_relay.
                on_publish(
                    new_seq_lens,
                    confidence=confidence,
                    confidence_seq_lens=prefix_lens,
                )
            else:
                on_publish(new_seq_lens)

        self._verify_executor.commit_hidden(
            batch=batch,
            layout=layout,
            hidden_strided=hidden_strided,
            verify_window=verify_window,
            logits_output=logits_output,
            commit_lens=commit_lens,
            bs=bs,
            run_compact=run_compact,
        )
        logits_output.hidden_states = None

        self._maybe_record_sts_collect(
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            bs=bs,
        )
        self._confidence_probe.maybe_observe(
            carries_confidence=self._verify_planner.carries_confidence,
            is_compact_mode=self._verify_planner.is_compact_mode,
            confidence_raw=self._verify_planner.last_confidence_raw,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            bs=bs,
        )
        self._decision_dumper.maybe_dump(
            forward_ct=batch.forward_iter,
            bs=bs,
            mode=self._verify_planner.mode_value,
            budget=verify_token_budget,
            lag_steps=self._verify_planner.lag_steps,
            verify_lens=layout.verify_lens if layout is not None else None,
            confidence=confidence,
            req_pool_indices=batch.req_pool_indices,
            rids=[req.rid for req in batch.reqs],
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            bonus_tokens=bonus,
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            commit_lens=commit_lens,
        )

        next_draft_input = make_next_draft_input(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            # Uncapped full-block accept incl bonus; cap_trim_lens is 0 outside
            # CAP_ACCEPT so this equals commit_lens there. finalized carries the
            # int32 cap-trim mirror from FinalizeAcceptLens.
            block_accept_lens=commit_lens + finalized.cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=new_seq_lens,
        )

    def _maybe_record_sts_collect(
        self,
        *,
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor,
        bs: int,
    ) -> None:
        collect_path = envs.SGLANG_DSPARK_STS_COLLECT_PATH.get()
        if not collect_path:
            return
        if not self._verify_planner.carries_confidence:
            return
        confidence_raw = self._verify_planner.last_confidence_raw
        if confidence_raw is None:
            return
        if self._sts_recorder is None:
            self._sts_recorder = StsDataRecorder(
                path_stem=collect_path,
                gamma=self.gamma,
                flush_every=_STS_COLLECT_FLUSH_EVERY,
            )
        target_predict = torch.argmax(target_logits, dim=-1).view(
            bs, self.verify_num_draft_tokens
        )
        num_correct_drafts, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=verify_ids_2d,
            target_predict=target_predict,
        )
        self._sts_recorder.record(
            confidence_raw=confidence_raw,
            num_correct_drafts=num_correct_drafts,
        )

    def _resolve_verify_token_budget(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        confidence: Optional[torch.Tensor],
        prefix_lens: torch.Tensor,
    ) -> Optional[int]:
        # The verify budget K source. Overlap: already computed off-critical-path by
        # the scheduler prepare hook (get_confidence_budget_prepare) from the
        # two-steps-prior relayed confidence and attached to the draft input. Non-
        # overlap: no relay, so compute it synchronously from this step's confidence
        # (the host carry was sized with relay_lag_steps=0 to supply the full lag).
        if not self._verify_planner.schedules_verify_budget or confidence is None:
            return None
        if not self.server_args.disable_overlap_schedule:
            return draft_input.verify_token_budget
        return self._verify_planner.compute_budget_sync(
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

    def get_confidence_budget_prepare(self):
        # Injected into the scheduler's overlap prepare window when this worker
        # schedules a ragged verify budget; None disables the hook (static mode / no
        # head). Bound to the planner so all DSpark budget logic stays out of the
        # shared scheduler code.
        if not self._verify_planner.schedules_verify_budget:
            return None
        return self._verify_planner.prepare_verify_budget
