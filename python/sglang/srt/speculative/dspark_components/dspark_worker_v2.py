import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
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
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
)
from sglang.srt.speculative.dspark_components.dspark_accept import (
    accept_draft_tokens,
    build_out_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_confidence import (
    _CONFIDENCE_RELAY_LAG_STEPS,
    _CONFIDENCE_RELAY_RING_DEPTH,
    ConfidenceRelay,
    compute_confidence,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DsparkDraftSampler,
    make_next_draft_input,
    resolve_greedy_mask,
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_info import (
    DraftBlockResult,
    DraftForwardResult,
    DraftProposal,
    RaggedVerifyWindow,
    TargetVerifyResult,
    VerifyWindow,
)
from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
    build_sps_cost_table,
    compute_verify_token_budget,
)
from sglang.srt.speculative.dspark_components.dspark_utils import (
    dspark_gamma_from_num_draft_tokens,
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    alloc_verify_window,
    apply_logits_adjustments_strided,
    build_ragged_verify_window,
    ragged_layout_exceeds_captured_grid,
    scatter_compact_to_strided,
    uniform_ragged_layout,
    verify_layout_graph_num_tokens_floor,
    verify_layout_grid,
    verify_lens_broadcast_group,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.utils import get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)


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
        #
        # The target attn backend is constructed after the worker (the scheduler runs
        # init_attention_backends later), so the verify-prep self-add capability is
        # resolved lazily on first verify and cached.
        self._verify_backend_self_adds_seq_lens_cache: Optional[bool] = None
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

        # Optional confidence relay. The head and its relay are inert (no
        # buffers, no events, no compute) when the draft model lacks a
        # confidence head, so the a+b lossless decode path is unchanged.
        self._confidence_head = getattr(self.draft_model, "confidence_head", None)
        self._confidence_relay = ConfidenceRelay(
            device=self.device,
            gamma=self.gamma,
            model_runner=self.model_runner,
        )
        if self._confidence_head is not None and self.tp_rank == 0:
            logger.info(
                "DSpark confidence head enabled (with_markov=%s); confidence is "
                "relayed through a step-indexed ring (depth=%d, lag=%d) on the "
                "forward stream and is advisory only.",
                getattr(self._confidence_head, "with_markov", True),
                _CONFIDENCE_RELAY_RING_DEPTH,
                _CONFIDENCE_RELAY_LAG_STEPS,
            )

        # Ragged-verify mode and the confidence prefix scheduler. The scheduler is
        # inert (None) unless the mode is cap-accept/compact AND a confidence head is
        # present, so the static-mode / no-head path is byte-identical to the static
        # uniform-gamma worker. cap-accept runs the full bs*(gamma+1) window and
        # only caps accept at per-request ell_r; compact (real-N) is not wired here.
        self._ragged_verify_mode = read_ragged_verify_mode()
        self._verify_scheduler: Optional[ConfidencePrefixScheduler] = None
        if self._ragged_verify_mode is not RaggedVerifyMode.STATIC:
            if self._confidence_head is None:
                raise ValueError(
                    f"DSpark ragged-verify mode {self._ragged_verify_mode.value!r} "
                    f"schedules per-request verify lengths from the draft confidence "
                    f"head, but this DSpark draft checkpoint has no confidence head -- "
                    f"the checkpoint is wrong/incomplete (it ships no "
                    f"enable_confidence_head + trained confidence_head weights). Use a "
                    f"draft checkpoint that includes the confidence head, or run "
                    f"SGLANG_RAGGED_VERIFY_MODE=static."
                )
            self._verify_scheduler = ConfidencePrefixScheduler(
                sps_table=build_sps_cost_table(
                    server_args=self.server_args,
                    verify_num_draft_tokens=self.verify_num_draft_tokens,
                    tp_rank=self.tp_rank,
                ),
                cfg=DSparkScheduleConfig(gamma=self.gamma),
            )
            if self.tp_rank == 0:
                logger.info(
                    "DSpark ragged-verify scheduler enabled (mode=%s).",
                    self._ragged_verify_mode.value,
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
        # Greedy proposal folds into the draft cuda graph via the draft_sampler hook;
        # sampling batches fall back to eager. tp=1 only.
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
        if capture_decode_cuda_graph:
            # Must run before capture so the draft graph folds the sampler in.
            self._draft_sampler = self._maybe_build_draft_sampler()
            self.draft_model_runner.draft_sampler = self._draft_sampler
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

        if get_tp_group().world_size != 1:
            # compute_base_logits' vocab all-gather is uncapturable above tp=1.
            return _eager("tp>1")
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

    def _maybe_schedule_ragged_layout(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        device: torch.device,
    ) -> Optional[RaggedVerifyLayout]:
        # Gate: STATIC -> None (uniform path). CAP_ACCEPT/COMPACT build a ragged
        # layout from the two-steps-prior-confidence verify_lens (see
        # _schedule_verify_lens for the ring lag); COMPACT additionally token-keys
        # the graph and scatter-packs the verify window.
        if self._ragged_verify_mode is RaggedVerifyMode.STATIC:
            return None
        verify_lens = self._schedule_verify_lens(
            req_pool_indices=req_pool_indices, prefix_lens=prefix_lens, device=device
        )
        if verify_lens is None:
            # COMPACT token-keys the verify graph and captures it with a ragged
            # layout, so a layout-less compact verify (confidence not ready at
            # startup / after reset) must still carry a degenerate uniform layout
            # to hit the same token-keyed graph (C3); otherwise it would fall
            # through to a bs-keyed replay key that the token-keyed graph dict
            # never recorded. CAP_ACCEPT stays bs-keyed, so None is correct there.
            if self._ragged_verify_mode is RaggedVerifyMode.COMPACT:
                return uniform_ragged_layout(
                    bs=len(req_pool_indices),
                    device=device,
                    verify_num_draft_tokens=self.verify_num_draft_tokens,
                    ragged_verify_mode=self._ragged_verify_mode,
                    model_runner=self.model_runner,
                )
            return None
        verify_lens_cpu = verify_lens.to("cpu").tolist()
        if ragged_layout_exceeds_captured_grid(
            num_reqs=len(verify_lens_cpu),
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        ):
            return None
        grid = verify_layout_grid(
            verify_lens_cpu=verify_lens_cpu,
            ragged_verify_mode=self._ragged_verify_mode,
            model_runner=self.model_runner,
        )
        graph_num_tokens_floor = verify_layout_graph_num_tokens_floor(
            num_reqs=len(verify_lens_cpu),
            ragged_verify_mode=self._ragged_verify_mode,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )
        return RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=device,
            grid=grid,
            graph_num_tokens_floor=graph_num_tokens_floor,
        )

    def _schedule_verify_lens(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        # Shared cutoff/full schedule: derive per-request verify_lens (= 1 + ell_r)
        # from the confidence relay ring, broadcast from rank 0 across the (attention)
        # TP group for cross-rank shape consistency. Returns None (-> uniform full
        # block) whenever the ring has not been written yet.
        #
        # Two distinct survival sources (paper §5.2):
        #   - K source = two-steps-prior survival (_two_steps_prior_k_survival):
        #     fixes the verify budget K, causally independent of this step's tokens.
        #   - sort source = current live survival (_current_live_sort_survival):
        #     ranks/truncates admission by the actual up-to-date confidence.
        # budget is computed from k_survival now; verify_lens is ranked by
        # sort_survival now — no cross-step cache.
        # Losslessness does NOT depend on either source: it is guaranteed by the
        # accept-cap in _cap_correct_len (a torch.minimum that only shrinks accept),
        # after which the bonus is re-read from the target's true distribution at the
        # cap index. The split affects only scheduling quality, never correctness.
        if self._verify_scheduler is None:
            return None
        k_survival = self._confidence_relay.two_steps_prior_k_survival(
            req_pool_indices=req_pool_indices, prefix_lens=prefix_lens
        )
        sort_survival = self._confidence_relay.current_live_sort_survival(
            req_pool_indices=req_pool_indices
        )
        if k_survival is None or sort_survival is None:
            return None

        verify_lens = self._verify_scheduler.compute_verify_lens(
            k_survival=k_survival, sort_survival=sort_survival
        ).to(device=device, dtype=torch.int32)

        if envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER.get():
            self._log_verify_lens_decision(
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                k_survival=k_survival,
                sort_survival=sort_survival,
                verify_lens=verify_lens,
            )

        broadcast_group, group_size = verify_lens_broadcast_group(
            tp_size=self.server_args.tp_size
        )
        if group_size > 1:
            # GroupCoordinator.broadcast maps the local src rank to a global rank via
            # self.ranks[src], so passing src=0 broadcasts from the group-local rank 0
            # (not a global rank) even when the group's ranks[0] != 0 (review A6).
            broadcast_group.broadcast(verify_lens, src=0)

        return verify_lens

    def _log_verify_lens_decision(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        k_survival: torch.Tensor,
        sort_survival: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> None:
        cfg = self._verify_scheduler.cfg
        budget = compute_verify_token_budget(
            history_survival_probs=k_survival,
            sps_table=self._verify_scheduler.sps_table,
            cfg=cfg,
        )
        max_len = cfg.resolved_max_verify_len()
        req_ids = req_pool_indices.tolist()
        prefixes = prefix_lens.tolist()
        lens = verify_lens.tolist()
        sort_rows = sort_survival.to(torch.float32).tolist()
        logger.info(
            "[DSPARK-CPS] step=%d num_reqs=%d budget=%d gamma=%d verify_len_range=[%d,%d]",
            self._confidence_relay.step_ct,
            len(req_ids),
            budget,
            cfg.gamma,
            cfg.min_verify_len,
            max_len,
        )
        for row in range(len(req_ids)):
            survival_str = "[" + ", ".join(f"{p:.3f}" for p in sort_rows[row]) + "]"
            logger.info(
                "[DSPARK-CPS]   req=%d prefix=%d verify_len=%d sort_survival=%s",
                int(req_ids[row]),
                int(prefixes[row]),
                int(lens[row]),
                survival_str,
            )

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

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

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
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _propose_draft_block(
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
        fwd = self._run_draft_block_forward(
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

    def _relay_confidence(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        draft_hidden: Optional[torch.Tensor],
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        # Confidence dispatch (capability seam, R9). With markov + sampling now in the
        # worker, the model's own ``forward_head`` no longer runs, so a V4 draft
        # instead exposes a ``compute_confidence`` hook evaluated at its correct tap
        # (post-hc_head PRE-norm, stashed by the just-completed forward); the worker
        # calls it with the anchor + the just-sampled draft tokens. Dense backbones
        # expose no such hook, so the worker computes confidence from the post-norm
        # ``draft_hidden`` exactly as before (byte-identical). Confidence is advisory
        # only and off by default; losslessness never depends on it.
        compute_confidence_hook = getattr(self.draft_model, "compute_confidence", None)
        if compute_confidence_hook is not None:
            with torch.inference_mode():
                confidence = compute_confidence_hook(
                    anchor_tokens=anchor_tokens,
                    sampled_tokens=draft_tokens,
                )
        else:
            assert draft_hidden is not None
            confidence = compute_confidence(
                draft_hidden=draft_hidden,
                anchor_tokens=anchor_tokens,
                draft_tokens=draft_tokens,
                confidence_head=self._confidence_head,
                markov_head=self.draft_model.markov_head,
                gamma=self.gamma,
            )
        if confidence is None:
            return
        self._confidence_relay.stash(
            req_pool_indices=req_pool_indices,
            confidence=confidence,
            prefix_lens=prefix_lens,
        )

    def _run_draft_block_forward(
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
        # ``_inject_target_hidden_to_draft_kv``. ``commit_lens`` (= 1 + correct_len)
        # bounds the per-request committed window, and for the MLA single-latent pool
        # selects the accepted bonus slot per request.
        if run_compact:
            self._inject_ragged_hidden_to_draft_kv(
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
        self._inject_target_hidden_to_draft_kv(
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

        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )

        sampling_info = batch.sampling_info
        proposal = self._propose_draft_block(
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

        if self._confidence_head is not None:
            self._relay_confidence(
                req_pool_indices=batch.req_pool_indices,
                prefix_lens=prefix_lens,
                draft_hidden=proposal.draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
            )

        layout = self._maybe_schedule_ragged_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
        )
        if self._confidence_head is not None:
            # Advance the relay step counter once per decode step, after this step's
            # write (in _stash_confidence) and lagged read (in _schedule_verify_lens)
            # so both used the same step_ct; the next step then writes the next slot.
            self._confidence_relay.advance_step()
        run_compact = (
            self._ragged_verify_mode is RaggedVerifyMode.COMPACT and layout is not None
        )

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        if run_compact:
            target_verify, hidden_strided = self._run_target_verify_mode_compact(
                batch=batch,
                layout=layout,
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=bs,
                device=device,
                sampling_info=sampling_info,
            )
        else:
            target_verify = self._run_target_verify_mode_non_compact(
                batch=batch,
                draft_input=draft_input,
                verify_ids_2d=verify_ids_2d,
                verify_window=verify_window,
                sampling_info=sampling_info,
            )
            hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        correct_len, bonus = accept_draft_tokens(
            candidates=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            cutoff_layout=layout,
        )

        commit_lens = correct_len.to(torch.int32) + 1
        out_tokens = build_out_tokens(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        self._commit_verify_hidden(
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

        next_draft_input = make_next_draft_input(
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
