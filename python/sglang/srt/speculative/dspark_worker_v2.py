import logging
from typing import Optional

import msgspec
import torch

from sglang.srt.distributed import get_tp_group
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
from sglang.srt.utils.async_probe import maybe_detect_in_closed_range

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


class _DraftForwardResult(msgspec.Struct, frozen=True):
    # Output of the unified draft block forward. ``raw_hidden`` is the model's
    # un-reshaped backbone hidden (dense: 2-D ``[bs*gamma, d]``; dsv4: 3-D
    # ``[bs*gamma, hc, d]``); it is fed straight to ``compute_base_logits`` (the model
    # owns the matmul / hc-collapse). ``draft_hidden_3d`` is ``raw_hidden.view(bs,
    # gamma, -1)``, the dense markov / dense confidence input. dsv4's markov takes no
    # hidden and its confidence reads the model-stashed ``_x_post_hc``, so dsv4 ignores
    # ``draft_hidden_3d``.
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor


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
        #
        # Until a profiled (non-flat) table ships the hardware-aware scheduler is a
        # no-op: lookup() returns a constant, so the verify-token budget degenerates
        # to verify-all and every request keeps verify_len == gamma (the scheduler's
        # resolved_max_verify_len caps at gamma, so compact verifies the anchor plus
        # up to gamma-1 drafts; this is lossless -- _cap_correct_len caps accept and
        # the bonus is re-read from the target distribution). The
        # verify_lens >= 1 anchor contract (see DSparkScheduleConfig.min_verify_len
        # and schedule_verify_lens_topk's lower-bound clamp) MUST be in place before
        # any profiled table is supplied, because a non-flat table yields small K
        # and would otherwise drive verify_len to 0.
        sps_table_path = self.server_args.speculative_dspark_sps_table_path
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

    def _make_next_draft_input(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
    ) -> DFlashDraftInputV2:
        # The next step's draft state is just the bonus tokens + committed seq lens.
        # The next anchor's target hidden is read back from the draft KV pool (it was
        # written there by the commit injection), not relayed through the spec input,
        # so the legacy Eagle-shaped ``hidden_states`` slot stays the empty placeholder.
        return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)

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
        # any_sampling == not is_all_greedy, read host-side (is_all_greedy is a
        # Python bool on sampling_info) so this branch draws no GPU sync. No
        # sampling_info -> all-greedy fast path (argmax only, no RNG draw).
        any_sampling = sampling_info is not None and not sampling_info.is_all_greedy

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
        # Closed interval: sigmoid saturates to exactly 0.0/1.0 at fp32 for large
        # logits. Advisory only; async + gated, no per-step sync.
        maybe_detect_in_closed_range(confidence, 0.0, 1.0, "DSpark confidence")
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
        # layout from the lagged-confidence verify_lens (see _schedule_verify_lens
        # for the >= 1-step lag); COMPACT additionally token-keys the graph and
        # scatter-packs the verify window.
        if self._ragged_verify_mode is RaggedVerifyMode.STATIC:
            return None
        verify_lens = self._schedule_verify_lens(
            req_pool_indices=req_pool_indices, device=device
        )
        if verify_lens is None:
            # COMPACT token-keys the verify graph and captures it with a ragged
            # layout, so a layout-less compact verify (confidence not ready at
            # startup / after reset) must still carry a degenerate uniform layout
            # to hit the same token-keyed graph (C3); otherwise it would fall
            # through to a bs-keyed replay key that the token-keyed graph dict
            # never recorded. CAP_ACCEPT stays bs-keyed, so None is correct there.
            if self._ragged_verify_mode is RaggedVerifyMode.COMPACT:
                return self._uniform_ragged_layout(
                    bs=len(req_pool_indices), device=device
                )
            return None
        verify_lens_cpu = verify_lens.to("cpu").tolist()
        grid = self._verify_layout_grid(verify_lens_cpu=verify_lens_cpu)
        graph_num_tokens_floor = self._verify_layout_graph_num_tokens_floor(
            num_reqs=len(verify_lens_cpu)
        )
        return RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=device,
            grid=grid,
            graph_num_tokens_floor=graph_num_tokens_floor,
        )

    def _uniform_ragged_layout(
        self, *, bs: int, device: torch.device
    ) -> RaggedVerifyLayout:
        # The degenerate uniform layout (verify_lens = [gamma+1] * bs) that a
        # layout-less compact verify carries so it hits the same token-keyed graph
        # the real ragged batch does (C3). Geometry matches the static full block.
        verify_lens_cpu = [self.verify_num_draft_tokens] * bs
        grid = self._verify_layout_grid(verify_lens_cpu=verify_lens_cpu)
        graph_num_tokens_floor = self._verify_layout_graph_num_tokens_floor(num_reqs=bs)
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
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        # Shared cutoff/full schedule: derive per-request verify_lens (= 1 + ell_r)
        # from the confidence relay buffer, broadcast from rank 0 across the TP
        # group for cross-rank shape consistency. Returns None (-> uniform full
        # block) whenever the relay buffer has not been written yet.
        #
        # Gather source: the per-request confidence is gathered directly off the
        # device relay buffer (_confidence_buf[req_pool_indices], device->device),
        # so the critical path keeps the no-synchronize design (no req_pool_indices
        # D2H to index a pinned host buffer). This reads the LIVE device buffer
        # written on the forward stream this step rather than the >= 1-step lagged
        # pinned snapshot that pull_confidence_history exposes; the lag for THIS
        # path therefore collapses toward 0. Losslessness does NOT depend on the lag
        # size: it is guaranteed by the accept-cap in _cap_correct_len (a
        # torch.minimum that only shrinks accept), after which the bonus is re-read
        # from the target's true distribution at the cap index. The lag change
        # affects only scheduling quality (which budget K / ranking is used), never
        # correctness. pull_confidence_history's lagged-snapshot relay is retained
        # for callers that want the gated host copy.
        #
        # Sort source (deviation from paper §5.2's "sort by current confidence"):
        # the same survival snapshot feeds BOTH the budget K
        # (update_budget_from_history) AND the rank/truncate (compute_verify_lens),
        # so admission is ordered by one snapshot. This is intentional -- the
        # accept-cap makes the result lossless regardless of the sort source; the
        # deviation only affects throughput quality, never correctness.
        if self._verify_scheduler is None:
            return None
        if self._confidence_buf is None:
            return None

        confidence = self._confidence_buf[req_pool_indices]
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

    def _verify_layout_graph_num_tokens_floor(self, *, num_reqs: int) -> int:
        # The token-keyed capture grid {b * num_draft : b in capture_bs} ties each
        # token tier to one capture_bs, whose graph captured exactly b request
        # slots. A batch whose real total rounds up to a tier with fewer slots than
        # num_reqs would not fit, so floor the bucket to this batch's bs-derived
        # full block. Zero (no floor) when no token-keyed graph exists, where the
        # bucket is the real total run eager.
        if (
            self._ragged_verify_mode is not RaggedVerifyMode.COMPACT
            or self._ragged_capture_num_tokens() is None
        ):
            return 0
        return num_reqs * self.verify_num_draft_tokens

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

    def _propose_draft_block(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: _VerifyWindow,
        bs: int,
        device: str,
        target_model,
        sampling_info,
    ) -> _DraftProposal:
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
        base_logits = self.draft_model.compute_base_logits(fwd.raw_hidden).view(
            bs, self.gamma, -1
        )
        draft_block = self._sample_draft_block(
            base_logits=base_logits,
            anchor_tokens=draft_block_ids[:, 0],
            draft_hidden=fwd.draft_hidden_3d,
            sampling_info=sampling_info,
        )
        return _DraftProposal(
            draft_block_ids=draft_block_ids,
            draft_block=draft_block,
            draft_hidden=fwd.draft_hidden_3d,
        )

    def _relay_confidence(
        self,
        *,
        req_pool_indices: torch.Tensor,
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
        compute_confidence = getattr(self.draft_model, "compute_confidence", None)
        if compute_confidence is not None:
            with torch.inference_mode():
                confidence = compute_confidence(
                    anchor_tokens=anchor_tokens,
                    sampled_tokens=draft_tokens,
                )
        else:
            assert draft_hidden is not None
            confidence = self._compute_confidence(
                draft_hidden=draft_hidden,
                anchor_tokens=anchor_tokens,
                draft_tokens=draft_tokens,
            )
        if confidence is None:
            return
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
    ) -> _DraftForwardResult:
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
        return _DraftForwardResult(
            draft_block_ids=draft_block_ids,
            raw_hidden=raw_hidden,
            draft_hidden_3d=draft_hidden_3d,
        )

    def _run_target_verify_mode_non_compact(
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

        if batch.seq_lens_cpu is None:
            raise RuntimeError("DSpark decode expected batch.seq_lens_cpu, got None")
        seq_lens_cpu = batch.seq_lens_cpu + torch.tensor(
            verify_lens_cpu, dtype=batch.seq_lens_cpu.dtype
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

        verify_window = self._alloc_verify_window(batch=batch, bs=bs, device=device)

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
