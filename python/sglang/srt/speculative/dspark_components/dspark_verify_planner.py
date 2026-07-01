import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_LAG,
    FutureMap,
    ResolvedConfidence,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_confidence import (
    compute_confidence,
)
from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
    build_sps_cost_table,
)
from sglang.srt.speculative.dspark_components.dspark_sts_table import (
    load_sts_calibration_from_path,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    ragged_capture_num_tokens,
    ragged_layout_exceeds_captured_grid,
    uniform_ragged_layout,
    verify_layout_graph_num_tokens_floor,
    verify_layout_grid,
    verify_lens_broadcast_group,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
    round_up_grid,
)
from sglang.srt.utils.async_probe import maybe_assert_async

logger = logging.getLogger(__name__)


class DSparkVerifyPlanner:
    def __init__(
        self,
        *,
        draft_model,
        gamma: int,
        model_runner,
        device,
        tp_rank: int,
        server_args: ServerArgs,
        verify_num_draft_tokens: int,
    ) -> None:
        self.draft_model = draft_model
        self.gamma = gamma
        self.model_runner = model_runner
        self.device = device
        self.server_args = server_args
        self.verify_num_draft_tokens = verify_num_draft_tokens

        # Optional confidence head. The head and its host budget planner are inert
        # (no buffers, no compute) when the draft model lacks a confidence head, so
        # the a+b lossless decode path is unchanged.
        self._confidence_head = getattr(self.draft_model, "confidence_head", None)

        sts_path = server_args.speculative_dspark_confidence_sts_path
        if sts_path and self._confidence_head is not None:
            calibration = load_sts_calibration_from_path(sts_path)
            sts_temperatures = torch.tensor(
                calibration.temperatures, dtype=torch.float32, device=device
            )
            if envs.SGLANG_DSPARK_STS_COLLECT_PATH.get() and not bool(
                torch.all(sts_temperatures == 1.0)
            ):
                raise ValueError(
                    "DSpark STS data collection (SGLANG_DSPARK_STS_COLLECT_PATH) "
                    "requires identity temperatures, but a non-identity calibration "
                    f"was loaded from {sts_path}. Collect pre-calibration logits with "
                    "no table (omit --speculative-dspark-confidence-sts-path)."
                )
            if sts_temperatures.numel() != self.gamma:
                raise ValueError(
                    "DSpark STS calibration was fit for gamma="
                    f"{sts_temperatures.numel()} but the runtime gamma is "
                    f"{self.gamma}; refit the table for gamma={self.gamma} or omit "
                    "--speculative-dspark-confidence-sts-path."
                )
            self._confidence_head.sts_temperatures = sts_temperatures
            if tp_rank == 0:
                logger.info(
                    "DSpark STS calibration loaded from %s (gamma=%d); per-position "
                    "temperatures applied to confidence-head survival.",
                    sts_path,
                    self.gamma,
                )
        elif sts_path and self._confidence_head is None:
            if tp_rank == 0:
                logger.warning(
                    "DSpark STS calibration path given but no confidence head present "
                    "(static mode / head-less checkpoint); ignoring %s.",
                    sts_path,
                )

        # Ragged-verify mode and the host budget planner. The planner is inert (None)
        # unless the mode is cap-accept/compact AND a confidence head is present, so
        # the static-mode / no-head path is byte-identical to the static uniform-gamma
        # worker. cap-accept runs the full bs*(gamma+1) window and only caps accept at
        # per-request ell_r; compact (real-N) packs the verify window.
        self._ragged_verify_mode = read_ragged_verify_mode()
        self._schedule_cfg = DSparkScheduleConfig(gamma=self.gamma)
        self._budget_planner: Optional[HostConfidenceBudgetPlanner] = None
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
            self._require_prep_in_cuda_graph()
            sps_table = build_sps_cost_table(
                server_args=self.server_args,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
            )
            # The async FutureMap relay supplies CONFIDENCE_RELAY_RING_LAG steps of
            # lag under overlap: the deferred pinned ring reads an already-landed
            # lag-RING_LAG slot (sync-free), so the relay itself is that many steps
            # behind. Without overlap there is no relay, so the host carry supplies
            # the full lag. Kept equal to overlap_utils.CONFIDENCE_RELAY_RING_LAG so
            # carry_steps = total_lag - relay_lag_steps is correct.
            relay_lag_steps = (
                0
                if self.server_args.disable_overlap_schedule
                else CONFIDENCE_RELAY_RING_LAG
            )
            self._budget_planner = HostConfidenceBudgetPlanner(
                sps_table=sps_table,
                cfg=self._schedule_cfg,
                model_runner=self.model_runner,
                relay_lag_steps=relay_lag_steps,
            )
            if tp_rank == 0:
                logger.info(
                    "DSpark ragged-verify scheduler enabled (mode=%s, lag=%d, "
                    "relay_lag=%d).",
                    self._ragged_verify_mode.value,
                    self._budget_planner.lag_steps,
                    relay_lag_steps,
                )

    def _require_prep_in_cuda_graph(self) -> None:
        # B3: ragged-non-static DSpark requires the captured-graph prepare path so the
        # verify metadata is built device-side at replay. With
        # SGLANG_PREP_IN_CUDA_GRAPH=0 the replay path reads verify_lens_cpu per step on
        # the critical path, which defeats the sync-free design; fail fast (mirror
        # adaptive_unsupported_reason) rather than silently run the host-read path.
        if not envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            raise ValueError(
                f"DSpark ragged-verify mode {self._ragged_verify_mode.value!r} "
                f"requires SGLANG_PREP_IN_CUDA_GRAPH=1 (the captured-graph prepare "
                f"path). It is currently disabled, which would put per-step "
                f"verify_lens_cpu host reads on the critical path. Set "
                f"SGLANG_PREP_IN_CUDA_GRAPH=1 or run SGLANG_RAGGED_VERIFY_MODE=static."
            )

    @property
    def carries_confidence(self) -> bool:
        return self._confidence_head is not None

    @property
    def last_confidence_raw(self) -> Optional[torch.Tensor]:
        if self._confidence_head is None:
            return None
        return self._confidence_head._last_confidence_raw

    @property
    def schedules_verify_budget(self) -> bool:
        return self._budget_planner is not None

    @property
    def is_compact_mode(self) -> bool:
        return self._ragged_verify_mode is RaggedVerifyMode.COMPACT

    @property
    def mode_value(self) -> str:
        return self._ragged_verify_mode.value

    @property
    def lag_steps(self) -> Optional[int]:
        # The two-steps-prior causal lag the host budget planner applies (None when
        # no scheduler runs, i.e. static / no-head). Advisory metadata for the dump.
        if self._budget_planner is None:
            return None
        return self._budget_planner.lag_steps

    def should_run_compact(self, *, layout: Optional[RaggedVerifyLayout]) -> bool:
        return (
            self._ragged_verify_mode is RaggedVerifyMode.COMPACT and layout is not None
        )

    def compute_confidence_tensor(
        self,
        *,
        draft_hidden: Optional[torch.Tensor],
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Confidence dispatch (capability seam, R9). With markov + sampling now in the
        # worker, the model's own ``forward_head`` no longer runs, so a V4 draft
        # instead exposes a ``compute_confidence`` hook evaluated at its correct tap
        # (post-hc_head PRE-norm, stashed by the just-completed forward); the worker
        # calls it with the anchor + the just-sampled draft tokens. Dense backbones
        # expose no such hook, so the worker computes confidence from the post-norm
        # ``draft_hidden`` exactly as before (byte-identical). Confidence is advisory
        # only and off by default; losslessness never depends on it. Returns the
        # current-step confidence ([bs, gamma], device) for the worker to sort on and
        # publish into the relay -- no device ring is stashed here anymore.
        if self._confidence_head is None:
            return None
        compute_confidence_hook = getattr(self.draft_model, "compute_confidence", None)
        if compute_confidence_hook is not None:
            with torch.inference_mode():
                return compute_confidence_hook(
                    anchor_tokens=anchor_tokens,
                    sampled_tokens=draft_tokens,
                )
        assert draft_hidden is not None
        return compute_confidence(
            draft_hidden=draft_hidden,
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            confidence_head=self._confidence_head,
            markov_head=self.draft_model.markov_head,
            gamma=self.gamma,
        )

    def prepare_verify_budget(
        self, batch: ScheduleBatch, future_map: FutureMap
    ) -> None:
        # Overlap prepare hook (scheduler run_batch window, next to
        # resolve_seq_lens_cpu): pull the two-steps-prior confidence to host with no
        # fresh D2H, run the pure-CPU greedy, and attach the budget K to the draft
        # input the worker reads this step. Computed here so the host greedy overlaps
        # the previous forward; the per-request GPU sort still runs in the forward.
        draft_input = batch.spec_info
        if self._budget_planner is None or draft_input is None:
            return
        # Only decode steps feed the carry (the worker routes extend/prefill to
        # _forward_prefill, which has no verify budget); advancing the carry on a
        # prefill step would misalign the causal lag for the surrounding decodes.
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return
        resolved = future_map.resolve_confidence_cpu(batch)
        draft_input.verify_token_budget = self._budget_from_resolved(
            resolved=resolved, req_pool_indices_cpu=batch.req_pool_indices_cpu
        )

    def compute_budget_sync(
        self,
        *,
        confidence: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> Optional[int]:
        # Non-overlap fallback: no relay, so snapshot this step's confidence to host
        # synchronously (the non-overlap loop is already synchronous) and feed the same
        # carry + greedy; the carry (relay_lag_steps=0) supplies the full lag. generation
        # = each slot's current occupancy stamp for this step's confidence.
        # (prefix_lens is vestigial now the guard uses generation, not the seq_len stamp.)
        del prefix_lens
        if self._budget_planner is None:
            return None
        req_pool_indices_cpu = req_pool_indices.to("cpu").to(torch.int64)
        generation = self.model_runner.req_to_token_pool.req_generation[
            req_pool_indices_cpu
        ].clone()
        resolved = ResolvedConfidence(
            confidence=confidence.to("cpu"),
            generation=generation,
        )
        return self._budget_from_resolved(
            resolved=resolved, req_pool_indices_cpu=req_pool_indices_cpu
        )

    def _budget_from_resolved(
        self,
        *,
        resolved: Optional[ResolvedConfidence],
        req_pool_indices_cpu: torch.Tensor,
    ) -> Optional[int]:
        # None at cold start (nothing relayed yet) -> the worker falls back to the
        # uniform verify-all layout, identical to the pre-relay startup behavior.
        if resolved is None:
            return None
        # Each slot's CURRENT occupancy generation (host, no D2H) for the guard to
        # compare against the relayed confidence's stamped generation.
        current_generation = self.model_runner.req_to_token_pool.req_generation[
            req_pool_indices_cpu.to(torch.int64)
        ]
        return int(
            self._budget_planner.compute_budget(
                confidence=resolved.confidence,
                generation=resolved.generation,
                current_generation=current_generation,
                req_pool_indices_cpu=req_pool_indices_cpu,
            )
        )

    def schedule_layout(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        device: torch.device,
        confidence: Optional[torch.Tensor],
        budget: Optional[int],
    ) -> Optional[RaggedVerifyLayout]:
        # Gate: STATIC -> None (uniform path). CAP_ACCEPT/COMPACT build a ragged
        # layout from the per-request verify_lens (GPU-sorted by the current-step
        # confidence, sized by the relay-fed budget); COMPACT additionally token-keys
        # the graph and scatter-packs the verify window.
        if self._ragged_verify_mode is RaggedVerifyMode.STATIC:
            return None
        verify_lens = self._schedule_verify_lens(
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=budget,
        )
        if verify_lens is None:
            # COMPACT token-keys the verify graph and captures it with a ragged
            # layout, so a layout-less compact verify (confidence/budget not ready at
            # startup / after reset) must still carry a degenerate uniform layout to
            # hit the same token-keyed graph (C3); otherwise it would fall through to
            # a bs-keyed replay key that the token-keyed graph dict never recorded.
            # CAP_ACCEPT stays bs-keyed, so None is correct there.
            if self._ragged_verify_mode is RaggedVerifyMode.COMPACT:
                return uniform_ragged_layout(
                    bs=len(req_pool_indices),
                    device=device,
                    verify_num_draft_tokens=self.verify_num_draft_tokens,
                    ragged_verify_mode=self._ragged_verify_mode,
                    model_runner=self.model_runner,
                )
            return None
        # bs = verify_lens.shape[0] is tensor metadata (host, no D2H), so the grid gate
        # and the bs-derived tier are computed without pulling verify_lens off the
        # forward stream.
        bs = int(verify_lens.shape[0])
        if ragged_layout_exceeds_captured_grid(
            num_reqs=bs,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        ):
            return None
        graph_num_tokens_floor = verify_layout_graph_num_tokens_floor(
            num_reqs=bs,
            ragged_verify_mode=self._ragged_verify_mode,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )
        capture_num_tokens = ragged_capture_num_tokens(model_runner=self.model_runner)
        if graph_num_tokens_floor > 0 and capture_num_tokens is not None:
            # Sync-free device path (COMPACT + token-keyed graph): tier is bs-derived, so
            # verify_lens stays on the forward stream. total <= bs*(gamma+1) == floor (each
            # verify_len <= gamma+1), so round_up_grid(max(total,floor)) == round_up_grid(
            # floor); downstream reads device verify_lens + graph_num_tokens, no D2H.
            graph_num_tokens = round_up_grid(graph_num_tokens_floor, capture_num_tokens)
            return RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
            )
        # Eager / non-token-keyed fallback (cuda graph off, or non-COMPACT mode): the
        # forward is already synchronous, so this host D2H is harmless -- and the eager
        # `[total]` grid genuinely needs the exact total.
        verify_lens_cpu = verify_lens.to("cpu").tolist()
        grid = verify_layout_grid(
            verify_lens_cpu=verify_lens_cpu,
            ragged_verify_mode=self._ragged_verify_mode,
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
        confidence: Optional[torch.Tensor],
        budget: Optional[int],
    ) -> Optional[torch.Tensor]:
        # Derive per-request verify_lens (= 1 + ell_r) on the GPU from the current
        # step's confidence (lag 0, the sort source) ranked/truncated to the relay-fed
        # budget K (lag 2, the K source; paper §5.2's two distinct survival sources).
        # Broadcast from rank 0 across the (attention) TP group for cross-rank shape
        # consistency (B1). Returns None (-> uniform full block) before the relay /
        # confidence is ready.
        #
        # Losslessness does NOT depend on either source: it is guaranteed by the
        # accept-cap in _cap_correct_len (a torch.minimum that only shrinks accept),
        # after which the bonus is re-read from the target's true distribution at the
        # cap index. The split affects only scheduling quality, never correctness.
        if self._budget_planner is None or confidence is None or budget is None:
            return None
        sort_survival = torch.cumprod(confidence.to(torch.float32), dim=1)
        verify_lens = ScheduleVerifyLensTopk.execute(
            survival_probs=sort_survival,
            budget=budget,
            cfg=self._schedule_cfg,
        ).to(device=device, dtype=torch.int32)

        verify_lens_64 = verify_lens.to(torch.int64)
        # Measure admitted extra against the effective floor max(min_verify_len, 1) so
        # the anchor padding added by the lower-bound clamp is not miscounted as budget
        # overflow when an explicit min_verify_len=0 is clamped up to 1 (B5).
        effective_floor = max(self._schedule_cfg.min_verify_len, 1)
        maybe_assert_async(
            (verify_lens_64 - effective_floor).sum() <= budget,
            f"DSpark verify-len budget violated (budget={budget})",
        )

        if envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER.get():
            self._log_verify_lens_decision(
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                budget=budget,
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
        budget: int,
        sort_survival: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> None:
        cfg = self._schedule_cfg
        max_len = cfg.resolved_max_verify_len()
        req_ids = req_pool_indices.tolist()
        prefixes = prefix_lens.tolist()
        lens = verify_lens.tolist()
        sort_rows = sort_survival.to(torch.float32).tolist()
        logger.info(
            "[DSPARK-CPS] num_reqs=%d budget=%d gamma=%d verify_len_range=[%d,%d]",
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
