import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_cp_size,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
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
from sglang.srt.speculative.dspark_components.dspark_sps_online import (
    OnlineSpsProfiler,
)
from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    is_uninitialized_sps_table,
)
from sglang.srt.speculative.dspark_components.dspark_sts_table import (
    load_sts_calibration_from_path,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    dp_tier_budget,
    local_verify_tier_num_tokens,
    ragged_capture_num_tokens,
    ragged_layout_exceeds_captured_grid,
    uniform_ragged_layout,
    verify_layout_graph_num_tokens_floor,
    verify_layout_grid,
    verify_lens_broadcast_group,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    ScheduleVerifyLensTopk,
    compute_sort_survival,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
    round_up_grid,
)
from sglang.srt.utils.async_probe import maybe_assert_async
from sglang.srt.utils.common import require_mlp_tp_gather

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

        self._ragged_verify_mode = read_ragged_verify_mode()
        self._schedule_cfg = DSparkScheduleConfig(gamma=self.gamma)
        self._budget_planner: Optional[HostConfidenceBudgetPlanner] = None
        self._dynamic_graph_tier = False
        self._dp_tier_gather_enabled = False
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
            online_profiler = None
            if envs.SGLANG_DSPARK_ENABLE_SPS_ONLINE_PROFILE.get():
                online_profiler = OnlineSpsProfiler(
                    initial_table=sps_table,
                    rebuild_interval_steps=(
                        envs.SGLANG_DSPARK_SPS_ONLINE_REBUILD_INTERVAL.get()
                    ),
                    min_bin_samples=(
                        envs.SGLANG_DSPARK_SPS_ONLINE_MIN_BIN_SAMPLES.get()
                    ),
                )
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
                online_profiler=online_profiler,
                log_table_swaps=tp_rank == 0,
            )
            # Budget-tiered graph selection: replay the tier round_up(bs +
            # budget) instead of the pinned round_up(bs * (gamma+1)), so a
            # trimmed budget buys a cheaper captured graph. The tier must be
            # identical on every rank of the graph's collective group, and the
            # budget is a rank-local host int. With a shared table the budget
            # is a deterministic function of the TP-replicated confidence,
            # hence rank-consistent without communication; online SPS tables
            # are rank-local so tp>1 budgets diverge and pin to the legacy
            # tier. Under dp-attn ranks see different (bs, budget), so this
            # local gate stays off and the tier instead comes from the
            # dp_tier_num_tokens agreement gathered by the scheduler's MLP
            # sync (see schedule_layout).
            self._dynamic_graph_tier = not is_dp_attention_enabled() and not (
                online_profiler is not None and self.server_args.tp_size > 1
            )
            # Every predicate here is identical on all ranks (server args, env,
            # parallel topology), so ranks agree on whether the per-step tier
            # gather collective runs at all. attn_tp > 1 is excluded: within an
            # attention-TP group the confidence relay can resolve on rank 0 but
            # miss on a peer, and the peer's lens-less pinned fallback cannot
            # rendezvous with a trimmed tier. require_mlp_tp_gather must hold
            # because without it batch.global_num_tokens carries only the LOCAL
            # token count, which breaks every bs_max term the tier agreement is
            # built on. The gather itself runs inside the scheduler's
            # budget-prepare hook, which only fires on the overlap path;
            # PD-disagg / PP event loops have early returns that skip the hook,
            # so those stay on the pinned tier by the same static gate.
            self._dp_tier_gather_enabled = (
                self._ragged_verify_mode is RaggedVerifyMode.COMPACT
                and is_dp_attention_enabled()
                and get_attention_tp_size() == 1
                and get_attention_cp_size() == 1
                and require_mlp_tp_gather(self.server_args)
                and not self.server_args.disable_overlap_schedule
                and not self.server_args.speculative_skip_dp_mlp_sync
                and self.server_args.disaggregation_mode == "null"
                and self.server_args.pp_size == 1
                and not envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
            )
            if tp_rank == 0:
                sps_table_source = (
                    self.server_args.speculative_dspark_sps_table_path
                    or "uninitialized"
                )
                logger.info(
                    "DSpark ragged-verify scheduler enabled (mode=%s, lag=%d, "
                    "relay_lag=%d, sps_table=%s, online_sps_profile=%s, "
                    "graph_tier=%s).",
                    self._ragged_verify_mode.value,
                    self._budget_planner.lag_steps,
                    relay_lag_steps,
                    sps_table_source,
                    online_profiler is not None,
                    (
                        "dynamic"
                        if self._dynamic_graph_tier
                        else (
                            "dp-gathered" if self._dp_tier_gather_enabled else "pinned"
                        )
                    ),
                )
                if is_uninitialized_sps_table(sps_table) and online_profiler is None:
                    logger.warning(
                        "DSpark SPS table is uninitialized (flat) and online "
                        "profiling is disabled: the verify budget degenerates to "
                        "verify-all (zero scheduling gain). Pass a profiled "
                        "--speculative-dspark-sps-table-path or set "
                        "SGLANG_DSPARK_ENABLE_SPS_ONLINE_PROFILE=1."
                    )

    def _require_prep_in_cuda_graph(self) -> None:
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
        confidence_tap: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self._confidence_head is None:
            return None
        compute_confidence_hook = getattr(self.draft_model, "compute_confidence", None)
        if compute_confidence_hook is not None:
            assert (
                confidence_tap is not None
            ), "dsv4 compute_confidence needs the compute_base_logits tap"
            with torch.inference_mode():
                return compute_confidence_hook(
                    anchor_tokens=anchor_tokens,
                    sampled_tokens=draft_tokens,
                    x_post_hc=confidence_tap,
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
        draft_input = batch.spec_info
        if self._budget_planner is None:
            return
        if draft_input is None:
            # Idle batches carry no spec state but must still join the dp tier
            # gather below (collectives need every rank), contributing the
            # neutral 0. A non-empty batch without spec state is anomalous and
            # will take the lens-less pinned fallback, so it must pin everyone
            # via the -1 sentinel instead.
            local_tier_num_tokens = 0 if batch.batch_size() == 0 else -1
            self._maybe_gather_dp_verify_tier(
                batch=batch, local_tier_num_tokens=local_tier_num_tokens
            )
            return
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._budget_planner.note_non_decode_step()
            self._maybe_gather_dp_verify_tier(batch=batch, local_tier_num_tokens=0)
            return
        resolved = future_map.resolve_confidence_cpu(batch)
        draft_input.verify_token_budget = self._budget_from_resolved(
            resolved=resolved, req_pool_indices_cpu=batch.req_pool_indices_cpu
        )
        # The gathered hint and the budget the forward consumes come from this
        # single resolution; re-resolving later could land on a different
        # relay-ring state and break the cross-rank tier agreement.
        batch.spec_verify_tier_num_tokens = local_verify_tier_num_tokens(
            bs=batch.batch_size(),
            verify_token_budget=draft_input.verify_token_budget,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            min_verify_len=self._schedule_cfg.min_verify_len,
        )
        self._maybe_gather_dp_verify_tier(
            batch=batch, local_tier_num_tokens=batch.spec_verify_tier_num_tokens
        )

    def _maybe_gather_dp_verify_tier(
        self, *, batch: ScheduleBatch, local_tier_num_tokens: int
    ) -> None:
        if not self._dp_tier_gather_enabled:
            return
        # is_extend_in_batch is the post-MLP-sync GLOBAL flag, so every rank
        # takes the same branch: spec+dp never mixes prefill and decode steps
        # (and the mixing opt-outs are statically gated off), so on a global
        # prefill step all ranks skip and on a global decode step all ranks
        # gather -- the collective can never be one-sided.
        if batch.is_extend_in_batch:
            # Scrub any list a previous decode step left on this (persistent)
            # batch object so a later consumer can never read a stale tier.
            batch.global_spec_verify_tier_num_tokens = None
            return
        cpu_group = get_tp_group().cpu_group
        local_tensor = torch.tensor([local_tier_num_tokens], dtype=torch.int64)
        gathered = torch.empty(
            (torch.distributed.get_world_size(group=cpu_group),), dtype=torch.int64
        )
        torch.distributed.all_gather_into_tensor(
            gathered, local_tensor, group=cpu_group
        )
        batch.global_spec_verify_tier_num_tokens = gathered.tolist()

    def note_non_decode_step(self) -> None:
        if self._budget_planner is not None:
            self._budget_planner.note_non_decode_step()

    def compute_budget_sync(
        self,
        *,
        confidence: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> Optional[int]:
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
        if resolved is None:
            self._budget_planner.note_non_decode_step()
            return None
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
        global_num_reqs: Optional[int] = None,
        dp_tier_num_tokens: Optional[int] = None,
    ) -> Optional[RaggedVerifyLayout]:
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
            # The pinned uniform fallback and the dp tier agreement must stay
            # mutually exclusive: lens are None iff this rank's budget resolved
            # to None iff this rank contributed the -1 sentinel to the gather
            # iff every rank aggregated dp_tier_num_tokens to None. A non-None
            # tier here would mean this rank replays the pinned tier while
            # other ranks replay the trimmed tier -> collective hang.
            assert dp_tier_num_tokens is None, (
                "dp tier agreement present but local verify lens are None; "
                "the gathered hint and the local budget diverged"
            )
            if self._ragged_verify_mode is RaggedVerifyMode.COMPACT:
                return uniform_ragged_layout(
                    bs=len(req_pool_indices),
                    device=device,
                    verify_num_draft_tokens=self.verify_num_draft_tokens,
                    ragged_verify_mode=self._ragged_verify_mode,
                    model_runner=self.model_runner,
                    tier_num_reqs=global_num_reqs,
                )
            return None
        bs = int(verify_lens.shape[0])
        tier_num_reqs = bs if global_num_reqs is None else global_num_reqs
        min_verify_len = max(self._schedule_cfg.min_verify_len, 1)
        # None keeps the legacy pinned tier round_up(bs * (gamma+1)); a host
        # budget selects the budget-sized tier round_up(bs * min_verify_len +
        # budget). Under dp-attn the local gate is off and the tier instead
        # comes from the gathered agreement, identical on every rank by
        # construction -- but only relative to the GLOBAL request count: with
        # a local bs the small ranks would key a lower tier than their peers.
        tier_budget = budget if self._dynamic_graph_tier else None
        if dp_tier_num_tokens is not None:
            assert global_num_reqs is not None, (
                "dp tier agreement requires the dp-global request count; "
                "keying the tier off the local bs diverges across ranks"
            )
            tier_budget = dp_tier_budget(
                dp_tier_num_tokens=dp_tier_num_tokens,
                tier_num_reqs=tier_num_reqs,
                min_verify_len=min_verify_len,
            )
        if ragged_layout_exceeds_captured_grid(
            num_reqs=tier_num_reqs,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            tier_tokens_hint=(
                None
                if tier_budget is None
                else min(
                    tier_num_reqs * min_verify_len + tier_budget,
                    tier_num_reqs * self.verify_num_draft_tokens,
                )
            ),
        ):
            return None
        graph_num_tokens_floor = verify_layout_graph_num_tokens_floor(
            num_reqs=tier_num_reqs,
            ragged_verify_mode=self._ragged_verify_mode,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            verify_token_budget=tier_budget,
            min_verify_len=min_verify_len,
        )
        capture_num_tokens = ragged_capture_num_tokens(model_runner=self.model_runner)
        if graph_num_tokens_floor > 0 and capture_num_tokens is not None:
            graph_num_tokens = round_up_grid(graph_num_tokens_floor, capture_num_tokens)
            return RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
            )
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
        if self._budget_planner is None or confidence is None or budget is None:
            return None
        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=confidence,
            budget=budget,
            cfg=self._schedule_cfg,
        ).to(device=device, dtype=torch.int32)

        if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
            verify_lens_64 = verify_lens.to(torch.int64)
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
                sort_survival=compute_sort_survival(confidence),
                verify_lens=verify_lens,
            )

        broadcast_group, group_size = verify_lens_broadcast_group(
            tp_size=self.server_args.tp_size
        )
        if group_size > 1:
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
