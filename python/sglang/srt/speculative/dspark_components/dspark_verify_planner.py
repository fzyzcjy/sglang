import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_confidence import (
    _CONFIDENCE_RELAY_LAG_STEPS,
    _CONFIDENCE_RELAY_RING_DEPTH,
    ConfidenceRelay,
    compute_confidence,
)
from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
    build_sps_cost_table,
    compute_verify_token_budget,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    ragged_layout_exceeds_captured_grid,
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
        self.server_args = server_args
        self.verify_num_draft_tokens = verify_num_draft_tokens

        # Optional confidence relay. The head and its relay are inert (no
        # buffers, no events, no compute) when the draft model lacks a
        # confidence head, so the a+b lossless decode path is unchanged.
        self._confidence_head = getattr(self.draft_model, "confidence_head", None)
        self._confidence_relay = ConfidenceRelay(
            device=device,
            gamma=self.gamma,
            model_runner=self.model_runner,
        )
        if self._confidence_head is not None and tp_rank == 0:
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
                    tp_rank=tp_rank,
                ),
                cfg=DSparkScheduleConfig(gamma=self.gamma),
            )
            if tp_rank == 0:
                logger.info(
                    "DSpark ragged-verify scheduler enabled (mode=%s).",
                    self._ragged_verify_mode.value,
                )

    @property
    def carries_confidence(self) -> bool:
        return self._confidence_head is not None

    def advance_step(self) -> None:
        self._confidence_relay.advance_step()

    def should_run_compact(self, *, layout: Optional[RaggedVerifyLayout]) -> bool:
        return (
            self._ragged_verify_mode is RaggedVerifyMode.COMPACT and layout is not None
        )

    def relay_confidence(
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

    def schedule_layout(
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
        #   - K source = two-steps-prior survival (two_steps_prior_k_survival):
        #     fixes the verify budget K, causally independent of this step's tokens.
        #   - sort source = current live survival (_current_live_sort_survival):
        #     ranks/truncates admission by the actual up-to-date confidence.
        # budget is computed from two_steps_prior_k_survival now; verify_lens is ranked by
        # sort_survival now — no cross-step cache.
        # Losslessness does NOT depend on either source: it is guaranteed by the
        # accept-cap in _cap_correct_len (a torch.minimum that only shrinks accept),
        # after which the bonus is re-read from the target's true distribution at the
        # cap index. The split affects only scheduling quality, never correctness.
        if self._verify_scheduler is None:
            return None
        two_steps_prior_k_survival = self._confidence_relay.two_steps_prior_k_survival(
            req_pool_indices=req_pool_indices, prefix_lens=prefix_lens
        )
        sort_survival = self._confidence_relay.current_live_sort_survival(
            req_pool_indices=req_pool_indices
        )
        if two_steps_prior_k_survival is None or sort_survival is None:
            return None

        verify_lens = self._verify_scheduler.compute_verify_lens(
            two_steps_prior_k_survival=two_steps_prior_k_survival,
            sort_survival=sort_survival,
        ).to(device=device, dtype=torch.int32)

        if envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER.get():
            self._log_verify_lens_decision(
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                two_steps_prior_k_survival=two_steps_prior_k_survival,
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
        two_steps_prior_k_survival: torch.Tensor,
        sort_survival: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> None:
        cfg = self._verify_scheduler.cfg
        budget = compute_verify_token_budget(
            history_survival_probs=two_steps_prior_k_survival,
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
