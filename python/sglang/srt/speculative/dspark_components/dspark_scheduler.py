from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    SpsCostTable,
    load_sps_table_from_path,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    _value_independent_descending_order as _value_independent_descending_order,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    schedule_verify_lens_topk as schedule_verify_lens_topk,
)


class DSparkScheduleConfig(msgspec.Struct):
    gamma: int
    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6

    def resolved_max_verify_len(self) -> int:
        # The full verify window is gamma+1 (anchor + all gamma drafts) =
        # num_draft_tokens, which is what static verifies. Capping at gamma would
        # verify only gamma-1 drafts (the last draft never checked) and drop accept
        # below static.
        return self.max_verify_len or (self.gamma + 1)

    def validate(self) -> None:
        max_len = self.resolved_max_verify_len()
        if self.gamma < 1:
            raise ValueError(f"DSpark gamma must be >= 1, got {self.gamma}.")
        if not (0 <= self.min_verify_len <= max_len <= self.gamma + 1):
            raise ValueError(
                "DSpark verify-len config must satisfy 0 <= min <= max <= gamma+1, "
                f"got min={self.min_verify_len}, max={max_len}, gamma={self.gamma}."
            )
        if self.survival_eps < 0:
            raise ValueError(f"survival_eps must be >= 0, got {self.survival_eps}.")


def compute_verify_token_budget(
    *,
    history_survival_probs: torch.Tensor,
    sps_table: SpsCostTable,
    cfg: DSparkScheduleConfig,
) -> int:
    # cfg is validated once at planner construction and is immutable, so skip the
    # per-step re-validation on this hot path (called every decode step).
    num_requests = history_survival_probs.shape[0]
    max_len = cfg.resolved_max_verify_len()

    # Candidates span all gamma draft slots (cols 0..gamma-1) so the budget can reach
    # gamma (verify the full window); slicing [min_verify_len:max_len] gives gamma-1
    # slots and caps the budget one draft short.
    candidates = history_survival_probs[:, :max_len].flatten()
    candidates = candidates[candidates >= cfg.survival_eps].to(torch.float64)
    candidates_sorted = torch.sort(candidates, descending=True).values
    prefix_sum = torch.cumsum(candidates_sorted, dim=0)

    best_extra = 0
    best_theta = float("-inf")
    for extra in range(candidates_sorted.numel() + 1):
        top_sum = float(prefix_sum[extra - 1]) if extra > 0 else 0.0
        tau_star = num_requests + top_sum
        theta = tau_star * sps_table.lookup(num_requests + extra)
        if theta > best_theta:
            best_theta = theta
            best_extra = extra

    return best_extra


class HostConfidenceBudgetPlanner:
    """Host-side verify-budget source (paper section 5.2 two-steps-prior barrier).

    Owns a per-request-row host carry that shifts the FutureMap relay's confidence
    to the configured causal lag (default 2), applies the exact same-request guard
    (req-pool occupancy generation: use the relayed confidence iff the slot still
    holds the same request -- a recycled slot's generation differs), and runs the
    pure-CPU greedy. Everything is a host tensor, so the budget K is produced with
    zero D2H sync. Two feed paths share the carry + guard + greedy:

    - overlap: ``prepare_budget(resolved, req_pool_indices_cpu)`` consumes
      ``FutureMap.resolve_confidence_cpu`` in the scheduler prepare window.
    - non-overlap: ``compute_budget(...)`` is fed from a synchronous worker-side
      ``.cpu()`` (the async relay is absent without overlap).

    Losslessness never depends on the budget (guaranteed by the accept-cap in
    _cap_correct_len); the carry only affects scheduling quality and the lag.
    """

    def __init__(
        self,
        *,
        sps_table: SpsCostTable,
        cfg: DSparkScheduleConfig,
        model_runner,
        relay_lag_steps: int = 1,
    ) -> None:
        cfg.validate()
        self.sps_table = sps_table
        self.cfg = cfg
        # The carry buffer is sized from the req-pool, but req_to_token_pool is not
        # populated until after model-runner init (this planner is built during
        # scheduler/worker __init__), so the size is read lazily in _ensure_carry
        # (mirrors the former ConfidenceRelay.ensure_buffers).
        self._model_runner = model_runner
        # Total causal lag (>= 1 already yields the barrier; default 2 reproduces the
        # paper, tunable via env). The feed already supplies relay_lag_steps of lag (1
        # under the async overlap relay, 0 in the synchronous non-overlap fallback);
        # the host carry supplies the remainder.
        self.lag_steps = max(
            int(envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.get()), 1
        )
        self.carry_steps = max(self.lag_steps - int(relay_lag_steps), 0)
        self._carry_confidence: Optional[torch.Tensor] = None
        self._carry_generation: Optional[torch.Tensor] = None
        self._carry_pos = 0

    def compute_budget(
        self,
        *,
        confidence: torch.Tensor,
        generation: torch.Tensor,
        current_generation: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> int:
        # confidence [bs, gamma], generation [bs] (the req-pool occupancy generation
        # stamped with that confidence), current_generation [bs] (each slot's gen NOW)
        # -- all host, the relay's snapshot for this batch's rows. Shift to lag, guard,
        # greedy. req_pool_indices_cpu scatters/gathers the per-row carry.
        lagged_confidence, lagged_generation = self._shift_to_lag(
            confidence=confidence,
            generation=generation,
            req_pool_indices_cpu=req_pool_indices_cpu,
        )
        survival = self._two_steps_prior_survival(
            lagged_confidence=lagged_confidence,
            lagged_generation=lagged_generation,
            current_generation=current_generation,
        )
        return compute_verify_token_budget(
            history_survival_probs=survival,
            sps_table=self.sps_table,
            cfg=self.cfg,
        )

    def _shift_to_lag(
        self,
        *,
        confidence: torch.Tensor,
        generation: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # carry_steps == 0 (relay already supplies the full lag): use the relayed
        # value directly. Otherwise read the carry slot written carry_steps steps ago
        # for these rows (= the lag-steps-prior confidence + its occupancy generation),
        # then write this step's relayed value back. Rows idle for a cycle keep a stale
        # carry whose generation the freshness guard rejects.
        if self.carry_steps == 0:
            return confidence, generation
        self._ensure_carry(gamma=confidence.shape[-1])
        slot = self._carry_pos % self.carry_steps
        rows = req_pool_indices_cpu.to(torch.int64)
        lagged_confidence = self._carry_confidence[slot, rows].clone()
        lagged_generation = self._carry_generation[slot, rows].clone()
        self._carry_confidence[slot, rows] = confidence.to(torch.float32)
        self._carry_generation[slot, rows] = generation.to(torch.int64)
        self._carry_pos += 1
        return lagged_confidence, lagged_generation

    def _two_steps_prior_survival(
        self,
        *,
        lagged_confidence: torch.Tensor,
        lagged_generation: torch.Tensor,
        current_generation: torch.Tensor,
    ) -> torch.Tensor:
        # cumprod of the lag-steps-prior confidence, gated by the exact same-request
        # check: fresh iff the stamped occupancy generation equals the slot's current
        # generation and the slot is live (current_gen >= 1). Stale / recycled / cold-
        # start rows fall back to verify-all (survival = 1.0). No seq_len, no coincidence.
        k_survival = torch.cumprod(lagged_confidence.to(torch.float32), dim=1)
        current_gen = current_generation.to(torch.int64)
        fresh = (
            (current_gen >= 1) & (lagged_generation.to(torch.int64) == current_gen)
        ).view(-1, 1)
        return torch.where(fresh, k_survival, torch.ones_like(k_survival))

    def _ensure_carry(self, *, gamma: int) -> None:
        if self._carry_confidence is not None:
            return
        req_pool_size = int(self._model_runner.req_to_token_pool.req_to_token.shape[0])
        self._carry_confidence = torch.zeros(
            (self.carry_steps, req_pool_size, gamma), dtype=torch.float32
        )
        # Init 0 (= no occupancy); a never-written carry row mismatches any live
        # generation (>= 1), so the guard rejects it -> verify-all.
        self._carry_generation = torch.zeros(
            (self.carry_steps, req_pool_size),
            dtype=torch.int64,
        )


def build_sps_cost_table(
    *,
    server_args: ServerArgs,
    verify_num_draft_tokens: int,
) -> SpsCostTable:
    # A real --speculative-dspark-sps-table-path loads the pre-profiled,
    # hardware-aware table; the literal "const" sentinel deliberately opts into
    # a flat constant-SPS table (budget = verify-all-up-to-gamma, zero
    # throughput gain); anything else unset raises. The expected scheduler-on
    # workflow is to build the table offline with
    # sglang.benchmark.dspark_sps_profiler and pass it via
    # --speculative-dspark-sps-table-path (see
    # docs/advanced_features/dspark_sps_table.md).
    #
    # The flat "const" table makes the hardware-aware scheduler a no-op:
    # lookup() returns a constant, so the verify-token budget degenerates to
    # verify-all and every request keeps verify_len == gamma+1 (resolved_max_verify_len
    # = gamma+1, so compact verifies the anchor plus all gamma drafts = the full
    # window, matching static; this is lossless -- _cap_correct_len caps accept and
    # the bonus is re-read from the target distribution). The
    # verify_lens >= 1 anchor contract (see DSparkScheduleConfig.min_verify_len
    # and schedule_verify_lens_topk's lower-bound clamp) MUST be in place before
    # any profiled table is supplied, because a non-flat table yields small K
    # and would otherwise drive verify_len to 0.
    sps_table_path = server_args.speculative_dspark_sps_table_path
    if not sps_table_path:
        raise ValueError(
            "DSpark ragged-verify scheduler is enabled (mode != static) but "
            "--speculative-dspark-sps-table-path was not supplied. Build a "
            "hardware-aware SPS cost table offline against a plain (non-speculative) "
            "server with `python -m sglang.benchmark.dspark_sps_profiler` and pass "
            "its JSON path (see docs/advanced_features/dspark_sps_table.md). To "
            "deliberately run with a flat constant-SPS table instead "
            "(verify-all-up-to-gamma, zero throughput gain), pass "
            "--speculative-dspark-sps-table-path=const."
        )
    if sps_table_path != "const":
        return load_sps_table_from_path(sps_table_path)
    max_batch_tokens = max(
        1,
        int(server_args.max_running_requests or 1) * verify_num_draft_tokens,
    )
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[1.0],
        max_batch_tokens=max_batch_tokens,
    )
