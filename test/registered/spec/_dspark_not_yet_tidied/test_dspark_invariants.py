import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    DSparkScheduleConfig,
    compute_verify_token_budget,
    schedule_verify_lens_topk,
)
from sglang.srt.speculative.dspark_components.dspark_sps_table import SpsCostTable
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _survival_from_confidence(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence, dim=1)


def _flat_sps_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1], sample_steps_per_sec=[1.0], max_batch_tokens=256
    )


def _verify_lens(
    *,
    cfg: DSparkScheduleConfig,
    sps_table: SpsCostTable,
    two_steps_prior_k_survival: torch.Tensor,
    sort_survival: torch.Tensor,
) -> torch.Tensor:
    """Test-local compose mirroring the production budget->verify_lens path:
    budget from the two-steps-prior history survival, verify_lens from the current
    sort survival (what HostConfidenceBudgetPlanner does internally)."""
    budget = compute_verify_token_budget(
        history_survival_probs=two_steps_prior_k_survival, sps_table=sps_table, cfg=cfg
    )
    return schedule_verify_lens_topk(
        survival_probs=sort_survival, budget=budget, cfg=cfg
    )


class TestNonAnticipatingScheduler(CustomTestCase):
    """Property tests for the production budget->verify_lens compose (the angle the
    former ConfidencePrefixScheduler wrapper covered).

    Invariant 2 (non-anticipating): the decision to verify position k must not
    depend on the realized token x_{r,k}. The compose only consumes the
    lagged-confidence survival_probs tensor (the host's most recent retired
    snapshot, lag >= 1 step), never the verified tokens. These tests guard the
    invariants (budget adherence, range, determinism, eps gating) that make that safe.
    """

    def test_output_depends_only_on_survival_probs(self):
        """The scheduler sees only survival_probs (never tokens), so the same
        frozen snapshot yields identical verify_lens regardless of what tokens
        are realized -- the structural core of non-anticipating."""
        survival = torch.tensor(
            [[0.9, 0.8, 0.4, 0.1], [0.7, 0.6, 0.3, 0.05]], dtype=torch.float32
        )
        two_steps_prior_k_survival = torch.full((2, 4), 0.8, dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=4)
        sps_table = _flat_sps_table()
        out1 = _verify_lens(
            cfg=cfg,
            sps_table=sps_table,
            two_steps_prior_k_survival=two_steps_prior_k_survival,
            sort_survival=survival,
        )
        out2 = _verify_lens(
            cfg=cfg,
            sps_table=sps_table,
            two_steps_prior_k_survival=two_steps_prior_k_survival,
            sort_survival=survival.clone(),
        )
        self.assertTrue(torch.equal(out1, out2))

    def test_extra_budget_never_exceeds_frozen_budget(self):
        """sum(verify_lens - min_verify_len) <= K budget from two_steps_prior_k_survival (real invariant)."""
        torch.manual_seed(42)
        survival = torch.rand(8, 6, dtype=torch.float32) * 0.9 + 0.05
        two_steps_prior_k_survival = torch.full((8, 6), 0.5, dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=6)
        sps_table = _flat_sps_table()
        budget = compute_verify_token_budget(
            history_survival_probs=two_steps_prior_k_survival,
            sps_table=sps_table,
            cfg=cfg,
        )
        verify_lens = _verify_lens(
            cfg=cfg,
            sps_table=sps_table,
            two_steps_prior_k_survival=two_steps_prior_k_survival,
            sort_survival=survival,
        )
        extra = int((verify_lens - cfg.min_verify_len).sum().item())
        self.assertLessEqual(extra, budget)

    def test_verify_lens_clamped_to_min_max(self):
        """Every verify_len lies in [min_verify_len, max_verify_len]."""
        torch.manual_seed(43)
        survival = torch.rand(5, 4, dtype=torch.float32)
        two_steps_prior_k_survival = torch.full((5, 4), 0.5, dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=4)
        sps_table = _flat_sps_table()
        verify_lens = _verify_lens(
            cfg=cfg,
            sps_table=sps_table,
            two_steps_prior_k_survival=two_steps_prior_k_survival,
            sort_survival=survival,
        )
        self.assertTrue(bool((verify_lens >= cfg.min_verify_len).all()))
        self.assertTrue(bool((verify_lens <= cfg.resolved_max_verify_len()).all()))

    def test_below_eps_positions_do_not_drive_selection(self):
        """A position below survival_eps is a non-candidate: swapping its value
        (while keeping valid positions fixed) must not change verify_lens.

        two_steps_prior_k_survival is chosen so that compute_verify_token_budget returns 2,
        matching the 2 valid candidates in the base sort_survival.  With that
        budget the top-2 slot count per request stays [1, 1] even after
        positions 2-3 are boosted to 0.999, because only 2 extras can be
        admitted regardless."""
        sort_survival = torch.tensor(
            [[0.9, 0.8, 1e-7, 1e-7], [0.7, 0.6, 1e-7, 1e-7]], dtype=torch.float32
        )
        # Two valid candidates in the window (position 1 for each request);
        # flat table -> budget = 2 (= total valid count).
        two_steps_prior_k_survival = torch.tensor(
            [[0.8, 1e-7, 1e-7, 1e-7], [0.6, 1e-7, 1e-7, 1e-7]], dtype=torch.float32
        )
        cfg = DSparkScheduleConfig(gamma=4)
        sps_table = _flat_sps_table()
        base = _verify_lens(
            cfg=cfg,
            sps_table=sps_table,
            two_steps_prior_k_survival=two_steps_prior_k_survival,
            sort_survival=sort_survival,
        )
        # Perturb only the below-eps positions (future, invalid).
        perturbed = sort_survival.clone()
        perturbed[:, 2:] = 0.999
        changed = _verify_lens(
            cfg=cfg,
            sps_table=sps_table,
            two_steps_prior_k_survival=two_steps_prior_k_survival,
            sort_survival=perturbed,
        )
        # Budget-capped at 2; top-2 global selection picks 1 extra per request
        # in both base and perturbed -> verify_lens must be equal.
        self.assertTrue(torch.equal(base, changed))


class TestNonAnticipatingBudgetAllocation(CustomTestCase):
    """Non-anticipating invariants driven through the production
    schedule_verify_lens_topk selector (the angle the old in-file mock used to
    cover): a shared frozen budget is split across requests by survival rank, and
    ties resolve value-independently. These are distinct from
    test_dspark_scheduler.py::TestNonAnticipating, which perturbs the future of a
    single call; here we exercise multi-request budget contention and ties on the
    real selector, feeding survival = cumprod(raw confidence) as production does.
    """

    def test_budget_splits_toward_higher_survival_request(self):
        """A shared budget favors the request whose survival ranks higher."""
        confidence = torch.tensor(
            [[0.99, 0.99, 0.99], [0.50, 0.40, 0.30]], dtype=torch.float32
        )
        survival = _survival_from_confidence(confidence)
        cfg = DSparkScheduleConfig(gamma=3, min_verify_len=1)
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=2, cfg=cfg
        )
        self.assertGreater(
            int(verify_lens[0].item()),
            int(verify_lens[1].item()),
            "budget must prefer the higher-survival request's prefix",
        )

    def test_total_extra_never_exceeds_shared_budget(self):
        """sum(verify_lens - min_verify_len) <= budget across many requests."""
        torch.manual_seed(7)
        confidence = torch.rand(6, 5, dtype=torch.float32) * 0.4 + 0.55
        survival = _survival_from_confidence(confidence)
        cfg = DSparkScheduleConfig(gamma=5, min_verify_len=1)
        budget = 4
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=budget, cfg=cfg
        )
        extra = int((verify_lens.to(torch.int64) - cfg.min_verify_len).sum().item())
        self.assertLessEqual(extra, budget)

    def test_tie_allocation_is_value_independent(self):
        """Equal survival across requests splits the budget deterministically and
        spends exactly the budget (value-independent tie-break)."""
        survival = _survival_from_confidence(
            torch.full((3, 4), 0.8, dtype=torch.float32)
        )
        cfg = DSparkScheduleConfig(gamma=4, min_verify_len=1)
        first = schedule_verify_lens_topk(survival_probs=survival, budget=5, cfg=cfg)
        second = schedule_verify_lens_topk(survival_probs=survival, budget=5, cfg=cfg)
        self.assertTrue(torch.equal(first, second))
        extra = int((first.to(torch.int64) - cfg.min_verify_len).sum().item())
        self.assertEqual(extra, 5)


if __name__ == "__main__":
    unittest.main()
