import unittest

import torch

from sglang.srt.speculative.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
    schedule_verify_lens_topk,
)
from sglang.srt.speculative.dspark_sps_table import SpsCostTable
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _survival_from_confidence(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence, dim=1)


def _flat_sps_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1], sample_steps_per_sec=[1.0], max_batch_tokens=256
    )


class TestNonAnticipatingScheduler(CustomTestCase):
    """Property tests against the real ConfidencePrefixScheduler.

    Invariant 2 (non-anticipating): the decision to verify position k must not
    depend on the realized token x_{r,k}. c-v1 satisfies this structurally --
    compute_verify_lens only consumes the lagged-confidence survival_probs tensor
    (the host's most recent retired snapshot, lag >= 1 step), never the verified
    tokens. These tests guard the real scheduler's invariants (budget adherence,
    range, determinism, eps gating) that make that safe.
    """

    def _scheduler(self, gamma=4, budget=3) -> ConfidencePrefixScheduler:
        cfg = DSparkScheduleConfig(gamma=gamma)
        sched = ConfidencePrefixScheduler(sps_table=_flat_sps_table(), cfg=cfg)
        sched.cached_budget = budget
        return sched

    def test_output_depends_only_on_survival_probs(self):
        """The scheduler sees only survival_probs (never tokens), so the same
        frozen snapshot yields identical verify_lens regardless of what tokens
        are realized -- the structural core of non-anticipating."""
        survival = torch.tensor(
            [[0.9, 0.8, 0.4, 0.1], [0.7, 0.6, 0.3, 0.05]], dtype=torch.float32
        )
        sched = self._scheduler(gamma=4, budget=3)
        out1 = sched.compute_verify_lens(survival_probs=survival)
        out2 = sched.compute_verify_lens(survival_probs=survival.clone())
        self.assertTrue(torch.equal(out1, out2))

    def test_extra_budget_never_exceeds_frozen_budget(self):
        """sum(verify_lens - min_verify_len) <= frozen budget K (real invariant)."""
        survival = torch.rand(8, 6, dtype=torch.float32) * 0.9 + 0.05
        budget = 5
        sched = self._scheduler(gamma=6, budget=budget)
        verify_lens = sched.compute_verify_lens(survival_probs=survival)
        extra = int((verify_lens - sched.cfg.min_verify_len).sum().item())
        self.assertLessEqual(extra, budget)

    def test_verify_lens_clamped_to_min_max(self):
        """Every verify_len lies in [min_verify_len, max_verify_len]."""
        survival = torch.rand(5, 4, dtype=torch.float32)
        sched = self._scheduler(gamma=4, budget=10)
        verify_lens = sched.compute_verify_lens(survival_probs=survival)
        self.assertTrue(bool((verify_lens >= sched.cfg.min_verify_len).all()))
        self.assertTrue(
            bool((verify_lens <= sched.cfg.resolved_max_verify_len()).all())
        )

    def test_below_eps_positions_do_not_drive_selection(self):
        """A position below survival_eps is a non-candidate: swapping its value
        (while keeping valid positions fixed) must not change verify_lens."""
        survival = torch.tensor(
            [[0.9, 0.8, 1e-7, 1e-7], [0.7, 0.6, 1e-7, 1e-7]], dtype=torch.float32
        )
        sched = self._scheduler(gamma=4, budget=2)
        base = sched.compute_verify_lens(survival_probs=survival)
        # Perturb only the below-eps positions (future, invalid).
        perturbed = survival.clone()
        perturbed[:, 2:] = 0.999
        changed = sched.compute_verify_lens(survival_probs=perturbed)
        # Valid positions unchanged -> selection must not change.
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
