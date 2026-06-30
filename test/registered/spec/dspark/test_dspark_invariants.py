import re
import unittest
from pathlib import Path

import pytest
import torch

from sglang.srt.speculative.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
    schedule_verify_lens_topk,
)
from sglang.srt.speculative.dspark_sps_table import SpsCostTable
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _survival_from_confidence(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence, dim=1)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DECODE_RUNNER = (
    _REPO_ROOT / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
)
_SCHEDULER = _REPO_ROOT / "python/sglang/srt/managers/scheduler.py"


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


class TestWarBarrierCapability(CustomTestCase):
    """Invariant 3: the WAR/RAW barrier capability gate must cover DSPARK so the
    publisher branch actually records the read-done event for verify replay.
    """

    def test_dspark_supports_overalloc_war_verify(self):
        """supports_overalloc_war_verify() must return True for DSPARK."""
        algo = SpeculativeAlgorithm.from_string("DSPARK")
        self.assertTrue(
            algo.supports_overalloc_war_verify(),
            "DSPARK must qualify for the over-alloc WAR verify barrier.",
        )

    def test_dflash_also_supports_overalloc_war_verify(self):
        """DFLASH shares the same WAR verify capability (sibling algorithm)."""
        algo = SpeculativeAlgorithm.from_string("DFLASH")
        self.assertTrue(algo.supports_overalloc_war_verify())

    def test_eagle_does_not_support_overalloc_war_verify(self):
        """Non-block-draft algorithms (EAGLE) do not over-alloc verify replay."""
        algo = SpeculativeAlgorithm.from_string("EAGLE")
        self.assertFalse(algo.supports_overalloc_war_verify())


class TestWarBarrierAntiPattern(CustomTestCase):
    """Invariant 3 anti-pattern guard: the verify replay path must reuse the
    single existing read-done event channel. real-N must NOT introduce a second
    Event() / wait_stream (capability-over-special-case).
    """

    def _read_replay_block(self) -> str:
        """Return the source of decode_cuda_graph_runner's replay_session block."""
        text = _DECODE_RUNNER.read_text()
        start = text.find("with timer_ctx, self.backend.replay_session():")
        self.assertNotEqual(start, -1, "replay_session block not found.")
        end = text.find("\n\n", start)
        return text[start : end if end != -1 else len(text)]

    def test_single_read_done_event_in_replay_block(self):
        """The replay_session block records exactly one read-done Event()."""
        block = self._read_replay_block()
        event_count = len(re.findall(r"self\.device_module\.Event\(\)", block))
        self.assertEqual(
            event_count,
            1,
            "Verify replay must use exactly one read-done Event() (no second "
            "fence for real-N ragged metadata).",
        )

    def test_no_extra_wait_stream_in_replay_block(self):
        """The replay_session block must not add a private wait_stream fence."""
        block = self._read_replay_block()
        self.assertNotIn(
            "wait_stream",
            block,
            "Verify replay must not introduce a private wait_stream fence.",
        )

    def test_publisher_gates_on_war_capability(self):
        """The read-done publisher branch is gated on supports_overalloc_war_verify."""
        block = self._read_replay_block()
        self.assertIn(
            "supports_overalloc_war_verify()",
            block,
            "Publisher branch must gate on the WAR verify capability predicate.",
        )

    def test_scheduler_consumer_waits_on_read_done_event(self):
        """Scheduler _apply_war_barrier consumes the published read-done event."""
        text = _SCHEDULER.read_text()
        start = text.find("def _apply_war_barrier(self):")
        self.assertNotEqual(start, -1, "_apply_war_barrier not found.")
        end = text.find("\n    def ", start + 1)
        block = text[start : end if end != -1 else len(text)]
        self.assertIn("war_fastpath_read_done_event", block)
        self.assertIn("wait_event", block)


@unittest.skip(
    "BLOCKED on ragged-verify routing decision: the real-N (`compact`) path that "
    "writes ragged metadata buffers (verify_lens / extend_start_loc / "
    "qo_indptr_device) inside cuda-graph replay is under team design discussion. "
    "The WAR/RAW timing regression for real-N is stubbed and not wired to run."
)
class TestWarBarrierRealNTimingStub(CustomTestCase):
    """Invariant 3 real-N regression (BLOCKED): under spec-v2 overlap, the
    scheduler's write to the ragged metadata buffers must wait on the existing
    war_fastpath_read_done_event before overwriting, and multi-step output must
    be bit-equal to cap-accept under the same lagged ell_r.
    """

    def test_real_n_metadata_copy_precedes_read_done(self):
        """real-N ragged metadata copy_ must complete before read_done.record()."""
        pytest.skip("blocked on ragged-verify routing decision")

    def test_real_n_overlap_bit_equal_to_cutoff_only(self):
        """spec-v2 overlap real-N multi-step output must equal cap-accept."""
        pytest.skip("blocked on ragged-verify routing decision")


if __name__ == "__main__":
    unittest.main()
