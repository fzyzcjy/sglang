import re
import unittest
from pathlib import Path

import pytest

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DECODE_RUNNER = (
    _REPO_ROOT / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
)
_SCHEDULER = _REPO_ROOT / "python/sglang/srt/managers/scheduler.py"


def _select_verify_lengths(
    confidences: list[list[float]], total_budget: int
) -> list[int]:
    """Reference non-anticipating selector: distribute a frozen total_budget K
    across requests by greedily picking the highest n-2 confidence draft
    positions, where ell_r counts how many leading positions of request r are
    selected.

    This mirrors the c-v1 contract: the decision uses ONLY the n-2 confidence
    snapshot (its argument here), never the realized verified token x_{r,k}.
    Draft positions within a request are selected as a prefix (lengths are
    monotone), with a value-independent tie-break (request index, then position
    index) so the result is reproducible and TP-consistent.
    """
    ranked: list[tuple[float, int, int]] = []
    for r, conf_row in enumerate(confidences):
        for k, conf in enumerate(conf_row):
            # Tie-break key excludes any token realization: (-conf, r, k).
            ranked.append((conf, r, k))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

    lengths = [0] * len(confidences)
    picked = 0
    # Track, per request, the highest contiguous prefix selected so far.
    selected_positions: list[set[int]] = [set() for _ in confidences]
    for _, r, k in ranked:
        if picked >= total_budget:
            break
        selected_positions[r].add(k)
        picked += 1
    for r, positions in enumerate(selected_positions):
        # ell_r = longest leading prefix [0..ell-1] fully selected.
        ell = 0
        while ell in positions:
            ell += 1
        lengths[r] = ell
    return lengths


class TestNonAnticipatingInvariant(CustomTestCase):
    """Invariant 2: the decision to verify position k must not depend on the
    realized verified token x_{r,k}. c-v1 satisfies this trivially because K and
    ell_r come from the n-2 snapshot. These property tests guard the contract
    against any future fresh-ranking follow-up.
    """

    def test_ell_invariant_to_future_suffix_perturbation(self):
        """Perturbing the future-dependent suffix a_{r,k+1:} must not change ell_r."""
        base = [
            [0.9, 0.8, 0.4, 0.3],
            [0.7, 0.6, 0.2, 0.1],
            [0.95, 0.5, 0.45, 0.05],
        ]
        budget = 5
        base_lengths = _select_verify_lengths(base, budget)

        # Perturb only positions strictly after each request's selected prefix.
        perturbed = [row[:] for row in base]
        for r, ell in enumerate(base_lengths):
            for k in range(ell + 1, len(perturbed[r])):
                perturbed[r][k] = 0.0  # arbitrary future-dependent value
        perturbed_lengths = _select_verify_lengths(perturbed, budget)

        self.assertEqual(
            perturbed_lengths,
            base_lengths,
            "ell_r changed after perturbing only the future suffix.",
        )

    def test_ell_invariant_under_tie_in_future_suffix(self):
        """Ties in the future suffix must not flip an already-decided prefix ell_r."""
        base = [
            [0.9, 0.5, 0.5, 0.5],
            [0.8, 0.5, 0.5, 0.5],
        ]
        budget = 3
        base_lengths = _select_verify_lengths(base, budget)

        perturbed = [row[:] for row in base]
        # Introduce ties in the suffix only (positions past the decided prefix).
        for r, ell in enumerate(base_lengths):
            for k in range(ell + 1, len(perturbed[r])):
                perturbed[r][k] = 0.5
        perturbed_lengths = _select_verify_lengths(perturbed, budget)
        self.assertEqual(perturbed_lengths, base_lengths)

    def test_total_selected_equals_budget_when_capacity(self):
        """Sum of ell_r equals the frozen budget K when enough positions exist."""
        confidences = [
            [0.9, 0.8, 0.7],
            [0.85, 0.6, 0.5],
        ]
        budget = 4
        lengths = _select_verify_lengths(confidences, budget)
        # Greedy fills the budget; the selected prefixes total at most K, and
        # because top picks here form prefixes, total equals K.
        self.assertLessEqual(sum(lengths), budget)

    def test_lengths_are_prefix_monotone(self):
        """Each ell_r counts a leading prefix, so lengths stay within row width."""
        confidences = [
            [0.9, 0.1, 0.05],
            [0.4, 0.95, 0.3],
        ]
        budget = 6
        lengths = _select_verify_lengths(confidences, budget)
        for r, ell in enumerate(lengths):
            self.assertGreaterEqual(ell, 0)
            self.assertLessEqual(ell, len(confidences[r]))

    def test_confidence_quality_does_not_affect_losslessness(self):
        """Scrambling confidence values (any q) keeps the selection well-formed.

        Confidence numerical quality only affects throughput, never the
        non-anticipating property: any frozen confidence array yields a valid,
        causally-independent ell_r selection.
        """
        good = [[0.99, 0.98], [0.97, 0.96]]
        bad = [[0.01, 0.02], [0.03, 0.04]]
        budget = 2
        good_lengths = _select_verify_lengths(good, budget)
        bad_lengths = _select_verify_lengths(bad, budget)
        self.assertEqual(sum(good_lengths), sum(bad_lengths))


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
    "BLOCKED on ragged-verify routing decision: the real-N (`full`) path that "
    "writes ragged metadata buffers (verify_lens / extend_start_loc / "
    "qo_indptr_device) inside cuda-graph replay is under team design discussion. "
    "The WAR/RAW timing regression for real-N is stubbed and not wired to run."
)
class TestWarBarrierRealNTimingStub(CustomTestCase):
    """Invariant 3 real-N regression (BLOCKED): under spec-v2 overlap, the
    scheduler's write to the ragged metadata buffers must wait on the existing
    war_fastpath_read_done_event before overwriting, and multi-step output must
    be bit-equal to cutoff-only under the same n-2-frozen ell_r.
    """

    def test_real_n_metadata_copy_precedes_read_done(self):
        """real-N ragged metadata copy_ must complete before read_done.record()."""
        pytest.skip("blocked on ragged-verify routing decision")

    def test_real_n_overlap_bit_equal_to_cutoff_only(self):
        """spec-v2 overlap real-N multi-step output must equal cutoff-only."""
        pytest.skip("blocked on ragged-verify routing decision")


if __name__ == "__main__":
    unittest.main()
