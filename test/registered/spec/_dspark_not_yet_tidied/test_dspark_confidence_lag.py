import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
    compute_verify_token_budget,
)
from sglang.srt.speculative.dspark_components.dspark_sps_table import SpsCostTable
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    _CONFIDENCE_RELAY_LAG_STEPS,
    _CONFIDENCE_RELAY_RING_DEPTH,
    _CONFIDENCE_RELAY_UNSET_SEQ_LEN,
    DSparkWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER = _REPO_ROOT / "python/sglang/srt/speculative/dspark_worker_v2.py"


def _flat_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1], sample_steps_per_sec=[1.0], max_batch_tokens=4096
    )


def _make_ring_worker(*, gamma: int, req_pool_size: int) -> DSparkWorkerV2:
    """A worker stub with a pre-allocated confidence ring (bypasses model_runner)."""
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.gamma = gamma
    worker.device = torch.device("cpu")
    worker._confidence_step_ct = 0
    worker._confidence_ring = torch.zeros(
        (_CONFIDENCE_RELAY_RING_DEPTH, req_pool_size, gamma), dtype=torch.float32
    )
    worker._confidence_ring_seq_lens = torch.full(
        (_CONFIDENCE_RELAY_RING_DEPTH, req_pool_size),
        _CONFIDENCE_RELAY_UNSET_SEQ_LEN,
        dtype=torch.int64,
    )
    worker._verify_scheduler = ConfidencePrefixScheduler(
        sps_table=_flat_table(), cfg=DSparkScheduleConfig(gamma=gamma)
    )
    worker.server_args = SimpleNamespace(tp_size=1)
    return worker


def _drive_steps(worker, *, req_pool_indices, confidences, prefix_lens) -> None:
    """Stash one confidence/prefix per decode step, advancing step_ct between steps
    exactly as _forward_decode does (stash this step, advance, stash next step)."""
    last = len(confidences) - 1
    for step, (confidence, prefix) in enumerate(zip(confidences, prefix_lens)):
        worker._stash_confidence(
            req_pool_indices=req_pool_indices,
            confidence=confidence,
            prefix_lens=prefix,
        )
        if step < last:
            worker._confidence_step_ct += 1


class TestCrossStepLagBarrier(CustomTestCase):
    """The §5.2 causal barrier: step n's budget K is fixed by the confidence stashed
    _CONFIDENCE_RELAY_LAG_STEPS decode steps earlier, never by step n's own (current)
    confidence. This is the lossless-critical mechanism that had no test before.
    """

    def test_k_source_equals_lag_steps_prior_confidence(self):
        """The K-source survival equals cumprod of the confidence stashed lag steps ago."""
        gamma = 4
        worker = _make_ring_worker(gamma=gamma, req_pool_size=2)
        idx = torch.tensor([0, 1])
        # One confidence per step for lag + 1 steps; the lag-steps-prior step is 0.
        confidences = [
            torch.full((2, gamma), 0.30, dtype=torch.float32),  # step 0 (lag prior)
            torch.full((2, gamma), 0.50, dtype=torch.float32),  # step 1
            torch.full((2, gamma), 0.99, dtype=torch.float32),  # step 2 (current)
        ][: _CONFIDENCE_RELAY_LAG_STEPS + 1]
        prefix_lens = [
            torch.tensor([10 + step, 20 + step]) for step in range(len(confidences))
        ]
        _drive_steps(
            worker,
            req_pool_indices=idx,
            confidences=confidences,
            prefix_lens=prefix_lens,
        )

        k_survival = worker._two_steps_prior_k_survival(
            req_pool_indices=idx, prefix_lens=prefix_lens[-1]
        )
        expected = torch.cumprod(confidences[0].to(torch.float32), dim=1)
        self.assertTrue(torch.allclose(k_survival, expected))

    def test_current_confidence_does_not_change_k_only_sort(self):
        """Re-stashing step n's confidence leaves the K-source unchanged (barrier) but
        moves the sort-source (which tracks the current live confidence)."""
        gamma = 4
        worker = _make_ring_worker(gamma=gamma, req_pool_size=2)
        idx = torch.tensor([0, 1])
        confidences = [
            torch.full((2, gamma), 0.30, dtype=torch.float32),
            torch.full((2, gamma), 0.50, dtype=torch.float32),
            torch.full((2, gamma), 0.70, dtype=torch.float32),
        ][: _CONFIDENCE_RELAY_LAG_STEPS + 1]
        prefix_lens = [
            torch.tensor([10 + step, 20 + step]) for step in range(len(confidences))
        ]
        _drive_steps(
            worker,
            req_pool_indices=idx,
            confidences=confidences,
            prefix_lens=prefix_lens,
        )

        k_before = worker._two_steps_prior_k_survival(
            req_pool_indices=idx, prefix_lens=prefix_lens[-1]
        )
        sort_before = worker._current_live_sort_survival(req_pool_indices=idx)

        # Overwrite the CURRENT step's confidence (no step_ct advance: same step n).
        worker._stash_confidence(
            req_pool_indices=idx,
            confidence=torch.full((2, gamma), 0.10, dtype=torch.float32),
            prefix_lens=prefix_lens[-1],
        )
        k_after = worker._two_steps_prior_k_survival(
            req_pool_indices=idx, prefix_lens=prefix_lens[-1]
        )
        sort_after = worker._current_live_sort_survival(req_pool_indices=idx)

        self.assertTrue(torch.equal(k_before, k_after))
        self.assertFalse(torch.equal(sort_before, sort_after))

    def test_budget_k_is_set_by_prior_not_current_confidence(self):
        """Through the real scheduler: the budget K from the lagged K-source matches
        the lag-prior confidence's budget and differs from the current's."""
        gamma = 4
        worker = _make_ring_worker(gamma=gamma, req_pool_size=2)
        idx = torch.tensor([0, 1])
        # Low lag-prior confidence decays below survival_eps fast -> small budget;
        # high current confidence would yield a larger budget if it leaked into K.
        low = torch.full((2, gamma), 0.005, dtype=torch.float32)
        high = torch.full((2, gamma), 0.99, dtype=torch.float32)
        confidences = ([low] + [high] * _CONFIDENCE_RELAY_LAG_STEPS)[
            : _CONFIDENCE_RELAY_LAG_STEPS + 1
        ]
        prefix_lens = [
            torch.tensor([10 + step, 20 + step]) for step in range(len(confidences))
        ]
        _drive_steps(
            worker,
            req_pool_indices=idx,
            confidences=confidences,
            prefix_lens=prefix_lens,
        )

        k_survival = worker._two_steps_prior_k_survival(
            req_pool_indices=idx, prefix_lens=prefix_lens[-1]
        )
        sps_table = worker._verify_scheduler.sps_table
        cfg = worker._verify_scheduler.cfg
        budget_k = compute_verify_token_budget(
            history_survival_probs=k_survival, sps_table=sps_table, cfg=cfg
        )
        budget_from_low = compute_verify_token_budget(
            history_survival_probs=torch.cumprod(low.to(torch.float32), dim=1),
            sps_table=sps_table,
            cfg=cfg,
        )
        budget_from_high = compute_verify_token_budget(
            history_survival_probs=torch.cumprod(high.to(torch.float32), dim=1),
            sps_table=sps_table,
            cfg=cfg,
        )
        self.assertNotEqual(
            budget_from_low,
            budget_from_high,
            "degenerate scenario: low and current confidence yield the same budget",
        )
        self.assertEqual(budget_k, budget_from_low)


class TestRingIdentityGuard(CustomTestCase):
    """H1: a ring slot is keyed by req-pool row, so the lag-prior occupant of a row may
    be a different request. The prefix_len identity stamp masks such rows (and cold
    starts) to a verify-all fallback rather than carrying a stranger's confidence into K.
    """

    def test_same_request_row_uses_its_lag_prior_confidence(self):
        """A row continuously owned by one request keeps its lag-prior survival."""
        gamma = 4
        worker = _make_ring_worker(gamma=gamma, req_pool_size=2)
        idx = torch.tensor([0, 1])
        confidences = [
            torch.full((2, gamma), 0.30, dtype=torch.float32),
            torch.full((2, gamma), 0.50, dtype=torch.float32),
            torch.full((2, gamma), 0.80, dtype=torch.float32),
        ][: _CONFIDENCE_RELAY_LAG_STEPS + 1]
        # prefix grows by one token per step -> valid ancestor.
        prefix_lens = [
            torch.tensor([100 + step, 200 + step]) for step in range(len(confidences))
        ]
        _drive_steps(
            worker,
            req_pool_indices=idx,
            confidences=confidences,
            prefix_lens=prefix_lens,
        )
        k_survival = worker._two_steps_prior_k_survival(
            req_pool_indices=idx, prefix_lens=prefix_lens[-1]
        )
        expected = torch.cumprod(confidences[0].to(torch.float32), dim=1)
        self.assertTrue(torch.allclose(k_survival, expected))

    def test_reused_row_for_new_request_falls_back_to_verify_all(self):
        """A row whose lag-prior prefix_len is not an ancestor of the current prefix
        (a different request took the row) falls back to survival = 1.0 (verify-all)."""
        gamma = 4
        worker = _make_ring_worker(gamma=gamma, req_pool_size=2)
        idx = torch.tensor([0, 1])
        confidences = [
            torch.full((2, gamma), 0.30, dtype=torch.float32),
            torch.full((2, gamma), 0.50, dtype=torch.float32),
            torch.full((2, gamma), 0.80, dtype=torch.float32),
        ][: _CONFIDENCE_RELAY_LAG_STEPS + 1]
        # Row 0: lag-prior prefix 5000, current prefix 5 -> growth negative -> stale.
        # Row 1: a valid same-request ancestor (grows by one per step).
        prefix_lens = [
            torch.tensor([5000, 200 + step]) for step in range(len(confidences))
        ]
        prefix_lens[-1] = torch.tensor([5, 200 + len(confidences) - 1])
        _drive_steps(
            worker,
            req_pool_indices=idx,
            confidences=confidences,
            prefix_lens=prefix_lens,
        )
        k_survival = worker._two_steps_prior_k_survival(
            req_pool_indices=idx, prefix_lens=prefix_lens[-1]
        )
        # Row 0 masked to verify-all; row 1 keeps its lag-prior survival.
        self.assertTrue(torch.allclose(k_survival[0], torch.ones(gamma)))
        self.assertTrue(
            torch.allclose(
                k_survival[1], torch.cumprod(confidences[0][1].to(torch.float32), dim=0)
            )
        )

    def test_cold_start_first_steps_fall_back_to_verify_all(self):
        """Before lag steps of history exist, the lag-prior slot is unwritten so K
        falls back to verify-all for every request."""
        gamma = 4
        worker = _make_ring_worker(gamma=gamma, req_pool_size=2)
        idx = torch.tensor([0, 1])
        worker._stash_confidence(
            req_pool_indices=idx,
            confidence=torch.full((2, gamma), 0.30, dtype=torch.float32),
            prefix_lens=torch.tensor([10, 20]),
        )
        # Step 0: read slot (0 - lag) % depth is still all-sentinel -> verify-all.
        k_survival = worker._two_steps_prior_k_survival(
            req_pool_indices=idx, prefix_lens=torch.tensor([10, 20])
        )
        self.assertTrue(torch.allclose(k_survival, torch.ones((2, gamma))))


class TestConfidenceRelayRingTopology(CustomTestCase):
    """Topology guard: the relay is a step-indexed ring (depth >= 2), NOT the legacy
    single _confidence_buf. This deliberately reverses the prior anti-pattern test that
    enshrined a single buffer / "never a structured step-indexed n-2 double buffer".
    """

    def _relay_init_source(self) -> str:
        text = _WORKER.read_text()
        start = text.find("def _ensure_confidence_relay_buffers(self")
        self.assertNotEqual(start, -1, "_ensure_confidence_relay_buffers not found.")
        end = text.find("\n    def ", start + 1)
        return text[start : end if end != -1 else len(text)]

    def test_ring_depth_is_at_least_two(self):
        """The ring depth exceeds the lag and is >= 2 (a structured n-step buffer)."""
        self.assertGreaterEqual(_CONFIDENCE_RELAY_RING_DEPTH, 2)
        self.assertGreater(_CONFIDENCE_RELAY_RING_DEPTH, _CONFIDENCE_RELAY_LAG_STEPS)

    def test_relay_allocates_a_depth_indexed_ring(self):
        """_ensure_confidence_relay_buffers allocates a [depth, ...] ring, not a single
        2D buffer."""
        block = self._relay_init_source()
        self.assertIn("_CONFIDENCE_RELAY_RING_DEPTH", block)
        self.assertEqual(
            len(re.findall(r"self\._confidence_ring\s*=\s*torch\.empty", block)),
            1,
            "relay must allocate exactly one step-indexed ring.",
        )

    def test_legacy_single_buffer_relay_is_gone(self):
        """The legacy single-buffer relay (and its dead host-copy method) are removed."""
        text = _WORKER.read_text()
        self.assertNotIn("self._confidence_buf =", text)
        self.assertNotIn("def pull_confidence_history", text)


if __name__ == "__main__":
    unittest.main()
