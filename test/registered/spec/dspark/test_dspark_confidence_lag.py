import contextlib
import re
import unittest
from pathlib import Path

import torch

from sglang.srt.speculative.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
)
from sglang.srt.speculative.dspark_sps_table import SpsCostTable
from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER = _REPO_ROOT / "python/sglang/srt/speculative/dspark_worker_v2.py"


class _FakeEvent:
    """A CUDA-event stub whose query() result the test controls; record() bumps
    a counter so we can assert it was (not) called."""

    def __init__(self, *, ready: bool) -> None:
        self.ready = ready
        self.record_count = 0

    def record(self) -> None:
        self.record_count += 1

    def query(self) -> bool:
        return self.ready


class _NoopStreamModule:
    """A device-module stub so pull_confidence_history's copy branch runs on CPU
    without a real CUDA stream."""

    @staticmethod
    def stream(_stream):
        return contextlib.nullcontext()


def _flat_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1], sample_steps_per_sec=[1.0], max_batch_tokens=4096
    )


def _make_relay_worker(
    *,
    gamma: int,
    ready: bool,
    buf: torch.Tensor,
    pinned: torch.Tensor,
) -> DSparkWorkerV2:
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.gamma = gamma
    worker.device = torch.device("cpu")
    worker._confidence_buf = buf
    worker._confidence_cpu_pinned = pinned
    worker._confidence_ready = _FakeEvent(ready=ready)
    worker._confidence_d2h_stream = object()
    return worker


class TestPullConfidenceHistoryGating(CustomTestCase):
    """The confidence relay is non-blocking: the host reuses a stale CPU snapshot
    whenever the forward-stream write has not yet completed (event.query() is
    False), and only refreshes it once the write is visible. This is the source
    of the emergent (>= 1) GPU-completion lag -- there is no structured n-2 ring,
    just one buffer behind one event.
    """

    def test_not_ready_returns_stale_snapshot_without_copy(self):
        """query()=False -> pull returns the existing pinned snapshot, no D2H copy."""
        gamma = 3
        buf = torch.full((4, gamma), 0.9, dtype=torch.float32)
        stale = torch.full((4, gamma), 0.1, dtype=torch.float32)
        worker = _make_relay_worker(gamma=gamma, ready=False, buf=buf, pinned=stale)

        result = worker.pull_confidence_history()

        self.assertIs(result, stale)
        # The fresh device buffer (0.9) must NOT have been copied into the snapshot.
        self.assertTrue(torch.equal(result, torch.full((4, gamma), 0.1)))

    def test_ready_refreshes_snapshot_from_device_buffer(self):
        """query()=True -> pull copies the device buffer into the pinned snapshot."""
        gamma = 3
        buf = torch.full((4, gamma), 0.9, dtype=torch.float32)
        stale = torch.full((4, gamma), 0.1, dtype=torch.float32)
        worker = _make_relay_worker(gamma=gamma, ready=True, buf=buf, pinned=stale)

        orig_get_device_module = torch.get_device_module
        torch.get_device_module = lambda *_args, **_kwargs: _NoopStreamModule()
        try:
            result = worker.pull_confidence_history()
        finally:
            torch.get_device_module = orig_get_device_module

        self.assertTrue(torch.equal(result, torch.full((4, gamma), 0.9)))

    def test_returns_none_when_relay_uninitialized(self):
        """No event / no pinned buffer -> pull returns None (uniform fallback)."""
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.device = torch.device("cpu")
        worker._confidence_ready = None
        worker._confidence_cpu_pinned = None
        self.assertIsNone(worker.pull_confidence_history())


class TestStepBudgetUsesPriorSnapshot(CustomTestCase):
    """Step n's verify budget K must be derived from the lagged snapshot, never
    from step n's own confidence: while event.query() stays False, stashing
    step-n confidence cannot change the K computed from the prior snapshot.
    """

    def test_step_n_confidence_does_not_change_budget_while_event_pending(self):
        """Stashing step-n confidence with a pending event leaves K at step n-1's."""
        gamma = 4
        # Prior snapshot (step n-1): confidence decays so survival = cumprod falls
        # below survival_eps at the tail, dropping those candidates -> smaller K.
        # (0.005 -> cumprod [5e-3, 2.5e-5, 1.25e-7, ...]; positions >= 2 are gated.)
        prior = torch.full((2, gamma), 0.005, dtype=torch.float32)
        # Device buffer is overwritten with step-n high confidence (every position
        # survives -> larger K), but the event stays pending so the host must not
        # see it.
        buf = torch.full((2, gamma), 0.99, dtype=torch.float32)
        worker = _make_relay_worker(gamma=gamma, ready=False, buf=buf, pinned=prior)

        cfg = DSparkScheduleConfig(gamma=gamma)
        scheduler = ConfidencePrefixScheduler(sps_table=_flat_table(), cfg=cfg)

        snapshot = worker.pull_confidence_history()
        survival_prior = torch.cumprod(snapshot.to(torch.float32), dim=1)
        budget_from_prior = scheduler.update_budget_from_history(
            history_survival_probs=survival_prior
        )

        # What K *would* be if step-n confidence (the device buffer) leaked in.
        survival_now = torch.cumprod(buf.to(torch.float32), dim=1)
        budget_if_leaked = scheduler.update_budget_from_history(
            history_survival_probs=survival_now
        )

        self.assertNotEqual(
            budget_from_prior,
            budget_if_leaked,
            "test scenario is degenerate: prior and current would yield the same K",
        )
        # The pull returned the prior snapshot, so the scheduled K is the prior's.
        self.assertTrue(torch.equal(snapshot, prior))


class TestConfidenceRelayTopologyAntiPattern(CustomTestCase):
    """Anti-pattern guard mirroring test_dspark_invariants' source-string style:
    the relay must stay single-buffered (one _confidence_buf, one event), so the
    lag is emergent and >= 1, never a structured step-indexed n-2 double buffer.
    """

    def _relay_init_source(self) -> str:
        text = _WORKER.read_text()
        start = text.find("def _ensure_confidence_relay_buffers(self")
        self.assertNotEqual(start, -1, "_ensure_confidence_relay_buffers not found.")
        end = text.find("\n    def ", start + 1)
        return text[start : end if end != -1 else len(text)]

    def test_single_confidence_buffer_no_ring(self):
        """Exactly one _confidence_buf is allocated (no step-indexed ring)."""
        block = self._relay_init_source()
        assigns = re.findall(r"self\._confidence_buf\s*=\s*torch\.empty", block)
        self.assertEqual(
            len(assigns),
            1,
            "relay must allocate exactly one _confidence_buf (no double buffer).",
        )
        self.assertNotIn(
            "_confidence_buf_prev",
            _WORKER.read_text(),
            "no step n-2 / previous-step confidence buffer may exist.",
        )

    def test_single_confidence_ready_event(self):
        """Exactly one confidence-ready Event() backs the relay (no parity ring)."""
        block = self._relay_init_source()
        events = re.findall(
            r"self\._confidence_ready\s*=\s*device_module\.Event\(\)", block
        )
        self.assertEqual(
            len(events),
            1,
            "relay must use exactly one confidence-ready Event() (no ring).",
        )


if __name__ == "__main__":
    unittest.main()
