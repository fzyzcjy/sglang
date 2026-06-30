import types
import unittest

import torch

from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GAMMA = 3
_HIDDEN = 8


def _make_worker() -> DSparkWorkerV2:
    """A worker stub holding only the fields the smoke-tested helpers read.

    The dsv4 draft now flows through the SAME unified path as dense
    (``_run_draft_block_forward`` + ``_sample_draft_block``); the old capability
    flags (``_draft_owns_block_forward`` / ``_draft_owns_kv_injection`` /
    ``_draft_owns_confidence``), the ``_propose_draft_block_via_model`` route, the
    ``inject_target_hidden`` delegation, and the dense-vs-model confidence relay were
    deleted in the worker unification, so this stub no longer sets them.
    """
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.gamma = _GAMMA
    worker.verify_num_draft_tokens = _GAMMA + 1
    worker.device = _DEVICE
    worker._verify_backend_self_adds_seq_lens_cache = None
    return worker


class TestDsv4VerifyBackendSelfAdd(CustomTestCase):
    """The verify-backend self-add capability is resolved from the backend, not a flag."""

    def _worker_with_backend(self, backend) -> DSparkWorkerV2:
        worker = _make_worker()
        worker._target_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(attn_backend=backend)
        )
        return worker

    def test_v4_family_verify_backend_self_adds(self) -> None:
        """A backend owning the raw target-verify metadata builder self-adds seq lens."""
        backend = types.SimpleNamespace(
            make_forward_metadata_from_raw_verify=lambda *a, **k: None
        )
        worker = self._worker_with_backend(backend)
        self.assertTrue(worker._verify_backend_self_adds_seq_lens())

    def test_dense_verify_backend_does_not_self_add(self) -> None:
        """A dense backend lacks the raw verify builder, so the worker keeps the pre-add."""
        backend = types.SimpleNamespace()
        worker = self._worker_with_backend(backend)
        self.assertFalse(worker._verify_backend_self_adds_seq_lens())

    def test_self_add_capability_is_cached(self) -> None:
        """The resolved self-add capability is cached after first resolution."""
        backend = types.SimpleNamespace(
            make_forward_metadata_from_raw_verify=lambda *a, **k: None
        )
        worker = self._worker_with_backend(backend)
        worker._verify_backend_self_adds_seq_lens()
        worker._target_worker.model_runner.attn_backend = types.SimpleNamespace()
        self.assertTrue(worker._verify_backend_self_adds_seq_lens())


class TestDsv4HiddenGeometry(CustomTestCase):
    """Per-request committed / prefill-anchor hidden gather geometry (unified path)."""

    def test_select_committed_hidden_picks_bonus_position(self) -> None:
        """The committed hidden per request is gathered at the correct_len index."""
        worker = _make_worker()
        bs = 2
        window = _GAMMA + 1
        hidden = torch.arange(bs * window * _HIDDEN, dtype=torch.float32).view(
            bs * window, _HIDDEN
        )
        correct_len = torch.tensor([1, 3], dtype=torch.int32)
        committed = worker._select_committed_hidden(
            hidden=hidden, correct_len=correct_len, bs=bs
        )
        hidden_2d = hidden.view(bs, window, _HIDDEN)
        self.assertTrue(torch.equal(committed[0], hidden_2d[0, 1]))
        self.assertTrue(torch.equal(committed[1], hidden_2d[1, 3]))

    def test_select_prefill_last_hidden_picks_extend_boundaries(self) -> None:
        """The first-decode anchor hidden is gathered at each request's extend boundary."""
        worker = _make_worker()
        total = 7
        hidden = torch.arange(total * _HIDDEN, dtype=torch.float32).view(total, _HIDDEN)
        last = worker._select_prefill_last_hidden(hidden=hidden, extend_lens=[3, 4])
        self.assertTrue(torch.equal(last[0], hidden[2]))
        self.assertTrue(torch.equal(last[1], hidden[6]))


if __name__ == "__main__":
    unittest.main()
