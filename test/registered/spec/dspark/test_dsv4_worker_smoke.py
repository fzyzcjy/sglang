import types
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GAMMA = 5


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


if __name__ == "__main__":
    unittest.main()
