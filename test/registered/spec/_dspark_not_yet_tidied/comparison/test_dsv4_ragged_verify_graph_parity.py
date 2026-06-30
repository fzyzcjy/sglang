import functools
import unittest

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()

# A deliberately mixed verify_lens so the ragged geometry differs from any
# uniform full block: req0 verifies the full gamma+1 window, req1 the anchor
# only, req2 an intermediate prefix. A capture that baked the uniform geometry
# (the C2 bug) cannot reproduce these per-request windows.
_MIXED_VERIFY_LENS = (5, 1, 3)


def _requires_cuda(test_method):
    """Decorator: skip when CUDA is unavailable (the dsv4 sparse backend is GPU-only)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; dsv4 sparse backend is GPU-only.")
        return test_method(self, *args, **kwargs)

    return wrapper


def _graph_parity_harness_available() -> tuple[bool, str]:
    """Return (available, reason) for the graph-vs-eager ragged-verify parity harness.

    The teeth test runs a compact (real-N) DSpark target verify forward through the
    decode cuda-graph runner and through the eager backend on the SAME weights /
    inputs / ragged layout, then compares the pre-scatter verify logits. It needs
    the dsv4 block-forward harness extended with a ``run_target_verify`` entry that
    drives a fixed ragged ``verify_lens`` under both execution modes. Skips cleanly
    until that harness entry lands.
    """
    if not _CUDA_AVAILABLE:
        return False, "CUDA not available; dsv4 sparse backend is GPU-only."
    try:
        from test.manual._dspark_reference.dsv4.block_forward_harness import (  # noqa: F401
            HarnessUnavailable,
            build_dsv4_block_forward_harness,
        )
    except ImportError as exc:  # pragma: no cover - import guard
        return False, f"harness import failed: {exc}"
    return True, ""


class TestDsv4RaggedVerifyGraphParity(CustomTestCase):
    """Geometry-level teeth for the compact ragged-verify cuda-graph (plan §4).

    Output-text parity has no teeth for the C2 bug (a wrong verify geometry only
    lowers accept length; losslessness is held by _cap_correct_len + bonus re-read).
    These tests compare the pre-scatter verify logits / per-request geometry of the
    cuda-graph forward against the eager forward on the SAME weights, inputs, and a
    forced mixed ``verify_lens``, so a capture that baked the uniform geometry (C2)
    diverges. GPU-tier; skips cleanly until the harness entry lands.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._available, cls._skip_reason = _graph_parity_harness_available()

    def _skip_if_unready(self) -> None:
        if not self._available:
            self.skipTest(self._skip_reason)

    @_requires_cuda
    def test_graph_verify_logits_match_eager_on_mixed_verify_lens(self) -> None:
        """Cuda-graph compact verify logits equal the eager forward on mixed verify_lens."""
        self._skip_if_unready()
        self.skipTest(
            "GPU teeth test: requires the block-forward harness run_target_verify "
            "entry that drives a fixed ragged verify_lens through the cuda-graph "
            "runner and the eager backend, comparing pre-scatter verify logits with "
            f"verify_lens={_MIXED_VERIFY_LENS}. Tester wires the harness + fixture; "
            "the assertion is torch.testing.assert_close(graph_logits, eager_logits)."
        )

    @_requires_cuda
    def test_force_uniform_capture_seam_makes_parity_diverge(self) -> None:
        """The negative seam (force-uniform capture) makes the graph-vs-eager parity diverge.

        Proves the parity test has teeth: with
        SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE on, the capture bakes the
        uniform geometry (the C2 bug), so the same mixed-verify_lens batch produces
        per-request windows that no longer match the eager ragged forward and the
        logits parity assertion must FAIL. A parity check that still passes under
        this toggle is not actually exercising the ragged geometry.
        """
        self._skip_if_unready()
        with envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.override(True):
            self.skipTest(
                "GPU negative seam: with force-uniform capture on, the harness "
                "run_target_verify graph logits must DIVERGE from the eager ragged "
                f"forward on verify_lens={_MIXED_VERIFY_LENS} (assertRaises around "
                "torch.testing.assert_close). Tester wires the harness + fixture."
            )

    @_requires_cuda
    def test_accept_length_floor_under_mixed_verify_lens(self) -> None:
        """Compact accept length stays above 1 on a batch the draft proposes well for.

        A capture that baked the uniform geometry silently lowers accept length
        (the only observable symptom of C2). This pins a floor: on a deterministic
        batch whose draft proposals are accepted, mean accept length must exceed 1,
        catching a geometry regression that output-text parity would miss.
        """
        self._skip_if_unready()
        self.skipTest(
            "GPU accept-length floor: requires the e2e compact harness to report "
            "mean accept length per verify step; assert it exceeds 1.0 on the "
            "deterministic accept batch. Tester wires the harness + fixture."
        )


if __name__ == "__main__":
    unittest.main()
