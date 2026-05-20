"""Regression for PR #25015 EAGLE positions misalign."""

from __future__ import annotations

import inspect
import unittest

from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")


class TestEaglePositionsMatchWithFix(CustomTestCase):
    """PR #25015 keeps draft positions aligned across runtime and CUDA graph capture."""

    def test_eagle_v1_advances_positions_after_draft_forward(self) -> None:
        _assert_positions_advanced_after_forward(
            function=EAGLEWorker.draft_forward,
            forward_call="self.draft_model_runner.forward",
        )

    def test_eagle_v2_advances_positions_after_draft_forward(self) -> None:
        _assert_positions_advanced_after_forward(
            function=EagleDraftWorker.draft_forward,
            forward_call="self.draft_runner.forward",
        )

    def test_eagle_cuda_graph_capture_restores_positions(self) -> None:
        source = inspect.getsource(EAGLEDraftCudaGraphRunner.capture_one_batch_size)
        restore_index = source.index(
            "forward_batch.spec_info.hidden_states = hidden_states_backup"
        )
        compensate_index = source.index(
            "forward_batch.positions.sub_(self.eagle_worker.speculative_num_steps - 1)"
        )
        return_index = source.index("return ret", compensate_index)

        self.assertLess(restore_index, compensate_index)
        self.assertLess(compensate_index, return_index)


def _assert_positions_advanced_after_forward(
    *,
    function: object,
    forward_call: str,
) -> None:
    source = inspect.getsource(function)
    loop_index = source.index("for i in range(self.speculative_num_steps):")
    forward_index = source.index(forward_call, loop_index)
    advance_index = source.index("forward_batch.positions.add_(1)", forward_index)
    hidden_states_index = source.index("hidden_states = logits_output.hidden_states")

    assert hidden_states_index < advance_index


if __name__ == "__main__":
    unittest.main()
