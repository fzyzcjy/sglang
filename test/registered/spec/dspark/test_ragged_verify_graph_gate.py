import types
import unittest

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _gate_runner(
    *,
    capture_num_tokens: list[int],
    num_tokens_per_bs: int,
    require_mlp_sync: bool = False,
    require_mlp_tp_gather: bool = False,
    disable_padding: bool = False,
) -> types.SimpleNamespace:
    """A runner stub carrying only the fields _can_run_ragged_verify_graph reads."""
    return types.SimpleNamespace(
        capture_num_tokens=capture_num_tokens,
        num_tokens_per_bs=num_tokens_per_bs,
        require_mlp_sync=require_mlp_sync,
        require_mlp_tp_gather=require_mlp_tp_gather,
        disable_padding=disable_padding,
        is_encoder_decoder=False,
        enable_two_batch_overlap=False,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def _gate_layout(total_verify_tokens: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(total_verify_tokens=total_verify_tokens)


def _gate_forward_batch(
    *,
    batch_size: int,
    can_run_dp_cuda_graph: bool = True,
) -> types.SimpleNamespace:
    spec_info = types.SimpleNamespace(capture_hidden_mode=None)
    return types.SimpleNamespace(
        batch_size=batch_size,
        encoder_lens=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        spec_info=spec_info,
        can_run_dp_cuda_graph=can_run_dp_cuda_graph,
    )


def _can_run(runner, forward_batch, layout) -> bool:
    return DecodeCudaGraphRunner._can_run_ragged_verify_graph(
        runner, forward_batch, layout
    )


class TestRaggedVerifyGraphAdmission(CustomTestCase):
    def test_admits_when_total_and_bs_floor_fit(self):
        """A small ragged total whose bs floor fits the largest tier is admitted."""
        runner = _gate_runner(capture_num_tokens=[4, 8, 16, 32], num_tokens_per_bs=4)
        fb = _gate_forward_batch(batch_size=2)
        self.assertTrue(_can_run(runner, fb, _gate_layout(5)))

    def test_rejects_when_raw_bs_exceeds_max_bs(self):
        """raw_bs above the largest captured bs is rejected (avoids _pad_to_bucket crash)."""
        # capture_num_tokens max = 32 -> max_bs = 8; bs=9 floors to 9*4=36 > 32.
        runner = _gate_runner(capture_num_tokens=[4, 8, 16, 32], num_tokens_per_bs=4)
        fb = _gate_forward_batch(batch_size=9)
        self.assertFalse(_can_run(runner, fb, _gate_layout(9)))

    def test_rejects_when_total_exceeds_largest_tier(self):
        """A real total above the largest captured tier is rejected."""
        runner = _gate_runner(capture_num_tokens=[4, 8, 16, 32], num_tokens_per_bs=4)
        fb = _gate_forward_batch(batch_size=1)
        self.assertFalse(_can_run(runner, fb, _gate_layout(40)))

    def test_rejects_when_dp_cuda_graph_unavailable(self):
        """require_mlp_sync + can_run_dp_cuda_graph False falls back to eager."""
        runner = _gate_runner(
            capture_num_tokens=[4, 8, 16, 32],
            num_tokens_per_bs=4,
            require_mlp_sync=True,
        )
        fb = _gate_forward_batch(batch_size=2, can_run_dp_cuda_graph=False)
        self.assertFalse(_can_run(runner, fb, _gate_layout(5)))

    def test_require_mlp_tp_gather_fails_loud(self):
        """require_mlp_tp_gather is not supported and must assert, not silently pass."""
        runner = _gate_runner(
            capture_num_tokens=[4, 8, 16, 32],
            num_tokens_per_bs=4,
            require_mlp_tp_gather=True,
        )
        fb = _gate_forward_batch(batch_size=2)
        with self.assertRaises(AssertionError):
            _can_run(runner, fb, _gate_layout(5))

    def test_disable_padding_fails_loud(self):
        """disable_cuda_graph_padding is incompatible with bs padding and must assert."""
        runner = _gate_runner(
            capture_num_tokens=[4, 8, 16, 32],
            num_tokens_per_bs=4,
            disable_padding=True,
        )
        fb = _gate_forward_batch(batch_size=2)
        with self.assertRaises(AssertionError):
            _can_run(runner, fb, _gate_layout(5))


if __name__ == "__main__":
    unittest.main()
