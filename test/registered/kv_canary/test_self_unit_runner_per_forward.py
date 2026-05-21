from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from kv_canary_runner_unit_utils import (
    CanaryRunnerTestCase,
    RecordingEndpoint,
    make_config,
    make_forward_batch,
    make_group,
    make_runner,
)

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.runner import kernel_launch as kernel_launch_module
from sglang.srt.kv_canary.runner import per_forward as per_forward_module
from sglang.srt.kv_canary.runner import violation_manager as violation_manager_module
from sglang.srt.kv_canary.runner.enable_warner import _CanaryEnableWarner
from sglang.srt.kv_canary.state import ViolationLog
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestRunnerPerForward(CanaryRunnerTestCase):
    def test_per_forward_orchestrates_plan_head_tail(self) -> None:
        """Verify per-forward execution launches plan, head, and tail kernels."""
        calls: list[object] = []
        with patch.object(
            kernel_launch_module,
            "launch_canary_plan_kernels",
            lambda **kwargs: calls.append("plan"),
        ), patch.object(
            endpoint_module,
            "launch_canary_verify_kernel",
            lambda **kwargs: calls.append(
                ("verify", kwargs["context"].kernel_kind.name)
            ),
        ), patch.object(
            endpoint_module,
            "launch_canary_write_kernel",
            lambda **kwargs: calls.append(
                ("write", kwargs["context"].kernel_kind.name)
            ),
        ):
            runner = make_runner(device=self.device)
            forward_batch = make_forward_batch(self.device)
            with runner.with_forward_pass(forward_batch):
                runner.launch_head_kernels(forward_batch)
                runner.launch_tail_kernels(forward_batch)

        self.assertEqual(calls[0], "plan")
        self.assertTrue(
            any(
                call[0] == "verify" and "HEAD" in call[1]
                for call in calls[1:]
                if isinstance(call, tuple)
            )
        )
        self.assertTrue(
            any(
                call[0] == "verify" and "TAIL" in call[1]
                for call in calls[1:]
                if isinstance(call, tuple)
            )
        )


class TestLaunchEndpointsPerForward(CanaryRunnerTestCase):
    def test_launch_endpoints_per_forward_keeps_padded_token_tensors(self) -> None:
        """Verify endpoint launch preserves CUDA graph-stable tensor shapes."""
        group = make_group(device=self.device)
        endpoint = RecordingEndpoint(kernel_kind=CanaryLaunchTag.HEAD_K_FULL)
        forward_batch = make_forward_batch(self.device, bs=1, seq_lens_list=(1,))
        forward_batch.input_ids = torch.tensor(
            [101, 0, 0], dtype=torch.int64, device=self.device
        )
        forward_batch.positions = torch.tensor(
            [10, 0, 0], dtype=torch.int64, device=self.device
        )
        forward_batch.out_cache_loc = torch.tensor(
            [7, 0, 0], dtype=torch.int64, device=self.device
        )
        forward_batch.num_token_non_padded_cpu = 1

        kernel_launch_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=3, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.OFF,
            input_check_mode=False,
        )

        self.assertEqual(len(endpoint.calls), 1)
        call = endpoint.calls[0]
        self.assertTrue(
            torch.equal(
                call["input_ids"],
                torch.tensor([101, 0, 0], dtype=torch.int64, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                call["positions"],
                torch.tensor([10, 0, 0], dtype=torch.int64, device=self.device),
            )
        )
        self.assertTrue(
            torch.equal(
                call["out_cache_loc"],
                torch.tensor([7, 0, 0], dtype=torch.int64, device=self.device),
            )
        )

    def test_launch_endpoints_per_forward_accepts_int32_boundary_tensors(self) -> None:
        """Verify int32 ForwardBatch tensors are promoted at the canary boundary."""
        group = make_group(device=self.device)
        endpoint = RecordingEndpoint(kernel_kind=CanaryLaunchTag.HEAD_K_FULL)
        forward_batch = make_forward_batch(self.device, bs=1, seq_lens_list=(1,))
        forward_batch.input_ids = torch.tensor(
            [101], dtype=torch.int32, device=self.device
        )
        forward_batch.positions = torch.tensor(
            [10], dtype=torch.int32, device=self.device
        )
        forward_batch.out_cache_loc = torch.tensor(
            [7], dtype=torch.int32, device=self.device
        )

        kernel_launch_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=1, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.OFF,
            input_check_mode=False,
        )

        call = endpoint.calls[0]
        self.assertEqual(call["input_ids"].dtype, torch.int64)
        self.assertEqual(call["positions"].dtype, torch.int64)
        self.assertEqual(call["out_cache_loc"].dtype, torch.int64)

    def test_launch_endpoints_per_forward_accepts_strided_boundary_tensors(
        self,
    ) -> None:
        """Verify EAGLE-style strided ForwardBatch views are copied at the canary boundary."""
        group = make_group(device=self.device)
        endpoint = RecordingEndpoint(kernel_kind=CanaryLaunchTag.HEAD_K_FULL)
        forward_batch = make_forward_batch(self.device, bs=1, seq_lens_list=(1,))
        forward_batch.input_ids = torch.tensor(
            [[101, 102]], dtype=torch.int64, device=self.device
        )[:, 0]
        forward_batch.positions = torch.tensor(
            [[10, 11]], dtype=torch.int64, device=self.device
        )[:, 0]
        forward_batch.out_cache_loc = torch.tensor(
            [[7, 8]], dtype=torch.int64, device=self.device
        )[:, 0]

        kernel_launch_module.launch_endpoints_per_forward(
            endpoints=(endpoint,),
            group=group,
            tag_filter=lambda tag: True,
            verify_plan=VerifyPlan.allocate(verify_capacity=1, device=self.device),
            write_plan=WritePlan.allocate(write_req_capacity=1, device=self.device),
            forward_batch=forward_batch,
            expected_inputs=ExpectedInputs.allocate(capacity=1, device=self.device),
            violation_log=ViolationLog.allocate(ring_capacity=2, device=self.device),
            real_kv_hash_mode=RealKvHashMode.OFF,
            input_check_mode=False,
        )

        call = endpoint.calls[0]
        self.assertTrue(call["input_ids"].is_contiguous())
        self.assertTrue(call["positions"].is_contiguous())
        self.assertTrue(call["out_cache_loc"].is_contiguous())


class TestRunnerBeforeForward(CanaryRunnerTestCase):
    def test_input_check_is_disabled_while_cuda_graph_captures(self) -> None:
        """Verify graph capture cannot bake request-oracle input checks into replay."""
        config = make_config(input_check_mode=True)
        runner = make_runner(device=self.device, config=config)
        forward_batch = make_forward_batch(self.device)
        forward_batch.forward_mode = _FakeExtendForwardMode()
        orchestrator = runner._per_forward_orchestrator

        self.assertTrue(
            orchestrator._should_enable_input_check_for_launch(forward_batch)
        )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertFalse(
                orchestrator._should_enable_input_check_for_launch(forward_batch)
            )

    def test_input_check_is_disabled_for_draft_extend(self) -> None:
        """Verify generated draft-extend writes do not use request-oracle input checks."""
        mode = SimpleNamespace(
            is_decode=lambda: False,
            is_draft_extend=lambda include_v2=False: include_v2,
        )
        config = make_config(input_check_mode=True)
        runner = make_runner(device=self.device, config=config)
        forward_batch = make_forward_batch(self.device)
        forward_batch.forward_mode = mode

        self.assertFalse(
            runner._per_forward_orchestrator._should_enable_input_check_for_launch(
                forward_batch
            )
        )

    def test_speculative_decode_expected_positions_follow_cache_slots(self) -> None:
        """Verify EAGLE draft decode checks positions against the written cache slot."""
        req_to_token = torch.zeros(3, 128, dtype=torch.int32, device=self.device)
        req_to_token[1, 64] = 1001
        req_to_token[1, 65] = 1002
        req_to_token[2, 64] = 2001
        req_to_token[2, 65] = 2002
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor(
                [1, 2], dtype=torch.int64, device=self.device
            ),
            spec_info=SimpleNamespace(num_tokens_per_req=2),
            positions=torch.tensor(
                [65, 66, 65, 66], dtype=torch.int64, device=self.device
            ),
            out_cache_loc=torch.tensor(
                [1001, 1002, 2001, 2002], dtype=torch.int64, device=self.device
            ),
        )

        expected_positions = (
            per_forward_module._derive_positions_from_req_to_token_slots(
                forward_batch=forward_batch,
                req_to_token=req_to_token,
            )
        )

        self.assertTrue(
            torch.equal(
                expected_positions,
                torch.tensor([64, 65, 64, 65], dtype=torch.int64, device=self.device),
            )
        )

    def test_enable_warner_skips_host_sync_during_cuda_graph_capture(self) -> None:
        """Verify the enable warner does not synchronize events while CUDA graph capture is active."""
        warner = _CanaryEnableWarner(
            verify_capacity=1,
            d2h_stream=torch.cuda.current_stream(self.device),
        )
        enable = torch.ones(1, dtype=torch.int32, device=self.device)

        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            warner.tick(enable)

        self.assertIsNone(warner._pending_future)

    def test_end_of_step_skips_host_work_during_cuda_graph_capture(self) -> None:
        """Verify graph capture avoids host-side sweep, violation, health, and stats work."""
        runner = make_runner(device=self.device)
        calls: list[str] = []

        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with patch.object(
                runner._per_forward_orchestrator,
                "end_of_step",
                lambda: calls.append("per_forward"),
            ), patch.object(
                runner._sweep_orchestrator,
                "maybe_run_sweep",
                lambda: calls.append("sweep"),
            ), patch.object(
                runner._violation_manager,
                "step",
                lambda: calls.append("violation"),
            ), patch.object(
                runner._health_checker,
                "step",
                lambda: calls.append("health"),
            ), patch.object(
                runner._stats_logger,
                "step",
                lambda: calls.append("stats"),
            ):
                runner._end_of_step()

        self.assertEqual(calls, ["per_forward"])

    def test_violation_signal_written_during_capture_is_reported_after_replay(
        self,
    ) -> None:
        """Verify graph-captured violations are observed after replay without host sync in capture."""
        runner = make_runner(device=self.device)
        reported_steps: list[int] = []

        class _ReadyFuture:
            def __init__(self, value: int) -> None:
                self._value = value

            def wait(self) -> torch.Tensor:
                return torch.tensor([self._value], dtype=torch.uint8)

        def _record_signal(
            *, src_device: torch.Tensor, stream: torch.cuda.Stream
        ) -> _ReadyFuture:
            del stream
            return _ReadyFuture(int(src_device.detach().cpu().item()))

        runner._device_state.violation_log.violation_write_index.fill_(1)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            runner._end_of_step()

        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            with patch.object(
                violation_manager_module.FutureTensor,
                "device_to_host",
                _record_signal,
            ), patch.object(
                runner._violation_manager._violation_reporter,
                "log_or_raise_violation",
                lambda step_counter: reported_steps.append(step_counter),
            ):
                runner._end_of_step()
                runner._end_of_step()

        self.assertEqual(reported_steps, [runner._step_counter])

    def test_before_forward_does_not_throw_on_oversized_prefix_sum(self) -> None:
        """Verify oversized prefix sums are handled without host-side errors."""
        # Overflow no longer raises host-side: the plan kernel sets VerifyPlan.enable=0 and the
        # verify kernel skips the step on-device; host logs a throttled warning instead.
        runner = make_runner(device=self.device, per_forward_verify_capacity=4)
        forward_batch = make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        with runner.with_forward_pass(forward_batch):
            pass

    def test_before_forward_passes_when_sum_prefix_lens_fits(self) -> None:
        """Verify prefix sums within capacity pass before-forward handling."""
        # Same multi-req shape that breaks the old sizing now fits the new capacity formula.
        runner = make_runner(device=self.device, per_forward_verify_capacity=16)
        forward_batch = make_forward_batch(self.device, bs=2, seq_lens_list=(5, 5))
        with runner.with_forward_pass(forward_batch):
            pass


class _FakeExtendForwardMode:
    def is_decode(self) -> bool:
        return False

    def is_draft_extend(self, include_v2: bool = False) -> bool:
        return False


if __name__ == "__main__":
    unittest.main()
