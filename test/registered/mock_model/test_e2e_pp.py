from __future__ import annotations

import os
import unittest

from utils import mock_model_engine_kwargs, shutdown_engine_ignoring_zombie_reap

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


class TestE2EPipelineParallel(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        engine_kwargs = mock_model_engine_kwargs(
            pp_size=2,
            sampling_backend="pytorch",
            cuda_graph_max_bs=8,
            max_running_requests=32,
            context_length=2048,
            max_total_tokens=16384,
            piecewise_cuda_graph_max_tokens=512,
            disable_overlap_schedule=True,
        )
        os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = "0"
        cls.engine = sgl.Engine(
            model_path="Qwen/Qwen3-0.6B",
            **engine_kwargs,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutdown_engine_ignoring_zombie_reap(cls.engine)
        os.environ.pop("SGLANG_KV_CANARY_INPUT_CHECK", None)

    def test_pp_no_canary_violation(self) -> None:
        self.engine.generate(
            input_ids=_fake_prompt(32),
            sampling_params={"max_new_tokens": 4, "temperature": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
