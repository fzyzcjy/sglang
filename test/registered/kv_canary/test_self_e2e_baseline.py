from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class _BaselineBase(CanaryE2EBase):
    """No perturb, kv-canary=log, sweep off. Server should run clean with no canary
    violations and every request must come back 200."""

    __test__ = False

    kv_canary_mode = "log"
    extra_env = {}

    def test_no_violation(self) -> None:
        """Verify the baseline canary run completes without violations."""
        self.send_parallel_requests(n=4)
        self.assert_no_violation(wait_seconds=2.0)


class TestBaselineMha(_BaselineBase, unittest.TestCase):
    __test__ = True

    model_mode = "mha"


class TestBaselineSwa(_BaselineBase, unittest.TestCase):
    __test__ = True

    model_mode = "swa"


class TestBaselineHybridSwaRealDataSweep(CanaryE2EBase, unittest.TestCase):
    __test__ = True

    model_mode = "hybrid_swa"
    kv_canary_mode = "log"
    extra_env = {}
    extra_server_args = (
        "--kv-canary-real-data",
        "partial",
        "--kv-canary-sweep-interval",
        "4",
    )
    use_unique_prompts = True

    def test_no_violation_with_real_data_sweep(self) -> None:
        """Verify hybrid-SWA real-KV sweep has no ambient violations."""
        self.send_parallel_requests(n=8)
        self.send_parallel_requests(n=8)
        self.assert_no_violation(wait_seconds=5.0)


if __name__ == "__main__":
    unittest.main()
