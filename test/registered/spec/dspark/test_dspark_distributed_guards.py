import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import _handle_dspark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSparkDistributedGuards(CustomTestCase):
    def test_dp_attention_is_rejected(self):
        """DSpark + DP attention must fail-fast (c v1 is TP-only)."""
        server_args = SimpleNamespace(enable_dp_attention=True, pp_size=1)
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_pp_size_greater_than_one_is_rejected(self):
        """DSpark + pp_size>1 must fail-fast (c v1 supports pp_size==1 only)."""
        server_args = SimpleNamespace(enable_dp_attention=False, pp_size=2)
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)


if __name__ == "__main__":
    unittest.main()
