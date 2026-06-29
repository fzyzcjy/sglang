import unittest

from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestReadRaggedVerifyMode(CustomTestCase):
    def test_default_unset_is_off(self):
        """An unset SGLANG_RAGGED_VERIFY reads as OFF."""
        was_set = envs.SGLANG_RAGGED_VERIFY.is_set()
        backup = envs.SGLANG_RAGGED_VERIFY.get() if was_set else None
        envs.SGLANG_RAGGED_VERIFY.clear()
        try:
            self.assertEqual(read_ragged_verify_mode(), RaggedVerifyMode.OFF)
        finally:
            if was_set:
                envs.SGLANG_RAGGED_VERIFY.set(backup)

    def test_off_literal_is_off(self):
        """The literal 'off' reads as OFF."""
        with envs.SGLANG_RAGGED_VERIFY.override("off"):
            self.assertEqual(read_ragged_verify_mode(), RaggedVerifyMode.OFF)

    def test_empty_string_is_off(self):
        """An explicit empty string reads as OFF (matches the env-layer unset alias)."""
        with envs.SGLANG_RAGGED_VERIFY.override(""):
            self.assertEqual(read_ragged_verify_mode(), RaggedVerifyMode.OFF)

    def test_cutoff_only(self):
        """'cutoff-only' reads as CUTOFF_ONLY."""
        with envs.SGLANG_RAGGED_VERIFY.override("cutoff-only"):
            self.assertEqual(read_ragged_verify_mode(), RaggedVerifyMode.CUTOFF_ONLY)

    def test_full(self):
        """'full' reads as FULL."""
        with envs.SGLANG_RAGGED_VERIFY.override("full"):
            self.assertEqual(read_ragged_verify_mode(), RaggedVerifyMode.FULL)

    def test_invalid_value_raises_loudly(self):
        """An unrecognized value raises rather than silently falling back to default."""
        with envs.SGLANG_RAGGED_VERIFY.override("cutoff"):
            with self.assertRaises(ValueError):
                read_ragged_verify_mode()


if __name__ == "__main__":
    unittest.main()
