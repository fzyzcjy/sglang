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
    def test_default_unset_is_static(self):
        """An unset SGLANG_RAGGED_VERIFY reads as STATIC."""
        was_set = envs.SGLANG_RAGGED_VERIFY.is_set()
        backup = envs.SGLANG_RAGGED_VERIFY.get() if was_set else None
        envs.SGLANG_RAGGED_VERIFY.clear()
        try:
            self.assertEqual(read_ragged_verify_mode(), RaggedVerifyMode.STATIC)
        finally:
            if was_set:
                envs.SGLANG_RAGGED_VERIFY.set(backup)

    def test_canonical_values(self):
        """The canonical values static / cap-accept / compact round-trip exactly."""
        cases = {
            "static": RaggedVerifyMode.STATIC,
            "cap-accept": RaggedVerifyMode.CAP_ACCEPT,
            "compact": RaggedVerifyMode.COMPACT,
        }
        for value, expected in cases.items():
            with envs.SGLANG_RAGGED_VERIFY.override(value):
                self.assertEqual(read_ragged_verify_mode(), expected)

    def test_legacy_aliases(self):
        """Legacy spellings (off / cutoff-only / full) resolve to the renamed modes."""
        cases = {
            "off": RaggedVerifyMode.STATIC,
            "": RaggedVerifyMode.STATIC,
            "cutoff-only": RaggedVerifyMode.CAP_ACCEPT,
            "full": RaggedVerifyMode.COMPACT,
        }
        for value, expected in cases.items():
            with envs.SGLANG_RAGGED_VERIFY.override(value):
                self.assertEqual(read_ragged_verify_mode(), expected)

    def test_invalid_value_raises_loudly(self):
        """An unrecognized value raises rather than silently falling back to default."""
        with envs.SGLANG_RAGGED_VERIFY.override("cutoff"):
            with self.assertRaises(ValueError):
                read_ragged_verify_mode()


if __name__ == "__main__":
    unittest.main()
