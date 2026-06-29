import types
import unittest


from sglang.srt.layers.attention.dsa import utils as dsa_utils
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_forward_batch(
    *, forward_mode: ForwardMode, with_cp_metadata: bool
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        forward_mode=forward_mode,
        attn_cp_metadata=object() if with_cp_metadata else None,
    )


def _guard_fires(*, capture_dspark: bool, forward_batch) -> bool:
    """Reproduce the M2 guard predicate from DeepseekV4Model.forward verbatim."""
    return bool(capture_dspark and dsa_utils.dsa_use_prefill_cp(forward_batch))


class TestDsv4AuxCaptureCpGuard(CustomTestCase):
    def setUp(self) -> None:
        self._orig = dsa_utils.is_dsa_enable_prefill_cp

    def tearDown(self) -> None:
        dsa_utils.is_dsa_enable_prefill_cp = self._orig

    def _set_prefill_cp_enabled(self, value: bool) -> None:
        dsa_utils.is_dsa_enable_prefill_cp = lambda: value

    def test_guard_fires_only_with_capture_and_active_prefill_cp(self) -> None:
        """M2 guard triggers exactly when DSpark aux capture and active prefill CP coincide."""
        self._set_prefill_cp_enabled(True)
        cp_extend = _make_forward_batch(
            forward_mode=ForwardMode.EXTEND, with_cp_metadata=True
        )
        self.assertTrue(_guard_fires(capture_dspark=True, forward_batch=cp_extend))

    def test_guard_silent_when_capture_disabled(self) -> None:
        """Normal V4 prefill CP without DSpark capture never trips the guard."""
        self._set_prefill_cp_enabled(True)
        cp_extend = _make_forward_batch(
            forward_mode=ForwardMode.EXTEND, with_cp_metadata=True
        )
        self.assertFalse(_guard_fires(capture_dspark=False, forward_batch=cp_extend))

    def test_guard_silent_when_prefill_cp_disabled(self) -> None:
        """DSpark capture without prefill CP (the c-v1 CP-off path) never trips the guard."""
        self._set_prefill_cp_enabled(False)
        cp_extend = _make_forward_batch(
            forward_mode=ForwardMode.EXTEND, with_cp_metadata=True
        )
        self.assertFalse(_guard_fires(capture_dspark=True, forward_batch=cp_extend))

    def test_guard_silent_without_cp_metadata(self) -> None:
        """DSpark capture during a non-CP extend (no attn_cp_metadata) is allowed."""
        self._set_prefill_cp_enabled(True)
        plain_extend = _make_forward_batch(
            forward_mode=ForwardMode.EXTEND, with_cp_metadata=False
        )
        self.assertFalse(_guard_fires(capture_dspark=True, forward_batch=plain_extend))

    def test_guard_silent_for_decode(self) -> None:
        """DSpark capture during decode (not a CP extend) is allowed."""
        self._set_prefill_cp_enabled(True)
        decode = _make_forward_batch(
            forward_mode=ForwardMode.DECODE, with_cp_metadata=True
        )
        self.assertFalse(_guard_fires(capture_dspark=True, forward_batch=decode))


if __name__ == "__main__":
    unittest.main()
