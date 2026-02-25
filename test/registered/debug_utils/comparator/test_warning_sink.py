import sys

import pytest

from sglang.srt.debug_utils.comparator.output_types import (
    AnyWarning,
    ReplicatedMismatchWarning,
)
from sglang.srt.debug_utils.comparator.warning_sink import WarningSink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _make_warning(**overrides) -> ReplicatedMismatchWarning:
    defaults: dict = dict(
        axis="tp",
        group_index=0,
        differing_index=1,
        baseline_index=0,
        max_abs_diff=0.1,
    )
    defaults.update(overrides)
    return ReplicatedMismatchWarning(**defaults)


class TestWarningSink:
    def test_basic_collection(self) -> None:
        sink = WarningSink()
        warning = _make_warning()

        with sink.context() as collected:
            sink.add(warning)

        assert len(collected) == 1
        assert collected[0] is warning

    def test_nested_contexts(self) -> None:
        sink = WarningSink()
        outer_warning = _make_warning(group_index=0)
        inner_warning = _make_warning(group_index=1)

        with sink.context() as outer:
            sink.add(outer_warning)
            with sink.context() as inner:
                sink.add(inner_warning)
            assert len(inner) == 1
            assert inner[0] is inner_warning

        assert len(outer) == 1
        assert outer[0] is outer_warning

    def test_add_outside_context_no_fallback_raises(self) -> None:
        sink = WarningSink()
        with pytest.raises(RuntimeError, match="outside a warning_sink.context"):
            sink.add(_make_warning())

    def test_empty_context(self) -> None:
        sink = WarningSink()
        with sink.context() as collected:
            pass
        assert collected == []

    def test_fallback_called_outside_context(self) -> None:
        sink = WarningSink()
        received: list[AnyWarning] = []
        sink.set_fallback(lambda w: received.append(w))

        warning = _make_warning()
        sink.add(warning)

        assert len(received) == 1
        assert received[0] is warning

    def test_fallback_not_called_inside_context(self) -> None:
        sink = WarningSink()
        fallback_called: list[bool] = []
        sink.set_fallback(lambda w: fallback_called.append(True))

        warning = _make_warning()
        with sink.context() as collected:
            sink.add(warning)

        assert len(collected) == 1
        assert fallback_called == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
