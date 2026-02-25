from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Generator, Optional

from sglang.srt.debug_utils.comparator.output_types import AnyWarning


class WarningSink:
    def __init__(self) -> None:
        self._stack: list[list[AnyWarning]] = []
        self._fallback: Optional[Callable[[AnyWarning], None]] = None

    def set_fallback(self, fallback: Optional[Callable[[AnyWarning], None]]) -> None:
        self._fallback = fallback

    @contextmanager
    def context(self) -> Generator[list[AnyWarning], None, None]:
        bucket: list[AnyWarning] = []
        self._stack.append(bucket)
        try:
            yield bucket
        finally:
            popped = self._stack.pop()
            assert popped is bucket

    def add(self, warning: AnyWarning) -> None:
        if self._stack:
            self._stack[-1].append(warning)
        elif self._fallback is not None:
            self._fallback(warning)
        else:
            raise RuntimeError(
                "warning_sink.add() called outside a warning_sink.context() "
                "and no fallback is set"
            )


warning_sink = WarningSink()
