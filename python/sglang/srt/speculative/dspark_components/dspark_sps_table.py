from __future__ import annotations

import bisect
import logging
import statistics
import time
from collections import deque
from typing import Callable, Optional

import msgspec

logger = logging.getLogger(__name__)


class SpsCostTable(msgspec.Struct, frozen=True):
    sample_batch_tokens: list[int]
    sample_steps_per_sec: list[float]
    max_batch_tokens: int

    def __post_init__(self) -> None:
        if not self.sample_batch_tokens:
            raise ValueError("SpsCostTable requires at least one probe.")
        if self.sample_batch_tokens != sorted(set(self.sample_batch_tokens)):
            raise ValueError(
                "sample_batch_tokens must be strictly increasing (monotone-sorted "
                f"invariant), got {self.sample_batch_tokens}."
            )
        if len(self.sample_batch_tokens) != len(self.sample_steps_per_sec):
            raise ValueError(
                "sample_batch_tokens and sample_steps_per_sec must have equal length, "
                f"got {len(self.sample_batch_tokens)} vs {len(self.sample_steps_per_sec)}."
            )
        if self.max_batch_tokens < self.sample_batch_tokens[-1]:
            raise ValueError(
                "max_batch_tokens must be >= the largest probe, got "
                f"{self.max_batch_tokens} < {self.sample_batch_tokens[-1]}."
            )

    def lookup(self, batch_tokens: int) -> float:
        idx = bisect.bisect_right(self.sample_batch_tokens, batch_tokens) - 1
        idx = max(0, min(idx, len(self.sample_batch_tokens) - 1))
        return self.sample_steps_per_sec[idx]

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> SpsCostTable:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def profile_sps_table(
    *,
    probes: list[tuple[int, float]],
    max_batch_tokens: Optional[int] = None,
) -> SpsCostTable:
    if not probes:
        raise ValueError("profile_sps_table requires at least one probe.")

    sorted_probes = sorted(probes, key=lambda probe: probe[0])

    sample_batch_tokens: list[int] = []
    sample_steps_per_sec: list[float] = []
    for batch_tokens, steps_per_sec in sorted_probes:
        batch_tokens = int(batch_tokens)
        if batch_tokens < 1:
            raise ValueError(
                f"profile_sps_table requires batch_tokens >= 1, got {batch_tokens}."
            )
        if sample_batch_tokens and batch_tokens == sample_batch_tokens[-1]:
            raise ValueError(
                "profile_sps_table requires unique batch_tokens per probe; "
                f"batch_tokens={batch_tokens} appears more than once. Median the "
                "repeated samples per batch_tokens before calling the assembler."
            )
        sample_batch_tokens.append(batch_tokens)
        sample_steps_per_sec.append(float(steps_per_sec))

    resolved_max = (
        int(max_batch_tokens)
        if max_batch_tokens is not None
        else sample_batch_tokens[-1]
    )
    return SpsCostTable(
        sample_batch_tokens=sample_batch_tokens,
        sample_steps_per_sec=sample_steps_per_sec,
        max_batch_tokens=resolved_max,
    )


def load_sps_table_from_path(path: str) -> SpsCostTable:
    with open(path, "r", encoding="utf-8") as f:
        return SpsCostTable.from_json(f.read())


def build_uninitialized_sps_table(*, max_batch_tokens: int) -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[1.0],
        max_batch_tokens=max_batch_tokens,
    )


def is_uninitialized_sps_table(table: SpsCostTable) -> bool:
    return len(table.sample_batch_tokens) <= 1


def build_batch_size_sweep(max_num_tokens: int) -> list[int]:
    if max_num_tokens < 1:
        raise ValueError(f"max_num_tokens must be >= 1, got {max_num_tokens}.")
    raw = [
        1,
        2,
        4,
        8,
        *range(12, 128, 4),
        *range(128, 256, 16),
        *range(256, 1024, 32),
        *range(1024, 2048, 128),
        *range(2048, max_num_tokens + 1, 256),
    ]
    sweep = sorted({value for value in raw if 1 <= value <= max_num_tokens})
    if sweep[-1] != max_num_tokens:
        sweep.append(max_num_tokens)
    return sweep


ONLINE_SAMPLE_WINDOW = 128
ONLINE_MAX_STEP_INTERVAL_SECONDS = 1.0


class OnlineSpsProfiler:

    def __init__(
        self,
        *,
        initial_table: SpsCostTable,
        rebuild_interval_steps: int,
        min_bin_samples: int,
        sample_window: int = ONLINE_SAMPLE_WINDOW,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rebuild_interval_steps < 1:
            raise ValueError(
                f"rebuild_interval_steps must be >= 1, got {rebuild_interval_steps}."
            )
        if min_bin_samples < 1:
            raise ValueError(f"min_bin_samples must be >= 1, got {min_bin_samples}.")
        self._initial_table = initial_table
        self._initial_is_profiled = not is_uninitialized_sps_table(initial_table)
        if self._initial_is_profiled:
            self._bin_edges = list(initial_table.sample_batch_tokens)
        else:
            self._bin_edges = build_batch_size_sweep(initial_table.max_batch_tokens)
        self._samples: list[deque] = [
            deque(maxlen=sample_window) for _ in self._bin_edges
        ]
        self._rebuild_interval_steps = rebuild_interval_steps
        self._min_bin_samples = min_bin_samples
        self._clock = clock
        self._prev_stamp: Optional[tuple[float, int]] = None
        self._steps_since_rebuild = 0

    def note_non_decode_step(self) -> None:
        self._prev_stamp = None

    def observe_step(self, *, batch_tokens: int) -> Optional[SpsCostTable]:
        now = self._clock()
        prev = self._prev_stamp
        self._prev_stamp = (now, batch_tokens)
        if prev is not None:
            prev_time, prev_batch_tokens = prev
            dt = now - prev_time
            if 0.0 < dt <= ONLINE_MAX_STEP_INTERVAL_SECONDS:
                self._samples[self._bin_index(prev_batch_tokens)].append(dt)
        self._steps_since_rebuild += 1
        if self._steps_since_rebuild < self._rebuild_interval_steps:
            return None
        self._steps_since_rebuild = 0
        return self._rebuild()

    def num_measured_bins(self) -> int:
        return sum(
            1 for samples in self._samples if len(samples) >= self._min_bin_samples
        )

    def num_bins(self) -> int:
        return len(self._bin_edges)

    def _bin_index(self, batch_tokens: int) -> int:
        idx = bisect.bisect_right(self._bin_edges, batch_tokens) - 1
        return max(0, min(idx, len(self._bin_edges) - 1))

    def _rebuild(self) -> Optional[SpsCostTable]:
        measured: list[Optional[float]] = [
            (
                1.0 / statistics.median(samples)
                if len(samples) >= self._min_bin_samples
                else None
            )
            for samples in self._samples
        ]
        if all(value is None for value in measured):
            return None
        sample_steps_per_sec = [
            (
                value
                if value is not None
                else self._fallback_sps(measured=measured, idx=idx)
            )
            for idx, value in enumerate(measured)
        ]
        return SpsCostTable(
            sample_batch_tokens=list(self._bin_edges),
            sample_steps_per_sec=sample_steps_per_sec,
            max_batch_tokens=max(
                self._initial_table.max_batch_tokens, self._bin_edges[-1]
            ),
        )

    def _fallback_sps(self, *, measured: list, idx: int) -> float:
        if self._initial_is_profiled:
            return self._initial_table.lookup(self._bin_edges[idx])
        for j in range(idx - 1, -1, -1):
            if measured[j] is not None:
                return measured[j]
        for j in range(idx + 1, len(measured)):
            if measured[j] is not None:
                return measured[j]
        raise AssertionError("_rebuild guarantees at least one measured bin.")
