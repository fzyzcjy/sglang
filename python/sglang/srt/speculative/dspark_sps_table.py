from __future__ import annotations

import bisect
import logging
import statistics
from typing import TYPE_CHECKING, Callable, Optional

import msgspec

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker


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
    target_worker: BaseSpecWorker,
    probe_request_counts: list[int],
    gamma: int,
    iters: int,
    time_uniform_verify_step: Callable[[BaseSpecWorker, int, int], float],
    max_batch_tokens: Optional[int] = None,
) -> SpsCostTable:
    if gamma < 1:
        raise ValueError(f"profile_sps_table requires gamma >= 1, got {gamma}.")
    if iters < 1:
        raise ValueError(f"profile_sps_table requires iters >= 1, got {iters}.")

    probe_request_counts = sorted(set(int(r) for r in probe_request_counts))
    if not probe_request_counts or probe_request_counts[0] < 1:
        raise ValueError(
            "probe_request_counts must contain positive request counts, "
            f"got {probe_request_counts}."
        )

    verify_window = 1 + gamma
    sample_batch_tokens: list[int] = []
    sample_steps_per_sec: list[float] = []

    for num_requests in probe_request_counts:
        batch_tokens = num_requests * verify_window
        durations: list[float] = []
        for _ in range(iters):
            durations.append(
                time_uniform_verify_step(target_worker, num_requests, gamma)
            )
        median_duration = statistics.median(durations)
        steps_per_sec = (1.0 / median_duration) if median_duration > 0 else float("inf")

        sample_batch_tokens.append(batch_tokens)
        sample_steps_per_sec.append(steps_per_sec)
        logger.info(
            "Profiled SPS probe: num_requests=%s, batch_tokens=%s, "
            "median_duration=%.6fs, steps_per_sec=%.3f",
            num_requests,
            batch_tokens,
            median_duration,
            steps_per_sec,
        )

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
