from __future__ import annotations

import bisect
import logging
from typing import Optional

import msgspec

logger = logging.getLogger(__name__)


def floor_probe_index(edges: list[int], batch_tokens: int) -> int:
    # Tensor mirror: dspark_scheduler._lookup_sps_tensor (keep in sync).
    idx = bisect.bisect_right(edges, batch_tokens) - 1
    return max(0, min(idx, len(edges) - 1))


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
        return self.sample_steps_per_sec[
            floor_probe_index(self.sample_batch_tokens, batch_tokens)
        ]

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> SpsCostTable:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def _interp_clamped(xs: list[int], ys: list[float], x: float) -> float:
    # Piecewise-linear with edge clamp. xs strictly increasing.
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    hi = bisect.bisect_right(xs, x)
    lo = hi - 1
    frac = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + frac * (ys[hi] - ys[lo])


class SpsAdditiveCostTable(msgspec.Struct, frozen=True):
    """Additive step-cost model: T(bs, M) = bias + alpha(bs) + theta(M), in
    seconds, with bs the running request count and M = bs + K the total packed
    verify tokens (1-token-per-request floor + K trimmable candidates).

    The three components are mutually independent lookup functions:
    - bias: per-step fixed cost (launch / scheduling), invariant to bs and M.
    - alpha(bs): request-scaling cost (draft pass, per-request KV/attention),
      invariant to M -- the part trimming can never recover.
    - theta(M): verify-token-scaling cost of the target forward over M tokens;
      trimming lowers M and only this term.

    Indexed on M (not K): along an off-diagonal (bs, K) grid the K-ranges are
    disjoint across bs, but the M = bs + K ranges overlap, so theta is
    identifiable (the collapse test: equal M + higher bs -> higher T isolates
    alpha). Unlike the 1D diagonal SpsCostTable, whose single slope conflates
    alpha' with theta' and overestimates the marginal saving of trimming.

    Gauge convention (identifiability): alpha(bs_probes[0]) = 0 and
    theta(m_probes[0]) = 0; bias absorbs the reference-point cost. Only the
    sum is ever consumed, so the gauge is a storage convention.
    """

    bias_seconds: float
    bs_probes: list[int]
    alpha_seconds: list[float]
    m_probes: list[int]
    theta_seconds: list[float]

    def __post_init__(self) -> None:
        for name, probes, values in (
            ("bs", self.bs_probes, self.alpha_seconds),
            ("m", self.m_probes, self.theta_seconds),
        ):
            if not probes:
                raise ValueError(f"SpsAdditiveCostTable requires {name}_probes.")
            if probes != sorted(set(probes)):
                raise ValueError(
                    f"{name}_probes must be strictly increasing, got {probes}."
                )
            if len(probes) != len(values):
                raise ValueError(
                    f"{name}_probes and its values must have equal length, got "
                    f"{len(probes)} vs {len(values)}."
                )
        if self.bias_seconds <= 0:
            raise ValueError(f"bias_seconds must be > 0, got {self.bias_seconds}.")

    def step_time(self, *, num_reqs: int, budget: int) -> float:
        # budget = K above the floor; the verify forward sees M = num_reqs + K.
        return (
            self.bias_seconds
            + _interp_clamped(self.bs_probes, self.alpha_seconds, float(num_reqs))
            + _interp_clamped(
                self.m_probes, self.theta_seconds, float(num_reqs + budget)
            )
        )

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> SpsAdditiveCostTable:
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


def load_sps_table_from_path(path: str):
    # Sniff by the discriminating top-level key: only the additive model
    # carries "bias_seconds"; the 1D diagonal table stays the default.
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    if '"bias_seconds"' in data:
        return SpsAdditiveCostTable.from_json(data)
    return SpsCostTable.from_json(data)


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
