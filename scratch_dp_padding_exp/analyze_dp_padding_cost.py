"""Analytic cost model for SGLang DP-attention padding modes (MAX_LEN vs SUM_LEN).

Run: uv run python analyze_dp_padding_cost.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModeCost:
    mode: str
    local_rows_per_rank: List[int]
    global_buffer_rows: int

    @property
    def attn_rows_total(self) -> int:
        return sum(self.local_rows_per_rank)

    @property
    def mlp_rows_per_rank(self) -> int:
        return self.global_buffer_rows


def ceil_align(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def compute_mode_cost(
    mode: str, aligned_tokens: List[int], attn_tp_size: int
) -> ModeCost:
    dp_size = len(aligned_tokens)

    if mode == "MAX_LEN":
        max_len = max(aligned_tokens)
        return ModeCost(
            mode=mode,
            local_rows_per_rank=[max_len] * dp_size,
            global_buffer_rows=max_len * dp_size,
        )

    return ModeCost(
        mode=mode,
        local_rows_per_rank=list(aligned_tokens),
        global_buffer_rows=sum(aligned_tokens),
    )


def heuristic_mode(aligned_tokens: List[int]) -> str:
    dp_size = len(aligned_tokens)
    if sum(aligned_tokens) * 2 >= max(aligned_tokens) * dp_size:
        return "MAX_LEN"
    return "SUM_LEN"


@dataclass(frozen=True)
class CommCost:
    gather_bytes_per_rank: float
    combine_bytes_per_rank: float

    @property
    def total_bytes_per_rank(self) -> float:
        return self.gather_bytes_per_rank + self.combine_bytes_per_rank


def compute_comm_cost(
    cost: ModeCost, hidden_size: int, dtype_bytes: int, tp_size: int
) -> CommCost:
    # Ring-collective per-rank bus traffic for an output buffer of B bytes over
    # `tp_size` ranks: all_gather / reduce_scatter move B*(n-1)/n, all_reduce
    # moves 2*B*(n-1)/n.
    buffer_bytes = cost.global_buffer_rows * hidden_size * dtype_bytes
    factor = (tp_size - 1) / tp_size

    if cost.mode == "MAX_LEN":
        return CommCost(
            gather_bytes_per_rank=buffer_bytes * factor,
            combine_bytes_per_rank=buffer_bytes * factor,
        )

    return CommCost(
        gather_bytes_per_rank=2 * buffer_bytes * factor,
        combine_bytes_per_rank=2 * buffer_bytes * factor,
    )


def compute_gather_local_traffic(cost: ModeCost, hidden_size: int, dtype_bytes: int) -> int:
    # SUM_LEN's _dp_gather_via_all_reduce first memsets the whole global buffer
    # (write B) and memcpys the local slice in (read+write). MAX_LEN's
    # all_gather_into_tensor needs neither.
    buffer_bytes = cost.global_buffer_rows * hidden_size * dtype_bytes
    if cost.mode == "MAX_LEN":
        return 0
    return buffer_bytes


SCENARIOS = [
    ("single long prefill, others idle", [8192, 0, 0, 0, 0, 0, 0, 0]),
    ("one prefill + 7 decode(bs=64)", [8192, 64, 64, 64, 64, 64, 64, 64]),
    ("balanced prefill 8x8192", [8192] * 8),
    ("balanced prefill 8x1024 (chunked)", [1024] * 8),
    ("mildly skewed prefill", [8192, 4096, 4096, 4096, 2048, 2048, 1024, 1024]),
    ("2x long + 6 short", [8192, 8192, 512, 512, 512, 512, 512, 512]),
    ("pure decode bs=64 each", [64] * 8),
]


def main() -> None:
    hidden_size = 7168
    dtype_bytes = 2
    attn_tp_size = 1
    dp_size = 8
    tp_size = dp_size * attn_tp_size

    print(f"dp_size={dp_size} attn_tp_size={attn_tp_size} hidden={hidden_size} dtype={dtype_bytes}B\n")

    header = (
        f"{'scenario':38} {'mode':8} {'attn rows':>10} {'waste%':>7} "
        f"{'mlp rows/rank':>14} {'comm MB/rank':>13} {'memset MB':>10} {'heuristic':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, raw_tokens in SCENARIOS:
        aligned = [ceil_align(n, attn_tp_size) for n in raw_tokens]
        real_rows = sum(aligned)
        pick = heuristic_mode(aligned)
        for mode in ("MAX_LEN", "SUM_LEN"):
            cost = compute_mode_cost(mode, aligned, attn_tp_size)
            comm = compute_comm_cost(cost, hidden_size, dtype_bytes, tp_size)
            memset = compute_gather_local_traffic(cost, hidden_size, dtype_bytes)
            waste = (
                100.0 * (cost.attn_rows_total - real_rows) / cost.attn_rows_total
                if cost.attn_rows_total
                else 0.0
            )
            print(
                f"{name if mode == 'MAX_LEN' else '':38} {mode:8} "
                f"{cost.attn_rows_total:10d} {waste:6.1f}% {cost.mlp_rows_per_rank:14d} "
                f"{comm.total_bytes_per_rank / 1e6:13.1f} {memset / 1e6:10.1f} "
                f"{(pick if mode == 'MAX_LEN' else ''):>10}"
            )
        print()


if __name__ == "__main__":
    main()
