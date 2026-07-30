"""Summarize [DPPAD] lines from an sglang server log into per-step padding stats.

Run: uv run python summarize_dppad_log.py <server.log> [<server.log> ...]
"""

from __future__ import annotations

import ast
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LINE_RE = re.compile(
    r"\[DPPAD\] n=(?P<n>\d+) rank=(?P<rank>\d+) mode=(?P<mode>\w+) "
    r"raw=(?P<raw>\[[^\]]*\]) final=(?P<final>\[[^\]]*\]) local=(?P<local>\d+) "
    r"buffer=(?P<buffer>\d+) mode_fwd=(?P<fwd>\S+) extend_in_batch=(?P<extend>\w+)"
)


@dataclass(frozen=True)
class Step:
    raw: List[int]
    mode: str
    local: int
    buffer: int
    forward_mode: str

    @property
    def real_rows(self) -> int:
        return sum(self.raw)

    @property
    def max_len_rows(self) -> int:
        return max(self.raw) * len(self.raw)

    @property
    def balance(self) -> float:
        # 1.0 == perfectly balanced; 1/dp_size == only one rank has tokens
        return self.real_rows / self.max_len_rows if self.max_len_rows else 1.0


def parse(path: Path, only_rank: int = 0) -> List[Step]:
    steps: List[Step] = []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE_RE.search(line)
        if not m or int(m.group("rank")) != only_rank:
            continue
        steps.append(
            Step(
                raw=ast.literal_eval(m.group("raw")),
                mode=m.group("mode"),
                local=int(m.group("local")),
                buffer=int(m.group("buffer")),
                forward_mode=m.group("fwd"),
            )
        )
    return steps


def report(path: Path) -> None:
    steps = parse(path)
    if not steps:
        print(f"{path}: no [DPPAD] lines")
        return

    dp_size = len(steps[0].raw)
    modes = Counter(s.mode for s in steps)
    balances = [s.balance for s in steps]

    real_total = sum(s.real_rows for s in steps)
    max_total = sum(s.max_len_rows for s in steps)

    # what the pure communication heuristic (pre-PR#10414) would have chosen
    heuristic_max = sum(1 for s in steps if s.real_rows * 2 >= s.max_len_rows)

    print(f"\n=== {path} ===")
    print(f"steps logged (rank 0): {len(steps)}   dp_size={dp_size}")
    print(f"observed modes: {dict(modes)}")
    print(f"pure-comm heuristic would pick MAX_LEN in {heuristic_max}/{len(steps)} steps "
          f"({100 * heuristic_max / len(steps):.1f}%)")
    print(f"per-step balance = sum(tokens)/(max(tokens)*dp_size), 1.0 = perfect:")
    print(f"  min={min(balances):.3f} p25={statistics.quantiles(balances, n=4)[0]:.3f} "
          f"median={statistics.median(balances):.3f} "
          f"p75={statistics.quantiles(balances, n=4)[2]:.3f} max={max(balances):.3f} "
          f"mean={statistics.mean(balances):.3f}")
    print(f"rows that would enter attention projections cluster-wide:")
    print(f"  SUM_LEN (actual real rows): {real_total}")
    print(f"  MAX_LEN (padded):           {max_total}  "
          f"(+{100 * (max_total - real_total) / real_total:.1f}% wasted)")

    buckets = Counter()
    for b in balances:
        if b >= 0.95:
            buckets["balanced (>=0.95)"] += 1
        elif b >= 0.5:
            buckets["mild (0.5-0.95)"] += 1
        elif b >= 0.25:
            buckets["skewed (0.25-0.5)"] += 1
        else:
            buckets["very skewed (<0.25)"] += 1
    print("balance buckets:")
    for k in ["balanced (>=0.95)", "mild (0.5-0.95)", "skewed (0.25-0.5)", "very skewed (<0.25)"]:
        print(f"  {k:22} {buckets[k]:6d}  ({100 * buckets[k] / len(steps):5.1f}%)")


def main() -> None:
    for arg in sys.argv[1:]:
        report(Path(arg))


if __name__ == "__main__":
    main()
