"""Join [STEP] and [PCG] lines and report wasted attention rows per padding mode.

Rows reaching the attention projections for one step on one rank:
  * graphed step (a [PCG] line exists for the same counter on the same rank):
    the captured bucket -- the local tensors are padded up to it
  * eager step: local_rows, i.e. what _pad_inputs_to_size padded to

Waste = those rows minus the rank's real pre-padding token count.

Run: uv run python summarize_step_pcg.py <server.log> [...]
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

STEP_RE = re.compile(
    r"DP(?P<logrank>\d+) [^\]]*\] \[STEP\] n=(?P<n>\d+) rank=(?P<rank>\d+) "
    r"mode=(?P<mode>\w+) raw=(?P<raw>\[[^\]]*\]) final=(?P<final>\[[^\]]*\]) "
    r"local_rows=(?P<local>\d+) buffer=(?P<buffer>\d+) fwd=(?P<fwd>\S+) "
    r"bcg_ok=(?P<bcg>\w+)"
)
PCG_RE = re.compile(
    r"DP(?P<logrank>\d+) [^\]]*\] \[PCG\] n=(?P<n>\d+) local_rows_in=(?P<local>\d+) "
    r"bucket=(?P<bucket>\d+) mode=(?P<mode>\S+)"
)


@dataclass
class Step:
    n: int
    rank: int
    mode: str
    raw: List[int]
    local_rows: int
    buffer: int
    fwd: str
    bcg_ok: str
    bucket: Optional[int] = None

    @property
    def real_rows(self) -> int:
        return self.raw[self.rank] if self.rank < len(self.raw) else 0

    @property
    def attn_rows(self) -> int:
        return self.bucket if self.bucket is not None else self.local_rows

    @property
    def graphed(self) -> bool:
        return self.bucket is not None


@dataclass
class Totals:
    steps: int = 0
    graphed: int = 0
    real: int = 0
    attn: int = 0
    modes: Counter = field(default_factory=Counter)


def parse(path: Path, rank: int = 0) -> List[Step]:
    steps: Dict[int, Step] = {}
    pcg: Dict[int, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m and int(m.group("logrank")) == rank:
            n = int(m.group("n"))
            steps[n] = Step(
                n=n,
                rank=int(m.group("rank")),
                mode=m.group("mode"),
                raw=ast.literal_eval(m.group("raw")),
                local_rows=int(m.group("local")),
                buffer=int(m.group("buffer")),
                fwd=m.group("fwd"),
                bcg_ok=m.group("bcg"),
            )
            continue
        m = PCG_RE.search(line)
        if m and int(m.group("logrank")) == rank:
            pcg[int(m.group("n"))] = int(m.group("bucket"))
    for n, bucket in pcg.items():
        if n in steps:
            steps[n].bucket = bucket
    return [steps[n] for n in sorted(steps)]


def report(path: Path) -> None:
    steps = parse(path)
    if not steps:
        print(f"{path.parent.name}: no [STEP] lines")
        return

    tot = Totals()
    for s in steps:
        tot.steps += 1
        tot.graphed += int(s.graphed)
        tot.real += s.real_rows
        tot.attn += s.attn_rows
        tot.modes[s.mode] += 1

    waste = tot.attn - tot.real
    pct = 100.0 * waste / tot.real if tot.real else 0.0
    print(f"\n=== {path.parent.name} ===")
    print(f"extend steps (rank 0 view): {tot.steps}   modes={dict(tot.modes)}")
    print(f"prefill CUDA graph used in : {tot.graphed} steps "
          f"({100 * tot.graphed / tot.steps:.1f}%)")
    print(f"real local tokens          : {tot.real}")
    print(f"rows reaching attention    : {tot.attn}")
    print(f"wasted attention rows      : {waste} ({pct:+.1f}% vs real)")

    graphed = [s for s in steps if s.graphed]
    if graphed:
        g_real = sum(s.real_rows for s in graphed)
        g_attn = sum(s.attn_rows for s in graphed)
        print(f"  graphed subset: real={g_real} attn={g_attn} "
              f"waste={g_attn - g_real} ({100 * (g_attn - g_real) / g_real:+.1f}%)")
        print(f"  graphed bucket histogram: "
              f"{dict(Counter(s.bucket for s in graphed).most_common(8))}")
    eager = [s for s in steps if not s.graphed]
    if eager:
        e_real = sum(s.real_rows for s in eager)
        e_attn = sum(s.attn_rows for s in eager)
        print(f"  eager subset  : real={e_real} attn={e_attn} "
              f"waste={e_attn - e_real} ({100 * (e_attn - e_real) / e_real:+.1f}%)")


def main() -> None:
    for arg in sys.argv[1:]:
        report(Path(arg))


if __name__ == "__main__":
    main()
