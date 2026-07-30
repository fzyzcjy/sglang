"""Summarize [GEMMPROBE] lines into average GEMM M per module.

`rows/calls` is the average M the module actually saw. Comparing that against
the average local real token count per step tells us whether a module runs on
LOCAL tokens (ratio ~1) or on the DP-gathered buffer (ratio ~dp_size).

Run: uv run python summarize_gemm_probe.py <server.log> [...]
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

STEP_RE = re.compile(
    r"\[GEMMPROBE\] rank=(?P<rank>\d+) steps=(?P<steps>\d+) "
    r"real_local_tokens=(?P<real>\d+)"
)
MOD_RE = re.compile(
    r"\[GEMMPROBE\] rank=(?P<rank>\d+) module=(?P<mod>\S+) rows=(?P<rows>\d+) "
    r"calls=(?P<calls>\d+) ratio=(?P<ratio>[\d.]+)"
)

ATTENTION_HINTS = (
    "self_attn",
    "input_layernorm",
    "post_attention_layernorm",
)


@dataclass
class ModuleStat:
    module: str
    rows: int
    calls: int

    @property
    def avg_m(self) -> float:
        return self.rows / self.calls if self.calls else 0.0


def parse_last_dump(path: Path, rank: int = 0) -> tuple[int, int, List[ModuleStat]]:
    """The probe dumps cumulatively; keep the last (largest) dump for `rank`."""
    text = path.read_text(errors="replace")
    steps = real = 0
    mods: Dict[str, ModuleStat] = {}
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m and int(m.group("rank")) == rank:
            if int(m.group("steps")) >= steps:
                steps = int(m.group("steps"))
                real = int(m.group("real"))
                mods = {}
            continue
        m = MOD_RE.search(line)
        if m and int(m.group("rank")) == rank:
            mods[m.group("mod")] = ModuleStat(
                module=m.group("mod"),
                rows=int(m.group("rows")),
                calls=int(m.group("calls")),
            )
    return steps, real, list(mods.values())


def report(path: Path) -> None:
    steps, real, mods = parse_last_dump(path)
    if not mods or not steps:
        print(f"{path}: no complete [GEMMPROBE] dump")
        return

    avg_local = real / steps
    print(f"\n=== {path.parent.name} ===")
    print(f"steps={steps} real_local_tokens={real} avg_local_tokens_per_step={avg_local:.1f}")
    header = f"{'module':58} {'avg M':>10} {'M/avg_local':>12} {'group':>9}"
    print(header)
    print("-" * len(header))
    for stat in sorted(mods, key=lambda s: -s.avg_m):
        group = (
            "attn"
            if any(hint in stat.module for hint in ATTENTION_HINTS)
            else "post-gather"
        )
        print(
            f"{stat.module:58} {stat.avg_m:10.1f} "
            f"{stat.avg_m / avg_local:12.3f} {group:>9}"
        )


def main() -> None:
    for arg in sys.argv[1:]:
        report(Path(arg))


if __name__ == "__main__":
    main()
