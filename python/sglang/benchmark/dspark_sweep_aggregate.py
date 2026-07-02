"""Aggregate the static-mode dataset acc_len sweep into a ranking.

Parses progress.log lines of the form
  <dataset>\\tt<temp>\\t<think>\t acc_len=<x>\t n=<k>\t out_len=<o>\t accuracy=<a>
(the tag part carries literal backslash-t from the bash driver; normalize both).
Reports a per-dataset x config matrix, the ranking by acc_len, and the extreme
high / low datasets, so the blog can pick the sharpest contrast.

  python3 aggregate_sweep.py /path/to/progress.log
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict


def parse(path: str) -> list[dict]:
    rows = []
    line_re = re.compile(
        r"acc_len=([0-9.]+).*?n=(\d+).*?out_len=([0-9.]+).*?accuracy=(\S+)"
    )
    for raw in open(path):
        line = raw.replace("\\t", "\t")
        if "acc_len=" not in line:
            continue
        head = line.split("\t")[0].strip()  # dataset
        m_cfg = re.search(r"t([01])\t(on|off)", line) or re.search(
            r"t([01]).*?(on|off)", line
        )
        m = line_re.search(line)
        if not m or not m_cfg:
            continue
        rows.append(
            {
                "dataset": head,
                "temp": int(m_cfg.group(1)),
                "think": m_cfg.group(2),
                "acc_len": float(m.group(1)),
                "n": int(m.group(2)),
                "out_len": float(m.group(3)),
                "accuracy": m.group(4),
            }
        )
    return rows


def main(path: str) -> None:
    rows = parse(path)
    if not rows:
        print("no rows parsed")
        return
    print(f"parsed {len(rows)} runs\n")

    # matrix: dataset -> {(temp,think): acc_len}
    mat: dict[str, dict] = defaultdict(dict)
    for r in rows:
        mat[r["dataset"]][(r["temp"], r["think"])] = r["acc_len"]

    configs = [(0, "on"), (0, "off"), (1, "on"), (1, "off")]
    header = (
        f"{'dataset':<18} "
        + " ".join(f"t{t}/{k:<3}" for t, k in configs)
        + f"  {'mean':>5} {'min':>5} {'max':>5}"
    )
    print(header)
    print("-" * len(header))

    # sort datasets by mean acc_len descending
    def dmean(ds):
        vs = list(mat[ds].values())
        return statistics.fmean(vs) if vs else 0.0

    for ds in sorted(mat, key=dmean, reverse=True):
        cells = []
        for c in configs:
            v = mat[ds].get(c)
            cells.append(f"{v:>5.2f}" if v is not None else "  -  ")
        vs = list(mat[ds].values())
        print(
            f"{ds:<18} "
            + " ".join(cells)
            + f"  {statistics.fmean(vs):>5.2f} {min(vs):>5.2f} {max(vs):>5.2f}"
        )

    print("\n=== ranking by single acc_len (all dataset x config) ===")
    for r in sorted(rows, key=lambda r: r["acc_len"], reverse=True):
        print(
            f"  {r['acc_len']:>5.2f}  {r['dataset']:<18} t{r['temp']}/{r['think']:<3} "
            f"out_len={r['out_len']:.0f} acc={r['accuracy']}"
        )

    accs = [r["acc_len"] for r in rows]
    print(
        f"\n=== spread: min={min(accs):.2f} max={max(accs):.2f} "
        f"range={max(accs)-min(accs):.2f} over {len(rows)} runs ==="
    )

    # thinking / temp marginal effects
    for dim, key in [("temp", "temp"), ("think", "think")]:
        groups: dict = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r["acc_len"])
        print(
            f"marginal by {dim}: "
            + ", ".join(
                f"{k}={statistics.fmean(v):.2f}(n{len(v)})"
                for k, v in sorted(groups.items(), key=str)
            )
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "progress.log")
