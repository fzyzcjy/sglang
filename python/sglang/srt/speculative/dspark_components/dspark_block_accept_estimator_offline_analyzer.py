from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgspec


class BlockEstimate(msgspec.Struct):
    lo: float
    hi: float
    category: str
    a_first: Optional[float]
    cap_trim_positive: bool
    true_block_accept: int


class LoadedRecords(msgspec.Struct):
    blocks: List[Dict[str, Any]]
    gathers: Dict[Tuple[str, int], List[List[Any]]]


def load_records(jsonl_path: Path) -> LoadedRecords:
    blocks: List[Dict[str, Any]] = []
    gathers: Dict[Tuple[str, int], List[List[Any]]] = defaultdict(list)
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            blocks.append(rec)
            for entry in rec.get("pg", []):
                src_fct, offset, p_lp, draft_token, realized_token = entry
                gathers[(rec["rid"], src_fct)].append(
                    [offset, p_lp, draft_token, realized_token]
                )
    return LoadedRecords(blocks=blocks, gathers=gathers)


def evaluate_block(
    rec: Dict[str, Any],
    gathers: Dict[Tuple[str, int], List[List[Any]]],
    gamma: int,
) -> BlockEstimate:
    cl = rec["cl"]
    window = rec["w"]
    cap_trim = rec.get("ct", 0)
    true_block_accept = cl + cap_trim + 1
    if "q_lp" not in rec:
        value = cl + 1.0
        return BlockEstimate(
            lo=value,
            hi=value,
            category="exact_in_window",
            a_first=None,
            cap_trim_positive=cap_trim >= 1,
            true_block_accept=true_block_accept,
        )

    q_lps = rec["q_lp"]
    entries = {e[0]: e for e in gathers.get((rec["rid"], rec["fct"]), [])}
    base = window + 1.0
    prod = 1.0
    lo_extra = 0.0
    category = "censored_resolved_exact"
    tail = 0.0
    a_first: Optional[float] = None

    for offset in range(window + 1, gamma + 1):
        entry = entries.get(offset)
        if entry is None:
            category = "censored_at_end"
            tail = prod * (gamma - offset + 1)
            break
        _, p_lp, draft_token, realized_token = entry
        a = min(1.0, math.exp(p_lp - q_lps[offset - window - 1]))
        if a_first is None:
            a_first = a
        prod *= a
        lo_extra += prod
        if draft_token != realized_token:
            if offset < gamma:
                category = "censored_diverged"
                tail = prod * (gamma - offset)
            break

    return BlockEstimate(
        lo=base + lo_extra,
        hi=base + lo_extra + tail,
        category=category,
        a_first=a_first,
        cap_trim_positive=cap_trim >= 1,
        true_block_accept=true_block_accept,
    )


def analyze(jsonl_path: Path, *, gamma: int, arm: str) -> Dict[str, Any]:
    loaded = load_records(jsonl_path)
    blocks = loaded.blocks
    results = [evaluate_block(rec, loaded.gathers, gamma) for rec in blocks]

    n = len(results)
    mean_lo = sum(r.lo for r in results) / n
    mean_hi = sum(r.hi for r in results) / n
    categories: Dict[str, int] = defaultdict(int)
    for r in results:
        categories[r.category] += 1

    censored = [r for r in results if r.category != "exact_in_window"]
    widths = [r.hi - r.lo for r in censored]

    out: Dict[str, Any] = {
        "arm": arm,
        "num_blocks": n,
        "estimate_lo": mean_lo,
        "estimate_hi": mean_hi,
        "estimate_mid": (mean_lo + mean_hi) / 2,
        "mean_bracket_width_overall": (mean_hi - mean_lo),
        "mean_bracket_width_censored": (sum(widths) / len(widths) if widths else 0.0),
        "categories": dict(categories),
        "censored_fraction": len(censored) / n if n else 0.0,
        "mean_window_drafts": sum(b["w"] for b in blocks) / n,
        "mean_cl": sum(b["cl"] for b in blocks) / n,
    }

    a_pairs = [
        (r.a_first, r.cap_trim_positive) for r in censored if r.a_first is not None
    ]
    if a_pairs:
        out["mean_analytic_a_first"] = sum(a for a, _ in a_pairs) / len(a_pairs)
        out["empirical_cap_trim_positive_rate"] = sum(1 for _, c in a_pairs if c) / len(
            a_pairs
        )

    if arm == "cap-accept":
        out.update(
            _cap_accept_truth(
                results=results, censored=censored, mean_lo=mean_lo, mean_hi=mean_hi
            )
        )

    return out


def _cap_accept_truth(
    *,
    results: List[BlockEstimate],
    censored: List[BlockEstimate],
    mean_lo: float,
    mean_hi: float,
) -> Dict[str, Any]:
    n = len(results)
    truths = [r.true_block_accept for r in results]
    true_mean = sum(truths) / n
    out: Dict[str, Any] = {
        "true_mean_block_accept": true_mean,
        "aggregate_truth_in_bracket": mean_lo - 1e-9 <= true_mean <= mean_hi + 1e-9,
        "aggregate_truth_error_vs_mid": (mean_lo + mean_hi) / 2 - true_mean,
    }
    if censored:
        censored_truth = sum(r.true_block_accept for r in censored) / len(censored)
        censored_lo = sum(r.lo for r in censored) / len(censored)
        censored_hi = sum(r.hi for r in censored) / len(censored)
        out["censored_true_mean"] = censored_truth
        out["censored_estimate_lo"] = censored_lo
        out["censored_estimate_hi"] = censored_hi
        out["censored_truth_in_bracket"] = (
            censored_lo - 1e-9 <= censored_truth <= censored_hi + 1e-9
        )
        out["calibration_bins"] = _calibration_bins(censored)
    return out


def _calibration_bins(
    censored: List[BlockEstimate], *, num_bins: int = 8
) -> List[Dict[str, Any]]:
    ordered = sorted(censored, key=lambda r: (r.lo + r.hi) / 2)
    bins: List[Dict[str, Any]] = []
    size = max(1, len(ordered) // num_bins)
    for start in range(0, len(ordered), size):
        chunk = ordered[start : start + size]
        bins.append(
            {
                "n": len(chunk),
                "mean_estimate_mid": sum((r.lo + r.hi) / 2 for r in chunk) / len(chunk),
                "mean_estimate_lo": sum(r.lo for r in chunk) / len(chunk),
                "mean_estimate_hi": sum(r.hi for r in chunk) / len(chunk),
                "mean_truth": sum(r.true_block_accept for r in chunk) / len(chunk),
            }
        )
    return bins


def analyze_driver_meta(meta_path: Path) -> Dict[str, Any]:
    total: Dict[str, float] = defaultdict(float)
    total_ct = 0.0
    n = 0
    with meta_path.open() as f:
        for line in f:
            rec = json.loads(line)
            meta = rec["meta_info"]
            ct = meta.get("spec_verify_ct", 0)
            if not ct:
                continue
            n += 1
            total_ct += ct
            for field in (
                "spec_accept_length",
                "spec_cap_length",
                "spec_block_accept_length",
            ):
                if meta.get(field) is not None:
                    total[field] += meta[field] * ct
    return {
        "num_requests_with_verify": n,
        "total_verify_ct": total_ct,
        **{f"{k}_step_weighted": v / total_ct for k, v in total.items() if total_ct},
    }


def main(
    recorder_jsonl: str,
    gamma: int,
    arm: str,
    driver_meta: Optional[str] = None,
    output: Optional[str] = None,
) -> None:
    result = analyze(Path(recorder_jsonl), gamma=gamma, arm=arm)
    if driver_meta is not None:
        result["driver_meta"] = analyze_driver_meta(Path(driver_meta))

    text = json.dumps(result, indent=2)
    print(text)
    if output is not None:
        Path(output).write_text(text + "\n")


if __name__ == "__main__":
    import typer

    typer.run(main)
