from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional, Union

import polars as pl

from sglang.srt.debug_utils.comparator.aligner.token_align.entrypoint import (
    compute_maybe_alignment_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    TokenAlignPlan,
)
from sglang.srt.debug_utils.comparator.bundle_comparator import compare_bundle_pair
from sglang.srt.debug_utils.comparator.bundle_matcher import TensorInfoBundle, match_bundles
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import read_meta


def main() -> None:
    args = _parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    print_record(
        ConfigRecord.from_args(args),
        output_format=args.output_format,
    )

    df_baseline, df_target = _read_df(args)

    alignment_plan = compute_maybe_alignment_plan(args, df_baseline, df_target)

    bundle_pairs: list[Pair[TensorInfoBundle]] = match_bundles(
        df_baseline=df_baseline, df_target=df_target, skip_keys=_compute_skip_keys(args)
    )

    comparison_records = _execute_compare_bundle_pair(
        bundle_pairs=bundle_pairs,
        baseline_path=Path(args.baseline_path),
        target_path=Path(args.target_path),
        alignment_plan=alignment_plan,
        diff_threshold=args.diff_threshold,
    )
    _consume_comparison_records(
        comparison_records=comparison_records, output_format=args.output_format
    )


def _read_df(args):
    df_baseline = read_meta(args.baseline_path)

    df_target = read_meta(args.target_path)
    df_target = df_target.filter(
        (pl.col("step") >= args.start_step) & (pl.col("step") <= args.end_step)
    )
    if args.filter:
        df_target = df_target.filter(pl.col("filename").str.contains(args.filter))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

    return df_baseline, df_target


def _compute_skip_keys(args):
    skip_keys: set[str] = {"dump_index", "filename"}
    if args.grouping == "logical":
        skip_keys |= {"rank", "step"}
    return skip_keys


def _execute_compare_bundle_pair(
    *,
    bundle_pairs: list[Pair[TensorInfoBundle]],
    baseline_path: Path,
    target_path: Path,
    alignment_plan: Optional[TokenAlignPlan],
    diff_threshold: float,
) -> Iterator[Union[ComparisonRecord, SkipRecord]]:
    for pair in bundle_pairs:
        if not pair.y:
            continue

        yield compare_bundle_pair(
            bundles=pair,
            baseline_path=baseline_path,
            target_path=target_path,
            alignment_plan=alignment_plan,
            diff_threshold=diff_threshold,
        )


def _consume_comparison_records(
    *,
    comparison_records: Iterator[Union[ComparisonRecord, SkipRecord]],
    output_format: str,
) -> None:
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}

    for record in comparison_records:
        counts[record.category] += 1
        print_record(record, output_format=output_format)

    print_record(
        SummaryRecord(total=sum(counts.values()), **counts),
        output_format=output_format,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=1000000)
    parser.add_argument("--diff-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--filter", type=str, default=None, help="Regex to filter filenames"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json (JSONL, one JSON object per line)",
    )
    parser.add_argument(
        "--grouping",
        type=str,
        choices=["logical", "raw"],
        default="logical",
        help="Grouping mode: logical (cross-rank unshard) or raw (rank-by-rank)",
    )
    return parser.parse_args()
