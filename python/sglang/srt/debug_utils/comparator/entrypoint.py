from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    AUX_NAMES,
    SideAux,
    has_aux_tensors,
    load_and_normalize_aux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.executor import (
    execute_alignment,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.planner import (
    build_seqs_info,
    compute_alignment_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    AlignmentPlan,
    SeqsInfo,
)
from sglang.srt.debug_utils.comparator.output_types import (
    AlignWarning,
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.pipeline import (
    concat_steps,
    load_and_unshard_files,
)
from sglang.srt.debug_utils.comparator.row_matcher import MatchResult, match_rows
from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import read_meta

_BASE_SKIP_KEYS: set[str] = {"dump_index", "filename"}


def main() -> None:
    args = _parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    df_baseline = read_meta(args.baseline_path)

    df_target = read_meta(args.target_path)
    df_target = df_target.filter(
        (pl.col("step") >= args.start_step) & (pl.col("step") <= args.end_step)
    )
    if args.filter:
        df_target = df_target.filter(pl.col("filename").str.contains(args.filter))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

    print_record(
        ConfigRecord(
            baseline_path=args.baseline_path,
            target_path=args.target_path,
            diff_threshold=args.diff_threshold,
            start_step=args.start_step,
            end_step=args.end_step,
        ),
        output_format=args.output_format,
    )

    # --- alignment plan (logical mode only) ---
    plan: Optional[AlignmentPlan] = None
    if args.grouping == "logical":
        if has_aux_tensors(df_baseline) and has_aux_tensors(df_target):
            plan = _build_alignment_plan(
                args=args, df_baseline=df_baseline, df_target=df_target
            )
        else:
            print(
                "Warning: aux tensors missing, skipping token alignment",
                file=sys.stderr,
            )

    # --- skip_keys control grouping granularity ---
    skip_keys: set[str] = set(_BASE_SKIP_KEYS)
    if args.grouping == "logical":
        skip_keys |= {"rank", "step"}

    # --- unified match + iterate ---
    matches: list[MatchResult] = match_rows(
        df_baseline=df_baseline, df_target=df_target, skip_keys=skip_keys
    )

    comparison_records = _execute_comparisons(
        matches=matches,
        baseline_path=Path(args.baseline_path),
        target_path=Path(args.target_path),
        plan=plan,
        diff_threshold=args.diff_threshold,
    )
    _consume_comparison_records(comparison_records=comparison_records, output_format=args.output_format)


def _build_alignment_plan(
    *,
    args: argparse.Namespace,
    df_baseline: pl.DataFrame,
    df_target: pl.DataFrame,
) -> AlignmentPlan:
    """Load aux tensors, build token indices, and compute the alignment plan."""
    baseline_path: Path = Path(args.baseline_path)
    target_path: Path = Path(args.target_path)

    side_aux_baseline: SideAux = load_and_normalize_aux(
        dump_path=baseline_path, df=df_baseline
    )
    side_aux_target: SideAux = load_and_normalize_aux(
        dump_path=target_path, df=df_target
    )

    index_baseline: SeqsInfo = build_seqs_info(side_aux_baseline)
    index_target: SeqsInfo = build_seqs_info(side_aux_target)

    plan: AlignmentPlan = compute_alignment_plan(
        indices=Pair(x=index_baseline, y=index_target)
    )

    return plan


def _consume_comparison_records(
    *,
    comparison_records: Iterator[Union[ComparisonRecord, SkipRecord]],
    output_format: str,
) -> None:
    """Consume comparison comparison_records: count, print each, then emit summary."""
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}

    for record in comparison_records:
        counts[record.category] += 1
        print_record(record, output_format=output_format)

    print_record(
        SummaryRecord(total=sum(counts.values()), **counts),
        output_format=output_format,
    )


def _execute_comparisons(
    *,
    matches: list[MatchResult],
    baseline_path: Path,
    target_path: Path,
    plan: Optional[AlignmentPlan],
    diff_threshold: float,
) -> Iterator[Union[ComparisonRecord, SkipRecord]]:
    """Yield comparison records for all matches (unified raw/logical pipeline)."""
    for match in matches:
        if not match.rows_target:
            continue

        name: str = match.rows_target[0]["name"]

        tensors_b, b_warns = _load_and_unshard_by_step(
            rows=match.rows_baseline, base_path=baseline_path
        )
        tensors_t, t_warns = _load_and_unshard_by_step(
            rows=match.rows_target, base_path=target_path
        )
        all_warnings: list[AlignWarning] = b_warns + t_warns

        yield _compare_tensor(
            name=name,
            tensors_b=tensors_b,
            tensors_t=tensors_t,
            warnings=all_warnings,
            plan=plan,
            diff_threshold=diff_threshold,
        )


def _load_and_unshard_by_step(
    *, rows: list[dict[str, Any]], base_path: Path
) -> tuple[dict[int, torch.Tensor], list[AlignWarning]]:
    """Group rows by step, unshard within each step, return step->tensor mapping."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["step"], []).append(row)

    result: dict[int, torch.Tensor] = {}
    all_warnings: list[AlignWarning] = []

    for step in sorted(grouped):
        filenames: list[str] = [r["filename"] for r in grouped[step]]
        tensor, warnings = load_and_unshard_files(
            filenames=filenames, base_path=base_path
        )
        all_warnings.extend(warnings)
        if tensor is not None:
            result[step] = tensor

    return result, all_warnings


def _compare_tensor(
    *,
    name: str,
    tensors_b: dict[int, torch.Tensor],
    tensors_t: dict[int, torch.Tensor],
    warnings: list[AlignWarning],
    plan: Optional[AlignmentPlan],
    diff_threshold: float,
) -> Union[ComparisonRecord, SkipRecord]:
    """Compare a single tensor name by concatenating all steps into one pair."""
    if not tensors_b or not tensors_t:
        reason = "baseline_load_failed" if not tensors_b else "target_load_failed"
        return SkipRecord(name=name, reason=reason, align_warnings=warnings)

    if plan is not None and name not in AUX_NAMES:
        aligned: Pair[torch.Tensor] = execute_alignment(
            plan=plan, tensors=Pair(x=tensors_b, y=tensors_t)
        )
        combined_b, combined_t = aligned.x, aligned.y
    else:
        combined_b = concat_steps(tensors_b)
        combined_t = concat_steps(tensors_t)

    info = compare_tensors(
        x_baseline=combined_b,
        x_target=combined_t,
        name=name,
        diff_threshold=diff_threshold,
    )
    return ComparisonRecord(**info.model_dump(), align_warnings=warnings)


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
