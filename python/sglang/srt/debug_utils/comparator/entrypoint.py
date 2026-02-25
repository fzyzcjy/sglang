import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    SideAux,
    get_comparable_names,
    has_aux_tensors,
    load_and_normalize_aux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.executor import (
    execute_alignment,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.planner import (
    build_token_index,
    compute_alignment_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    AlignmentPlan,
    SideTokenIndex,
    format_alignment_summary,
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
    load_and_unshard_all_steps,
    process_tensor_group,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.dump_loader import filter_rows, read_meta

_NON_KEY_COLS = {"dump_index", "filename"}


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

    grouping: str = args.grouping

    if grouping == "raw":
        _run_raw(
            args=args,
            df_baseline=df_baseline,
            df_target=df_target,
        )
    else:
        plan: Optional[AlignmentPlan] = None
        if has_aux_tensors(df_baseline) and has_aux_tensors(df_target):
            plan = _build_alignment_plan(
                args=args,
                df_baseline=df_baseline,
                df_target=df_target,
            )
        else:
            print(
                "Warning: aux tensors missing, skipping token alignment",
                file=sys.stderr,
            )

        _run_logical(
            args=args,
            df_baseline=df_baseline,
            df_target=df_target,
            plan=plan,
        )


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

    index_baseline: SideTokenIndex = build_token_index(side_aux_baseline)
    index_target: SideTokenIndex = build_token_index(side_aux_target)

    plan: AlignmentPlan = compute_alignment_plan(
        index_a=index_baseline, index_b=index_target
    )
    print(format_alignment_summary(plan.summary), file=sys.stderr)

    return plan


def _run_logical(
    *,
    args: argparse.Namespace,
    df_baseline: pl.DataFrame,
    df_target: pl.DataFrame,
    plan: Optional[AlignmentPlan],
) -> None:
    """Unified logical comparison: concat all steps per name, optionally alignment-aware."""
    baseline_path: Path = Path(args.baseline_path)
    target_path: Path = Path(args.target_path)

    comparable_names: list[str] = get_comparable_names(
        df_baseline=df_baseline,
        df_target=df_target,
        exclude_aux=(plan is not None),
    )

    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}

    for tensor_name in comparable_names:
        tensors_b, b_warns = load_and_unshard_all_steps(
            name=tensor_name, df=df_baseline, dump_path=baseline_path
        )
        tensors_t, t_warns = load_and_unshard_all_steps(
            name=tensor_name, df=df_target, dump_path=target_path
        )
        all_warnings: list[AlignWarning] = b_warns + t_warns

        record = _compare_tensor(
            name=tensor_name,
            tensors_b=tensors_b,
            tensors_t=tensors_t,
            warnings=all_warnings,
            plan=plan,
            diff_threshold=args.diff_threshold,
        )

        counts[record.category] += 1
        print_record(record, output_format=args.output_format)

    print_record(
        SummaryRecord(total=sum(counts.values()), **counts),
        output_format=args.output_format,
    )


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

    if plan is not None:
        combined_b, combined_t = execute_alignment(
            plan=plan, tensors_a=tensors_b, tensors_b=tensors_t
        )
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


def _run_raw(
    *,
    args: argparse.Namespace,
    df_baseline: pl.DataFrame,
    df_target: pl.DataFrame,
) -> None:
    """Raw comparison path: rank-by-rank, no cross-rank unshard."""
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}

    key_cols: list[str] = [c for c in df_target.columns if c not in _NON_KEY_COLS]
    tensor_group_keys: pl.DataFrame = df_target.unique(subset=key_cols)

    for tensor_group_key in tensor_group_keys.iter_rows(named=True):
        conditions: dict[str, object] = {k: tensor_group_key[k] for k in key_cols}
        baseline_rows: list[dict] = filter_rows(df_baseline, conditions=conditions)
        target_rows: list[dict] = filter_rows(df_target, conditions=conditions)

        record = process_tensor_group(
            name=tensor_group_key["name"],
            baseline_filenames=[r["filename"] for r in baseline_rows],
            target_filenames=[r["filename"] for r in target_rows],
            baseline_path=Path(args.baseline_path),
            target_path=Path(args.target_path),
            diff_threshold=args.diff_threshold,
        )
        counts[record.category] += 1
        print_record(record, output_format=args.output_format)

    print_record(
        SummaryRecord(total=sum(counts.values()), **counts),
        output_format=args.output_format,
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
