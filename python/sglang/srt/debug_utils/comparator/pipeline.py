from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.aligner.reorder import (
    ReorderPlan,
    compute_reorder_plans,
    execute_reorder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.executor import (
    execute_unshard_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.planner import (
    compute_unshard_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.types import UnshardPlan
from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.output_types import (
    AlignWarning,
    ComparisonRecord,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows

Plan = Union[UnshardPlan, ReorderPlan]


def process_tensor_group(
    *,
    name: str,
    baseline_filenames: list[str],
    target_filenames: list[str],
    baseline_path: Path,
    target_path: Path,
    diff_threshold: float,
) -> ComparisonRecord | SkipRecord:
    b_tensors = _load_tensors(baseline_filenames, baseline_path)
    t_tensors = _load_tensors(target_filenames, target_path)

    b_plans, t_plans = _compute_plans(
        baseline_metas=[item.meta for item in b_tensors],
        target_metas=[item.meta for item in t_tensors],
    )

    b_extracted = _extract_tensors(b_tensors)
    t_extracted = _extract_tensors(t_tensors)
    del b_tensors, t_tensors

    b_tensor, b_warns = _execute_plans(b_extracted, b_plans)
    t_tensor, t_warns = _execute_plans(t_extracted, t_plans)
    all_warnings: list[AlignWarning] = b_warns + t_warns

    if b_tensor is None or t_tensor is None:
        reason = "baseline_load_failed" if b_tensor is None else "target_load_failed"
        return SkipRecord(name=name, reason=reason, align_warnings=all_warnings)

    info = compare_tensors(
        x_baseline=b_tensor,
        x_target=t_tensor,
        name=name,
        diff_threshold=diff_threshold,
    )

    return ComparisonRecord(**info.model_dump(), align_warnings=all_warnings)


def _load_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [ValueWithMeta.load(base_path / f) for f in filenames]


def _compute_plans(
    *,
    baseline_metas: list[dict[str, Any]],
    target_metas: list[dict[str, Any]],
) -> tuple[list[Plan], list[Plan]]:
    """This function deliberately takes metadata, since plan computation must never inspect actual tensor data."""
    return (
        _compute_plans_for_group(baseline_metas),
        _compute_plans_for_group(target_metas),
    )


def _compute_plans_for_group(metas: list[dict[str, Any]]) -> list[Plan]:
    if not metas or len(metas) == 1:
        return []

    dims_str = metas[0].get("dims")
    if dims_str is None:
        return []

    dim_specs = parse_dims(dims_str)
    parallel_infos = [normalize_parallel_info(meta) for meta in metas]

    unshard_plans = compute_unshard_plan(
        dim_specs=dim_specs, parallel_infos=parallel_infos
    )
    reorder_plans = compute_reorder_plans(
        dim_specs=dim_specs, parallel_infos=parallel_infos
    )
    return [*unshard_plans, *reorder_plans]


def _extract_tensors(
    loaded: list[ValueWithMeta],
) -> Optional[list[torch.Tensor]]:
    return [value for item in loaded if isinstance(value := item.value, torch.Tensor)]


def _execute_plans(
    tensors: list[torch.Tensor],
    plans: list[Plan],
) -> tuple[Optional[torch.Tensor], list[AlignWarning]]:
    if not tensors:
        return None, []

    if not plans:
        if len(tensors) != 1:
            return None, []
        return tensors[0], []

    warnings: list[AlignWarning] = []
    current = tensors
    for plan in plans:
        current, new_warnings = _execute_plan(current, plan)
        warnings.extend(new_warnings)

    assert len(current) == 1
    return current[0], warnings


def _execute_plan(
    tensors: list[torch.Tensor],
    plan: Plan,
) -> tuple[list[torch.Tensor], list[AlignWarning]]:
    if isinstance(plan, UnshardPlan):
        return execute_unshard_plan(plan, tensors)
    elif isinstance(plan, ReorderPlan):
        return execute_reorder_plan(plan, tensors), []
    else:
        raise NotImplementedError(f"Unknown {plan=}")


def load_and_unshard_for_step(
    *, name: str, step: int, df: pl.DataFrame, dump_path: Path
) -> tuple[Optional[torch.Tensor], list[AlignWarning]]:
    """Load all rank files for (name, step), unshard into a single tensor."""
    rows: list[dict] = filter_rows(df, conditions={"name": name, "step": step})
    if not rows:
        return None, []

    filenames: list[str] = [r["filename"] for r in rows]
    loaded: list[ValueWithMeta] = _load_tensors(filenames, dump_path)

    plans: list[Plan] = _compute_plans_for_group([item.meta for item in loaded])
    tensors: Optional[list[torch.Tensor]] = _extract_tensors(loaded)
    if tensors is None:
        return None, []

    return _execute_plans(tensors, plans)


def load_and_unshard_all_steps(
    *, name: str, df: pl.DataFrame, dump_path: Path
) -> tuple[dict[int, torch.Tensor], list[AlignWarning]]:
    """Load and unshard a tensor across all steps, returning step→tensor mapping."""
    step_values: list[int] = sorted(
        df.filter(pl.col("name") == name)["step"].unique().to_list()
    )

    result: dict[int, torch.Tensor] = {}
    all_warnings: list[AlignWarning] = []

    for step in step_values:
        tensor, warnings = load_and_unshard_for_step(
            name=name, step=step, df=df, dump_path=dump_path
        )
        all_warnings.extend(warnings)
        if tensor is not None:
            result[step] = tensor

    return result, all_warnings
