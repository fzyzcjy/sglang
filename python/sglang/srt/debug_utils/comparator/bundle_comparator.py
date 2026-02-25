"""Compare two tensor bundles which will be unified and then aligned."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from sglang.srt.debug_utils.comparator.aligner.reorder import (
    ReorderPlan,
    compute_reorder_plans,
    execute_reorder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import AUX_NAMES
from sglang.srt.debug_utils.comparator.aligner.token_align.executor import (
    execute_alignment,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import TokenAlignmentPlan
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
from sglang.srt.debug_utils.comparator.bundle_matcher import TensorBundle
from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import ValueWithMeta

_Plan = Union[UnshardPlan, ReorderPlan]


def compare_bundles(
    *,
    bundles: Pair[TensorBundle],
    baseline_path: Path,
    target_path: Path,
    alignment_plan: Optional[TokenAlignmentPlan],
    diff_threshold: float,
) -> Union[ComparisonRecord, SkipRecord]:
    """Compare a matched pair of tensor bundles across all steps."""
    name: str = bundles.y[0]["name"]

    tensors_b, b_warns = _load_and_unshard_by_step(
        rows=bundles.x, base_path=baseline_path
    )
    tensors_t, t_warns = _load_and_unshard_by_step(
        rows=bundles.y, base_path=target_path
    )
    all_warnings: list[AlignWarning] = b_warns + t_warns

    return _compare_tensor(
        name=name,
        tensors_b=tensors_b,
        tensors_t=tensors_t,
        warnings=all_warnings,
        alignment_plan=alignment_plan,
        diff_threshold=diff_threshold,
    )


def _compare_tensor(
    *,
    name: str,
    tensors_b: dict[int, torch.Tensor],
    tensors_t: dict[int, torch.Tensor],
    warnings: list[AlignWarning],
    alignment_plan: Optional[TokenAlignmentPlan],
    diff_threshold: float,
) -> Union[ComparisonRecord, SkipRecord]:
    """Compare a single tensor name by concatenating all steps into one pair."""
    if not tensors_b or not tensors_t:
        reason = "baseline_load_failed" if not tensors_b else "target_load_failed"
        return SkipRecord(name=name, reason=reason, align_warnings=warnings)

    if alignment_plan is not None and name not in AUX_NAMES:
        aligned: Pair[torch.Tensor] = execute_alignment(
            plan=alignment_plan, tensors=Pair(x=tensors_b, y=tensors_t)
        )
        combined_b, combined_t = aligned.x, aligned.y
    else:
        combined_b = _concat_steps(tensors_b)
        combined_t = _concat_steps(tensors_t)

    info = compare_tensors(
        x_baseline=combined_b,
        x_target=combined_t,
        name=name,
        diff_threshold=diff_threshold,
    )
    return ComparisonRecord(**info.model_dump(), align_warnings=warnings)


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
        tensor, warnings = _load_and_unshard_files(
            filenames=filenames, base_path=base_path
        )
        all_warnings.extend(warnings)
        if tensor is not None:
            result[step] = tensor

    return result, all_warnings


def _load_and_unshard_files(
    *, filenames: list[str], base_path: Path
) -> tuple[Optional[torch.Tensor], list[AlignWarning]]:
    """Load tensor files and unshard them into a single tensor."""
    if not filenames:
        return None, []

    loaded: list[ValueWithMeta] = _load_tensors(filenames, base_path)
    plans: list[_Plan] = _compute_plans_for_group([item.meta for item in loaded])
    tensors: list[torch.Tensor] = _extract_tensors(loaded)
    if not tensors:
        return None, []

    return _execute_plans(tensors, plans)


def _concat_steps(tensors: dict[int, torch.Tensor]) -> torch.Tensor:
    """Concat all step tensors into a single tensor (sorted by step key)."""
    sorted_tensors: list[torch.Tensor] = [tensors[s] for s in sorted(tensors)]
    if sorted_tensors[0].ndim == 0:
        return torch.stack(sorted_tensors)
    return torch.cat(sorted_tensors, dim=0)


def _load_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [ValueWithMeta.load(base_path / f) for f in filenames]


def _compute_plans_for_group(metas: list[dict[str, Any]]) -> list[_Plan]:
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
) -> list[torch.Tensor]:
    return [value for item in loaded if isinstance(value := item.value, torch.Tensor)]


def _execute_plans(
    tensors: list[torch.Tensor],
    plans: list[_Plan],
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
    plan: _Plan,
) -> tuple[list[torch.Tensor], list[AlignWarning]]:
    if isinstance(plan, UnshardPlan):
        return execute_unshard_plan(plan, tensors)
    elif isinstance(plan, ReorderPlan):
        return execute_reorder_plan(plan, tensors), []
    else:
        raise NotImplementedError(f"Unknown {plan=}")
