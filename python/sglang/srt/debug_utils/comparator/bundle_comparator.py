"""Compare two tensor bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch

from sglang.srt.debug_utils.comparator.aligner.reorder import (
    ReordererPlan,
    compute_reorder_plans,
    execute_reorder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.executor import (
    execute_token_align,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import TokenAlignerPlan
from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    execute_unshard_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unshard_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.output_types import (
    AlignWarning,
    ComparisonRecord,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import ValueWithMeta

_Plan = Union[UnsharderPlan, ReordererPlan]


@dataclass(frozen=True)
class _StepGroupPlan:
    """Unshard + reorder plan for a single step."""

    step: int
    input_indices: list[int]
    unshard_reorder: list[_Plan]


@dataclass(frozen=True)
class _AlignPlan:
    """Unified plan: per-step unshard/reorder for both sides + cross-side token alignment."""

    side_plans: Pair[list[_StepGroupPlan]]
    token_align: Optional[TokenAlignerPlan]


def compare_bundle_pair(
    *,
    name: str,
    filenames_pair: Pair[list[str]],
    baseline_path: Path,
    target_path: Path,
    token_align_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
) -> Union[ComparisonRecord, SkipRecord]:
    # 1. Load (tensor + meta, ungrouped)
    loaded_pair: Pair[list[ValueWithMeta]] = Pair(
        x=_load_tensors(filenames=filenames_pair.x, base_path=baseline_path),
        y=_load_tensors(filenames=filenames_pair.y, base_path=target_path),
    )

    # Filter failed loads, keep meta/tensor aligned
    valid_pair: Pair[list[ValueWithMeta]] = Pair(
        x=[it for it in loaded_pair.x if isinstance(it.value, torch.Tensor)],
        y=[it for it in loaded_pair.y if isinstance(it.value, torch.Tensor)],
    )
    if not valid_pair.x or not valid_pair.y:
        reason = "baseline_load_failed" if not valid_pair.x else "target_load_failed"
        return SkipRecord(name=name, reason=reason, align_warnings=[])

    # 2. Plan (meta only)
    metas_pair: Pair[list[dict[str, Any]]] = Pair(
        x=[it.meta for it in valid_pair.x],
        y=[it.meta for it in valid_pair.y],
    )
    plan: _AlignPlan = _compute_plans(
        metas_pair=metas_pair, token_align_plan=token_align_plan
    )

    # 3. Execute (tensor + plan only)
    tensors_pair: Pair[list[torch.Tensor]] = Pair(
        x=[it.value for it in valid_pair.x],
        y=[it.value for it in valid_pair.y],
    )
    result: Optional[Pair[torch.Tensor]]
    result, align_warnings, failed_side = _execute_plans(
        tensors_pair=tensors_pair, plan=plan
    )

    if result is None:
        reason = (
            "baseline_load_failed"
            if failed_side == "baseline"
            else "target_load_failed"
        )
        return SkipRecord(name=name, reason=reason, align_warnings=align_warnings)

    # 4. Compare
    info = compare_tensor_pair(
        x_baseline=result.x,
        x_target=result.y,
        name=name,
        diff_threshold=diff_threshold,
    )
    return ComparisonRecord(**info.model_dump(), align_warnings=align_warnings)


def _load_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [ValueWithMeta.load(base_path / f) for f in filenames]


def _compute_plans(
    *,
    metas_pair: Pair[list[dict[str, Any]]],
    token_align_plan: Optional[TokenAlignerPlan],
) -> _AlignPlan:
    return _AlignPlan(
        side_plans=Pair(
            x=_compute_side_plans(metas=metas_pair.x),
            y=_compute_side_plans(metas=metas_pair.y),
        ),
        token_align=token_align_plan,
    )


def _compute_side_plans(metas: list[dict[str, Any]]) -> list[_StepGroupPlan]:
    """Group by step, compute unshard + reorder plans for each group."""
    step_to_indices: dict[int, list[int]] = {}
    for i, meta in enumerate(metas):
        step: int = int(meta["step"])
        step_to_indices.setdefault(step, []).append(i)

    result: list[_StepGroupPlan] = []
    for step in sorted(step_to_indices):
        indices: list[int] = step_to_indices[step]
        step_metas: list[dict[str, Any]] = [metas[i] for i in indices]
        plans: list[_Plan] = _compute_step_unshard_reorder(metas=step_metas)
        result.append(
            _StepGroupPlan(step=step, input_indices=indices, unshard_reorder=plans)
        )

    return result


def _compute_step_unshard_reorder(metas: list[dict[str, Any]]) -> list[_Plan]:
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


def _execute_plans(
    *,
    tensors_pair: Pair[list[torch.Tensor]],
    plan: _AlignPlan,
) -> tuple[Optional[Pair[torch.Tensor]], list[AlignWarning], Optional[str]]:
    """Execute unified unshard/reorder + token-align.

    Returns:
        - Combined tensor pair (or None if failed)
        - List of alignment warnings
        - Failed side ("baseline" or "target" if failed, None if successful)
    """

    # Per-side: unshard + reorder -> dict[step, tensor]
    step_tensors_b, b_warns = _execute_side_plans(
        tensors=tensors_pair.x, step_plans=plan.side_plans.x
    )
    step_tensors_t, t_warns = _execute_side_plans(
        tensors=tensors_pair.y, step_plans=plan.side_plans.y
    )
    all_warnings: list[AlignWarning] = b_warns + t_warns

    if not step_tensors_b or not step_tensors_t:
        failed_side = "baseline" if not step_tensors_b else "target"
        return None, all_warnings, failed_side

    # Cross-side: token alignment (or direct extraction for single-step)
    if plan.token_align is not None:
        combined: Pair[torch.Tensor] = execute_token_align(
            plan=plan.token_align,
            tensor_of_step_pair=Pair(x=step_tensors_b, y=step_tensors_t),
        )
    else:
        assert len(step_tensors_b) == 1 and len(step_tensors_t) == 1
        combined = Pair(
            x=list(step_tensors_b.values())[0],
            y=list(step_tensors_t.values())[0],
        )

    return combined, all_warnings, None


def _execute_side_plans(
    tensors: list[torch.Tensor],
    step_plans: list[_StepGroupPlan],
) -> tuple[dict[int, torch.Tensor], list[AlignWarning]]:
    """Execute per-step unshard + reorder for one side."""
    result: dict[int, torch.Tensor] = {}
    all_warnings: list[AlignWarning] = []

    for step_plan in step_plans:
        step_tensors: list[torch.Tensor] = [tensors[i] for i in step_plan.input_indices]
        tensor, warnings = _execute_step_plans(
            tensors=step_tensors, plans=step_plan.unshard_reorder
        )
        all_warnings.extend(warnings)
        if tensor is not None:
            result[step_plan.step] = tensor

    return result, all_warnings


def _execute_step_plans(
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
        current, new_warnings = _execute_single_plan(tensors=current, plan=plan)
        warnings.extend(new_warnings)

    assert len(current) == 1
    return current[0], warnings


def _execute_single_plan(
    tensors: list[torch.Tensor],
    plan: _Plan,
) -> tuple[list[torch.Tensor], list[AlignWarning]]:
    if isinstance(plan, UnsharderPlan):
        return execute_unshard_plan(plan, tensors)
    elif isinstance(plan, ReordererPlan):
        return execute_reorder_plan(plan, tensors), []
    else:
        raise NotImplementedError(f"Unknown {plan=}")
