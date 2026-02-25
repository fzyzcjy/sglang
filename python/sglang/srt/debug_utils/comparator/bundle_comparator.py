"""Compare two tensor bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from sglang.srt.debug_utils.comparator.aligner.reorder import (
    ReorderPlan,
    compute_reorder_plans,
    execute_reorder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.executor import (
    execute_token_align,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import TokenAlignPlan
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
from sglang.srt.debug_utils.comparator.bundle_matcher import (
    TensorInfo,
    TensorBundleInfo,
)
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

_Plan = Union[UnshardPlan, ReorderPlan]


def compare_bundle_pair(
    *,
    bundle_info_pair: Pair[TensorBundleInfo],
    baseline_path: Path,
    target_path: Path,
    token_align_plan: Optional[TokenAlignPlan],
    diff_threshold: float,
) -> Union[ComparisonRecord, SkipRecord]:
    name: str = bundle_info_pair.y[0].name

    tensors_b, b_warns = _load_and_align_by_step(
        infos=bundle_info_pair.x, base_path=baseline_path
    )
    tensors_t, t_warns = _load_and_align_by_step(
        infos=bundle_info_pair.y, base_path=target_path
    )
    align_warnings: list[AlignWarning] = b_warns + t_warns
    del b_warns, t_warns

    if not tensors_b or not tensors_t:
        reason = "baseline_load_failed" if not tensors_b else "target_load_failed"
        return SkipRecord(name=name, reason=reason, align_warnings=align_warnings)

    if token_align_plan is not None:
        combined: Pair[torch.Tensor] = execute_token_align(
            plan=token_align_plan,
            tensor_of_step_pair=Pair(x=tensors_b, y=tensors_t),
        )
    else:
        assert len(tensors_b) == 1 and len(tensors_t) == 1, (
            f"Expected single-step bundles without alignment plan, "
            f"got {len(tensors_b)} baseline steps and {len(tensors_t)} target steps"
        )
        combined = Pair(
            x=list(tensors_b.values())[0],
            y=list(tensors_t.values())[0],
        )

    info = compare_tensor_pair(
        x_baseline=combined.x,
        x_target=combined.y,
        name=name,
        diff_threshold=diff_threshold,
    )
    return ComparisonRecord(**info.model_dump(), align_warnings=align_warnings)


def _load_and_align_by_step(
    *, infos: list[TensorInfo], base_path: Path
) -> tuple[dict[int, torch.Tensor], list[AlignWarning]]:
    """Group rows by step, unshard within each step, return step->tensor mapping."""
    infos_of_step: dict[int, list[TensorInfo]] = {}
    for info in infos:
        infos_of_step.setdefault(info.step, []).append(info)

    result: dict[int, torch.Tensor] = {}
    all_warnings: list[AlignWarning] = []

    for step in sorted(infos_of_step):
        filenames: list[str] = [r.filename for r in infos_of_step[step]]
        tensor, warnings = _load_and_align_one(
            filenames=filenames, base_path=base_path
        )
        all_warnings.extend(warnings)
        if tensor is not None:
            result[step] = tensor

    return result, all_warnings


def _load_and_align_one(
    *, filenames: list[str], base_path: Path
) -> tuple[Optional[torch.Tensor], list[AlignWarning]]:
    """Load tensor files and unshard them into a single tensor."""
    if not filenames:
        return None, []

    tensors_with_meta: list[ValueWithMeta] = _load_tensors(filenames, base_path)
    tensors: list[torch.Tensor] = _extract_tensors(tensors_with_meta)
    if not tensors:
        return None, []

    plans: list[_Plan] = _compute_plans([item.meta for item in tensors_with_meta])
    return _execute_plans(tensors, plans)


def _load_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [ValueWithMeta.load(base_path / f) for f in filenames]


def _compute_plans(metas: list[dict[str, Any]]) -> list[_Plan]:
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
