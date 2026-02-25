from __future__ import annotations

from typing import Any, Optional

from sglang.srt.debug_utils.comparator.aligner.reorderer.planner import (
    compute_reorderer_plans,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignPlan,
    StepGroupPlan,
    StepPlan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.utils import Pair


def compute_align_plan(
    *,
    metas_pair: Pair[list[dict[str, Any]]],
    token_aligner_plan: Optional[TokenAlignerPlan],
) -> AlignPlan:
    return AlignPlan(
        side_plans=metas_pair.map(lambda metas: _compute_side_plans(metas=metas)),
        token_align=token_aligner_plan,
    )


def _compute_side_plans(metas: list[dict[str, Any]]) -> list[StepGroupPlan]:
    """Group by step, compute unshard + reorder plans for each group."""
    step_to_indices: dict[int, list[int]] = {}
    for i, meta in enumerate(metas):
        step: int = int(meta["step"])
        step_to_indices.setdefault(step, []).append(i)

    result: list[StepGroupPlan] = []
    for step in sorted(step_to_indices):
        indices: list[int] = step_to_indices[step]
        step_metas: list[dict[str, Any]] = [metas[i] for i in indices]
        plans: list[StepPlan] = _compute_step_unshard_reorder(metas=step_metas)
        result.append(
            StepGroupPlan(step=step, input_indices=indices, unshard_reorder=plans)
        )

    return result


def _compute_step_unshard_reorder(metas: list[dict[str, Any]]) -> list[StepPlan]:
    if not metas or len(metas) == 1:
        return []

    dims_str = metas[0].get("dims")
    if dims_str is None:
        return []

    dim_specs = parse_dims(dims_str)
    parallel_infos = [normalize_parallel_info(meta) for meta in metas]

    unsharder_plans = compute_unsharder_plan(
        dim_specs=dim_specs, parallel_infos=parallel_infos
    )
    reorderer_plans = compute_reorderer_plans(
        dim_specs=dim_specs, parallel_infos=parallel_infos
    )
    return [*unsharder_plans, *reorderer_plans]
