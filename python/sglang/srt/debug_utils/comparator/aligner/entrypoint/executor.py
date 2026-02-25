from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.aligner.reorderer.executor import (
    execute_reorderer_plan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan
from sglang.srt.debug_utils.comparator.aligner.token_aligner.executor import (
    execute_token_aligner,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPlan,
    AlignerPerStepPlan,
    AlignerPerStepSubPlan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.output_types import AlignWarning
from sglang.srt.debug_utils.comparator.utils import Pair


@dataclass(frozen=True)
class AlignerResult:
    tensors: Optional[Pair[torch.Tensor]]
    warnings: list[AlignWarning]
    failed_side_xy: Optional[str]  # "x" or "y"; None if success


def execute_aligner_plan(
    *,
    tensors_pair: Pair[list[torch.Tensor]],
    plan: AlignerPlan,
) -> AlignerResult:
    """Execute unified unshard/reorder + token-align."""

    # Per-side: unshard + reorder -> dict[step, tensor]
    step_tensors_x, x_warns = _execute_side_plans(
        tensors=tensors_pair.x, step_plans=plan.per_step_plans.x
    )
    step_tensors_y, y_warns = _execute_side_plans(
        tensors=tensors_pair.y, step_plans=plan.per_step_plans.y
    )
    all_warnings: list[AlignWarning] = x_warns + y_warns

    if not step_tensors_x or not step_tensors_y:
        failed_side_xy: str = "x" if not step_tensors_x else "y"
        return AlignerResult(
            tensors=None, warnings=all_warnings, failed_side_xy=failed_side_xy
        )

    # Cross-side: token alignment (or direct extraction for single-step)
    if plan.token_aligner_plan is not None:
        combined: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan.token_aligner_plan,
            tensor_of_step_pair=Pair(x=step_tensors_x, y=step_tensors_y),
        )
    else:
        assert len(step_tensors_x) == 1 and len(step_tensors_y) == 1
        combined = Pair(
            x=list(step_tensors_x.values())[0],
            y=list(step_tensors_y.values())[0],
        )

    return AlignerResult(
        tensors=combined, warnings=all_warnings, failed_side_xy=None
    )


def _execute_side_plans(
    tensors: list[torch.Tensor],
    step_plans: list[AlignerPerStepPlan],
) -> tuple[dict[int, torch.Tensor], list[AlignWarning]]:
    """Execute per-step unshard + reorder for one side."""
    result: dict[int, torch.Tensor] = {}
    all_warnings: list[AlignWarning] = []

    for step_plan in step_plans:
        step_tensors: list[torch.Tensor] = [tensors[i] for i in step_plan.input_indices]
        tensor, warnings = _execute_step_plans(
            tensors=step_tensors, plans=step_plan.sub_plans
        )
        all_warnings.extend(warnings)
        if tensor is not None:
            result[step_plan.step] = tensor

    return result, all_warnings


def _execute_step_plans(
    tensors: list[torch.Tensor],
    plans: list[AlignerPerStepSubPlan],
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
    plan: AlignerPerStepSubPlan,
) -> tuple[list[torch.Tensor], list[AlignWarning]]:
    if isinstance(plan, UnsharderPlan):
        return execute_unsharder_plan(plan, tensors)
    elif isinstance(plan, ReordererPlan):
        return execute_reorderer_plan(plan, tensors), []
    else:
        raise NotImplementedError(f"Unknown {plan=}")
