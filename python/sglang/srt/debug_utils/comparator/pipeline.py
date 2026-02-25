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
from sglang.srt.debug_utils.comparator.output_types import AlignWarning
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows

Plan = Union[UnshardPlan, ReorderPlan]


def load_and_unshard_files(
    *, filenames: list[str], base_path: Path
) -> tuple[Optional[torch.Tensor], list[AlignWarning]]:
    """Load tensor files and unshard them into a single tensor."""
    if not filenames:
        return None, []

    loaded: list[ValueWithMeta] = _load_tensors(filenames, base_path)
    plans: list[Plan] = _compute_plans_for_group([item.meta for item in loaded])
    tensors: list[torch.Tensor] = _extract_tensors(loaded)
    if not tensors:
        return None, []

    return _execute_plans(tensors, plans)


def _load_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [ValueWithMeta.load(base_path / f) for f in filenames]


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
) -> list[torch.Tensor]:
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


def concat_steps(tensors: dict[int, torch.Tensor]) -> torch.Tensor:
    """Concat all step tensors into a single tensor (sorted by step key)."""
    sorted_tensors: list[torch.Tensor] = [tensors[s] for s in sorted(tensors)]
    if sorted_tensors[0].ndim == 0:
        return torch.stack(sorted_tensors)
    return torch.cat(sorted_tensors, dim=0)


def _load_and_unshard_for_step(
    *, name: str, step: int, df: pl.DataFrame, dump_path: Path
) -> tuple[Optional[torch.Tensor], list[AlignWarning]]:
    """Load all rank files for (name, step), unshard into a single tensor."""
    rows: list[dict] = filter_rows(df, conditions={"name": name, "step": step})
    filenames: list[str] = [r["filename"] for r in rows]
    return load_and_unshard_files(filenames=filenames, base_path=dump_path)


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
        tensor, warnings = _load_and_unshard_for_step(
            name=name, step=step, df=df, dump_path=dump_path
        )
        all_warnings.extend(warnings)
        if tensor is not None:
            result[step] = tensor

    return result, all_warnings
