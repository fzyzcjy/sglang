"""Compare two tensor bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from sglang.srt.debug_utils.comparator.aligner.executor import (
    AlignResult,
    execute_align_plan,
)
from sglang.srt.debug_utils.comparator.aligner.planner import compute_align_plan
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.types import AlignPlan
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import ValueWithMeta

_FAILED_SIDE_MAP: dict[str, str] = {"x": "baseline", "y": "target"}


def compare_bundle_pair(
    *,
    name: str,
    filenames_pair: Pair[list[str]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_plan: Optional[TokenAlignerPlan],
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
    plan: AlignPlan = compute_align_plan(
        metas_pair=metas_pair, token_aligner_plan=token_aligner_plan
    )

    # 3. Execute (tensor + plan only)
    tensors_pair: Pair[list[torch.Tensor]] = Pair(
        x=[it.value for it in valid_pair.x],
        y=[it.value for it in valid_pair.y],
    )
    align_result: AlignResult = execute_align_plan(
        tensors_pair=tensors_pair, plan=plan
    )

    if align_result.tensors is None:
        assert align_result.failed_side_xy is not None
        side_name: str = _FAILED_SIDE_MAP[align_result.failed_side_xy]
        reason = f"{side_name}_load_failed"
        return SkipRecord(
            name=name, reason=reason, align_warnings=align_result.warnings
        )

    # 4. Compare
    info = compare_tensor_pair(
        x_baseline=align_result.tensors.x,
        x_target=align_result.tensors.y,
        name=name,
        diff_threshold=diff_threshold,
    )
    return ComparisonRecord(**info.model_dump(), align_warnings=align_result.warnings)


def _load_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [ValueWithMeta.load(base_path / f) for f in filenames]
