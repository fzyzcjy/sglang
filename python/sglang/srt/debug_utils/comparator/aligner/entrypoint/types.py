from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.utils import Pair

StepPlan = Union[UnsharderPlan, ReordererPlan]


@dataclass(frozen=True)
class StepGroupPlan:
    """Unshard + reorder plan for a single step."""

    step: int
    input_indices: list[int]
    unshard_reorder: list[StepPlan]


@dataclass(frozen=True)
class AlignerPlan:
    """Unified plan: per-step unshard/reorder for both sides + cross-side token alignment."""

    side_plans: Pair[list[StepGroupPlan]]
    token_aligner_plan: Optional[TokenAlignerPlan]
