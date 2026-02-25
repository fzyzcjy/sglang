from __future__ import annotations

from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase


class SeqInfo(_FrozenBase):
    """Information for a sequence, containing information to locate all the tokens inside the sequence."""

    input_ids: list[int]
    positions: list[int]
    steps: list[int]
    indices: list[int]


class SeqsInfo(_FrozenBase):
    """All sequences for one side across all steps."""

    sequences: dict[int, SeqInfo]
    layout: str


class AlignmentPlan(_FrozenBase):
    """Token alignment plan.

    match_steps.x[i] + match_indices.x[i] and match_steps.y[i] + match_indices.y[i]
    correspond to the same logical token.
    """

    match_steps: Pair[tuple[int, ...]]
    match_indices: Pair[tuple[int, ...]]
