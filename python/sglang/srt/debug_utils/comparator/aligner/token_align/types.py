from __future__ import annotations

from sglang.srt.debug_utils.comparator.utils import _FrozenBase


class SequenceRecord(_FrozenBase):
    """All tokens of a single sequence across all steps."""

    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    steps: tuple[int, ...]
    indices: tuple[int, ...]


class SideTokenIndex(_FrozenBase):
    """Global token index for one side across all steps."""

    sequences: dict[int, SequenceRecord]
    framework: str
    layout: str


class SeqMatchInfo(_FrozenBase):
    """Statistics for one pair of matched sequences."""

    seq_id_a: int
    seq_id_b: int
    num_tokens_a: int
    num_tokens_b: int
    num_matched: int


class SideInfo(_FrozenBase):
    framework: str
    layout: str
    num_sequences: int
    num_tokens: int
    num_steps: int


class AlignmentSummary(_FrozenBase):
    side_a: SideInfo
    side_b: SideInfo
    sequence_matches: tuple[SeqMatchInfo, ...]
    unmatched_seq_ids_a: tuple[int, ...]
    unmatched_seq_ids_b: tuple[int, ...]
    num_matched_tokens: int


class AlignmentPlan(_FrozenBase):
    """Token alignment plan.

    match_steps_a[i] + match_indices_a[i] and match_steps_b[i] + match_indices_b[i]
    correspond to the same logical token.
    """

    match_steps_a: tuple[int, ...]
    match_indices_a: tuple[int, ...]
    match_steps_b: tuple[int, ...]
    match_indices_b: tuple[int, ...]

    layout_a: str
    layout_b: str

    summary: AlignmentSummary
