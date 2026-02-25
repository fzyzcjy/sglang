from __future__ import annotations

from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase


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

    seq_ids: Pair[int]
    num_tokens: Pair[int]
    num_matched: int


class SideInfo(_FrozenBase):
    framework: str
    layout: str
    num_sequences: int
    num_tokens: int
    num_steps: int


class AlignmentSummary(_FrozenBase):
    sides: Pair[SideInfo]
    sequence_matches: tuple[SeqMatchInfo, ...]
    unmatched_seq_ids: Pair[tuple[int, ...]]
    num_matched_tokens: int


class AlignmentPlan(_FrozenBase):
    """Token alignment plan.

    match_steps.x[i] + match_indices.x[i] and match_steps.y[i] + match_indices.y[i]
    correspond to the same logical token.
    """

    match_steps: Pair[tuple[int, ...]]
    match_indices: Pair[tuple[int, ...]]

    layouts: Pair[str]

    summary: AlignmentSummary


def format_alignment_summary(summary: AlignmentSummary) -> str:
    lines: list[str] = [
        "Alignment Summary:",
        f"  Side A: {summary.sides.x.framework} ({summary.sides.x.layout}), "
        f"{summary.sides.x.num_sequences} sequences, "
        f"{summary.sides.x.num_tokens} tokens, "
        f"{summary.sides.x.num_steps} steps",
        f"  Side B: {summary.sides.y.framework} ({summary.sides.y.layout}), "
        f"{summary.sides.y.num_sequences} sequences, "
        f"{summary.sides.y.num_tokens} tokens, "
        f"{summary.sides.y.num_steps} steps",
        f"  Matched: {len(summary.sequence_matches)} sequence pairs, "
        f"{summary.num_matched_tokens} tokens",
    ]

    if summary.unmatched_seq_ids.x:
        lines.append(f"  Unmatched A: {summary.unmatched_seq_ids.x}")
    if summary.unmatched_seq_ids.y:
        lines.append(f"  Unmatched B: {summary.unmatched_seq_ids.y}")

    return "\n".join(lines)
