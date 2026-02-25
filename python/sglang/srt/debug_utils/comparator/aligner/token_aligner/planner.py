from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple, Optional

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
)
from sglang.srt.debug_utils.comparator.utils import Pair


def compute_token_aligner_plan(
    seqs_info_pair: Pair[TokenAlignerSeqsInfo],
) -> TokenAlignerPlan:
    """Compute a token alignment plan from two side token seqs_info_pair."""
    matched_pairs: list[tuple[int, int]] = _match_sequences(
        seqs=Pair(x=seqs_info_pair.x.sequences, y=seqs_info_pair.y.sequences)
    )

    steps: Pair[list[int]] = Pair(x=[], y=[])
    indices: Pair[list[int]] = Pair(x=[], y=[])

    for seq_id_x, seq_id_y in matched_pairs:
        rec: Pair[TokenAlignerSeqInfo] = Pair(
            x=seqs_info_pair.x.sequences[seq_id_x],
            y=seqs_info_pair.y.sequences[seq_id_y],
        )

        pos_to: Pair[dict[int, int]] = rec.map(
            lambda r: {pos: idx for idx, pos in enumerate(r.positions)}
        )
        assert len(pos_to.x) == len(rec.x.positions), "duplicate positions in side X"
        assert len(pos_to.y) == len(rec.y.positions), "duplicate positions in side Y"

        common_positions: set[int] = set(pos_to.x.keys()) & set(pos_to.y.keys())

        for pos in sorted(common_positions):
            idx: Pair[int] = Pair(x=pos_to.x[pos], y=pos_to.y[pos])

            input_id: Pair[int] = Pair(
                x=rec.x.input_ids[idx.x],
                y=rec.y.input_ids[idx.y],
            )
            if input_id.x != input_id.y:
                raise ValueError(
                    f"Sanity check failed: input_id mismatch at position {pos} "
                    f"for seq_id_x={seq_id_x}, seq_id_y={seq_id_y}: "
                    f"{input_id.x} != {input_id.y}"
                )

            steps.x.append(rec.x.steps[idx.x])
            indices.x.append(rec.x.indices[idx.x])
            steps.y.append(rec.y.steps[idx.y])
            indices.y.append(rec.y.indices[idx.y])

    return TokenAlignerPlan(
        match_steps=steps.map(tuple),
        match_indices=indices.map(tuple),
    )


# -------------------- Sequence matcher --------------------


def _match_sequences(
    seqs: Pair[dict[int, TokenAlignerSeqInfo]],
) -> list[tuple[int, int]]:
    """For each y (target) sequence, find a matching x (baseline) sequence.

    Two-pass: exact match first, then prefix match for remaining.
    """
    x_lookup: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for seq_id, rec in seqs.x.items():
        x_lookup[tuple(rec.input_ids)].append(seq_id)

    claimed_x_ids: set[int] = set()
    matched_seq_id_pairs: list[tuple[int, int]] = []

    for seq_id_y in sorted(seqs.y.keys()):
        seq_y: TokenAlignerSeqInfo = seqs.y[seq_id_y]

        matched_x: Optional[int] = _find_matching_x_exact(
            seq_y=seq_y, x_lookup=x_lookup, claimed_x_ids=claimed_x_ids
        )
        if matched_x is None:
            matched_x = _find_matching_x_prefix(
                seq_y=seq_y, x_seqs=seqs.x, claimed_x_ids=claimed_x_ids
            )

        if matched_x is not None:
            matched_seq_id_pairs.append((matched_x, seq_id_y))
            claimed_x_ids.add(matched_x)

    return matched_seq_id_pairs


def _find_matching_x_exact(
    *,
    seq_y: TokenAlignerSeqInfo,
    x_lookup: dict[tuple[int, ...], list[int]],
    claimed_x_ids: set[int],
) -> Optional[int]:
    """Find an x sequence with identical input_ids."""
    ids_y_key: tuple[int, ...] = tuple(seq_y.input_ids)
    candidates: list[int] = x_lookup.get(ids_y_key, [])
    for candidate in candidates:
        if candidate not in claimed_x_ids:
            return candidate
    return None


class _PrefixCandidate(NamedTuple):
    seq_id_x: int
    overlap_len: int


def _find_matching_x_prefix(
    *,
    seq_y: TokenAlignerSeqInfo,
    x_seqs: dict[int, TokenAlignerSeqInfo],
    claimed_x_ids: set[int],
) -> Optional[int]:
    """Find the x sequence with the longest prefix relationship to y."""
    ids_y: list[int] = seq_y.input_ids
    candidates: list[_PrefixCandidate] = [
        _PrefixCandidate(seq_id_x=seq_id_x, overlap_len=min(len(seq_x.input_ids), len(ids_y)))
        for seq_id_x, seq_x in x_seqs.items()
        if seq_id_x not in claimed_x_ids and _is_prefix_pair(seq_x.input_ids, ids_y)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.overlap_len).seq_id_x


def _is_prefix_pair(a: list[int], b: list[int]) -> bool:
    """True if a is a prefix of b, or b is a prefix of a."""
    shorter: list[int] = a if len(a) <= len(b) else b
    longer: list[int] = b if len(a) <= len(b) else a
    return longer[: len(shorter)] == shorter
