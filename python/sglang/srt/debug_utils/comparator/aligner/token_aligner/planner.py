from __future__ import annotations

from collections import defaultdict

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


# ---------------------------------------------------------------------------
# Sequence matching
# ---------------------------------------------------------------------------


def _match_sequences(
    seqs: Pair[dict[int, TokenAlignerSeqInfo]],
) -> list[tuple[int, int]]:
    """Two-pass sequence matching: exact then prefix."""
    matched_seq_id_pairs: list[tuple[int, int]] = []
    unmatched_seq_ids: Pair[set[int]] = Pair(
        x=set(seqs.x.keys()), y=set(seqs.y.keys())
    )

    y_lookup: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for seq_id, rec in seqs.y.items():
        y_lookup[tuple(rec.input_ids)].append(seq_id)

    for seq_id_x in sorted(seqs.x.keys()):
        if seq_id_x not in unmatched_seq_ids.x:
            continue
        ids_x_key: tuple[int, ...] = tuple(seqs.x[seq_id_x].input_ids)
        candidates: list[int] = y_lookup.get(ids_x_key, [])
        for candidate in candidates:
            if candidate in unmatched_seq_ids.y:
                matched_seq_id_pairs.append((seq_id_x, candidate))
                unmatched_seq_ids.x.discard(seq_id_x)
                unmatched_seq_ids.y.discard(candidate)
                break

    remaining_x: list[int] = sorted(
        unmatched_seq_ids.x,
        key=lambda s: len(seqs.x[s].input_ids),
        reverse=True,
    )
    remaining_y_by_len: list[tuple[int, list[int]]] = sorted(
        [(s, seqs.y[s].input_ids) for s in unmatched_seq_ids.y],
        key=lambda t: len(t[1]),
        reverse=True,
    )

    for seq_id_x in remaining_x:
        ids_x: list[int] = seqs.x[seq_id_x].input_ids
        best_match: int | None = None
        best_len: int = 0

        for seq_id_y, ids_y in remaining_y_by_len:
            if seq_id_y not in unmatched_seq_ids.y:
                continue

            shorter: list[int] = ids_x if len(ids_x) <= len(ids_y) else ids_y
            longer: list[int] = ids_y if len(ids_x) <= len(ids_y) else ids_x

            if longer[: len(shorter)] == shorter and len(shorter) > best_len:
                best_match = seq_id_y
                best_len = len(shorter)

        if best_match is not None:
            matched_seq_id_pairs.append((seq_id_x, best_match))
            unmatched_seq_ids.x.discard(seq_id_x)
            unmatched_seq_ids.y.discard(best_match)

    return matched_seq_id_pairs
