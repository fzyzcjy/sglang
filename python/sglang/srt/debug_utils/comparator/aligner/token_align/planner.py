from __future__ import annotations

from collections import defaultdict

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    AuxTensorsForStep,
    ExternalSeqId,
    SideAux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    AlignmentPlan,
    SeqInfo,
    SeqsInfo,
)
from sglang.srt.debug_utils.comparator.utils import Pair

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_seqs_info(side_aux: SideAux) -> SeqsInfo:
    """Build sequence info for one side from its auxiliary tensors."""
    return SeqsInfo(
        sequences=_build_token_index(side_aux),
        layout=side_aux.layout,
    )


def compute_alignment_plan(
    indices: Pair[SeqsInfo],
) -> AlignmentPlan:
    """Compute a token alignment plan from two side token indices."""
    matched_pairs: list[tuple[int, int]] = _match_sequences(
        seqs=Pair(x=indices.x.sequences, y=indices.y.sequences)
    )

    steps_a: list[int] = []
    indices_a: list[int] = []
    steps_b: list[int] = []
    indices_b: list[int] = []

    for seq_id_a, seq_id_b in matched_pairs:
        rec_a: SeqInfo = indices.x.sequences[seq_id_a]
        rec_b: SeqInfo = indices.y.sequences[seq_id_b]

        pos_to_a: dict[int, int] = {pos: idx for idx, pos in enumerate(rec_a.positions)}
        pos_to_b: dict[int, int] = {pos: idx for idx, pos in enumerate(rec_b.positions)}
        assert len(pos_to_a) == len(rec_a.positions), "duplicate positions in side A"
        assert len(pos_to_b) == len(rec_b.positions), "duplicate positions in side B"

        common_positions: set[int] = set(pos_to_a.keys()) & set(pos_to_b.keys())

        for pos in sorted(common_positions):
            a_idx: int = pos_to_a[pos]
            b_idx: int = pos_to_b[pos]

            a_input_id: int = rec_a.input_ids[a_idx]
            b_input_id: int = rec_b.input_ids[b_idx]
            if a_input_id != b_input_id:
                raise ValueError(
                    f"Sanity check failed: input_id mismatch at position {pos} "
                    f"for seq_id_a={seq_id_a}, seq_id_b={seq_id_b}: "
                    f"{a_input_id} != {b_input_id}"
                )

            steps_a.append(rec_a.steps[a_idx])
            indices_a.append(rec_a.indices[a_idx])
            steps_b.append(rec_b.steps[b_idx])
            indices_b.append(rec_b.indices[b_idx])

    return AlignmentPlan(
        match_steps=Pair(x=tuple(steps_a), y=tuple(steps_b)),
        match_indices=Pair(x=tuple(indices_a), y=tuple(indices_b)),
    )


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


class _SeqAccumulator:
    """Mutable accumulator for building SeqInfo incrementally."""

    __slots__ = ("input_ids", "positions", "steps", "indices")

    def __init__(self) -> None:
        self.input_ids: list[int] = []
        self.positions: list[int] = []
        self.steps: list[int] = []
        self.indices: list[int] = []


def _build_token_index(side_aux: SideAux) -> dict[int, SeqInfo]:
    """Build token index for any framework/layout using seq_ids for identity tracking."""
    external_to_internal: dict[ExternalSeqId, int] = {}
    next_internal_id: int = 0
    accum: dict[int, _SeqAccumulator] = {}

    for step in sorted(side_aux.steps.keys()):
        aux: AuxTensorsForStep = side_aux.steps[step]

        input_ids_flat: list[int] = aux.input_ids.flatten().tolist()
        positions_flat: list[int] = aux.positions.flatten().tolist()
        seq_lens_list: list[int] = aux.seq_lens.tolist()

        offset: int = 0
        for seq_index, slen in enumerate(seq_lens_list):
            ext_id: ExternalSeqId = aux.seq_ids[seq_index]

            if ext_id not in external_to_internal:
                external_to_internal[ext_id] = next_internal_id
                accum[next_internal_id] = _SeqAccumulator()
                next_internal_id += 1

            internal_id: int = external_to_internal[ext_id]
            acc: _SeqAccumulator = accum[internal_id]

            for j in range(slen):
                acc.input_ids.append(input_ids_flat[offset + j])
                acc.positions.append(positions_flat[offset + j])
                acc.steps.append(step)
                acc.indices.append(offset + j)

            offset += slen

    return {
        sid: SeqInfo(
            input_ids=acc.input_ids,
            positions=acc.positions,
            steps=acc.steps,
            indices=acc.indices,
        )
        for sid, acc in accum.items()
    }


# ---------------------------------------------------------------------------
# Sequence matching
# ---------------------------------------------------------------------------


def _match_sequences(
    seqs: Pair[dict[int, SeqInfo]],
) -> list[tuple[int, int]]:
    """Two-pass sequence matching: exact then prefix."""
    matched: list[tuple[int, int]] = []
    unmatched_a: set[int] = set(seqs.x.keys())
    unmatched_b: set[int] = set(seqs.y.keys())

    b_lookup: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for seq_id, rec in seqs.y.items():
        b_lookup[tuple(rec.input_ids)].append(seq_id)

    for seq_id_a in sorted(seqs.x.keys()):
        if seq_id_a not in unmatched_a:
            continue
        ids_a_key: tuple[int, ...] = tuple(seqs.x[seq_id_a].input_ids)
        candidates: list[int] = b_lookup.get(ids_a_key, [])
        for candidate in candidates:
            if candidate in unmatched_b:
                matched.append((seq_id_a, candidate))
                unmatched_a.discard(seq_id_a)
                unmatched_b.discard(candidate)
                break

    remaining_a: list[int] = sorted(
        unmatched_a, key=lambda s: len(seqs.x[s].input_ids), reverse=True
    )
    remaining_b_by_len: list[tuple[int, list[int]]] = sorted(
        [(s, seqs.y[s].input_ids) for s in unmatched_b],
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for seq_id_a in remaining_a:
        ids_a: list[int] = seqs.x[seq_id_a].input_ids
        best_match: int | None = None
        best_len: int = 0

        for seq_id_b, ids_b in remaining_b_by_len:
            if seq_id_b not in unmatched_b:
                continue

            shorter: list[int] = ids_a if len(ids_a) <= len(ids_b) else ids_b
            longer: list[int] = ids_b if len(ids_a) <= len(ids_b) else ids_a

            if longer[: len(shorter)] == shorter and len(shorter) > best_len:
                best_match = seq_id_b
                best_len = len(shorter)

        if best_match is not None:
            matched.append((seq_id_a, best_match))
            unmatched_a.discard(seq_id_a)
            unmatched_b.discard(best_match)

    return matched
