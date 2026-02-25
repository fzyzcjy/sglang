from __future__ import annotations

import warnings
from collections import defaultdict

import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    AuxTensorsForStep,
    SideAux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    AlignmentPlan,
    AlignmentSummary,
    SeqMatchInfo,
    SequenceRecord,
    SideInfo,
    SideTokenIndex,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_token_index(side_aux: SideAux) -> SideTokenIndex:
    """Build a global token index for one side from its auxiliary tensors."""
    if side_aux.framework == "sglang":
        sequences = _build_sglang_thd_index(side_aux)
    elif side_aux.layout == "bshd":
        sequences = _build_megatron_bshd_index(side_aux)
    else:
        sequences = _build_megatron_thd_index(side_aux)

    return SideTokenIndex(
        sequences=sequences,
        framework=side_aux.framework,
        layout=side_aux.layout,
    )


def compute_alignment_plan(
    index_a: SideTokenIndex,
    index_b: SideTokenIndex,
) -> AlignmentPlan:
    """Compute a token alignment plan from two side token indices."""
    matched_pairs: list[tuple[int, int]] = _match_sequences(
        seqs_a=index_a.sequences, seqs_b=index_b.sequences
    )

    steps_a: list[int] = []
    indices_a: list[int] = []
    steps_b: list[int] = []
    indices_b: list[int] = []
    match_infos: list[SeqMatchInfo] = []

    matched_a_ids: set[int] = set()
    matched_b_ids: set[int] = set()

    for seq_id_a, seq_id_b in matched_pairs:
        matched_a_ids.add(seq_id_a)
        matched_b_ids.add(seq_id_b)

        rec_a: SequenceRecord = index_a.sequences[seq_id_a]
        rec_b: SequenceRecord = index_b.sequences[seq_id_b]

        pos_to_a: dict[int, int] = {pos: idx for idx, pos in enumerate(rec_a.positions)}
        pos_to_b: dict[int, int] = {pos: idx for idx, pos in enumerate(rec_b.positions)}

        common_positions: set[int] = set(pos_to_a.keys()) & set(pos_to_b.keys())
        num_matched: int = 0

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
            num_matched += 1

        match_infos.append(
            SeqMatchInfo(
                seq_id_a=seq_id_a,
                seq_id_b=seq_id_b,
                num_tokens_a=len(rec_a.positions),
                num_tokens_b=len(rec_b.positions),
                num_matched=num_matched,
            )
        )

    unmatched_a: tuple[int, ...] = tuple(
        sorted(set(index_a.sequences.keys()) - matched_a_ids)
    )
    unmatched_b: tuple[int, ...] = tuple(
        sorted(set(index_b.sequences.keys()) - matched_b_ids)
    )

    summary = AlignmentSummary(
        side_a=_make_side_info(index_a),
        side_b=_make_side_info(index_b),
        sequence_matches=tuple(match_infos),
        unmatched_seq_ids_a=unmatched_a,
        unmatched_seq_ids_b=unmatched_b,
        num_matched_tokens=len(steps_a),
    )

    return AlignmentPlan(
        match_steps_a=tuple(steps_a),
        match_indices_a=tuple(indices_a),
        match_steps_b=tuple(steps_b),
        match_indices_b=tuple(indices_b),
        layout_a=index_a.layout,
        layout_b=index_b.layout,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------


def _build_sglang_thd_index(
    side_aux: SideAux,
) -> dict[int, SequenceRecord]:
    """Build token index for SGLang thd layout using req_pool_indices + rids."""
    rpi_to_rid: dict[int, str] = {}
    rpi_to_seq_id: dict[int, int] = {}
    next_seq_id: int = 0

    accum: dict[int, _SeqAccumulator] = {}

    for step in sorted(side_aux.steps.keys()):
        aux: AuxTensorsForStep = side_aux.steps[step]

        if aux.req_pool_indices is None:
            warnings.warn(
                f"req_pool_indices missing at step {step}, cannot build token index"
            )
            continue

        input_ids_flat: list[int] = aux.input_ids.flatten().tolist()
        positions_flat: list[int] = aux.positions.flatten().tolist()
        seq_lens_list: list[int] = aux.seq_lens.tolist()
        rpi_list: list[int] = aux.req_pool_indices.tolist()
        rids_list: list[str] = (
            list(aux.rids) if aux.rids is not None else [str(rpi) for rpi in rpi_list]
        )

        offset: int = 0
        for seg_idx, slen in enumerate(seq_lens_list):
            rpi: int = rpi_list[seg_idx]
            rid: str = rids_list[seg_idx]

            if rpi in rpi_to_rid and rpi_to_rid[rpi] != rid:
                old_seq_id: int = rpi_to_seq_id.pop(rpi)
                del rpi_to_rid[rpi]

            if rpi not in rpi_to_seq_id:
                rpi_to_rid[rpi] = rid
                rpi_to_seq_id[rpi] = next_seq_id
                accum[next_seq_id] = _SeqAccumulator()
                next_seq_id += 1

            seq_id: int = rpi_to_seq_id[rpi]
            acc: _SeqAccumulator = accum[seq_id]

            for j in range(slen):
                acc.input_ids.append(input_ids_flat[offset + j])
                acc.positions.append(positions_flat[offset + j])
                acc.steps.append(step)
                acc.indices.append(offset + j)

            offset += slen

    return {
        seq_id: SequenceRecord(
            input_ids=tuple(acc.input_ids),
            positions=tuple(acc.positions),
            steps=tuple(acc.steps),
            indices=tuple(acc.indices),
        )
        for seq_id, acc in accum.items()
    }


def _build_megatron_bshd_index(
    side_aux: SideAux,
) -> dict[int, SequenceRecord]:
    """Build token index for Megatron bshd layout."""
    accum: dict[int, _SeqAccumulator] = {}
    next_seq_id: int = 0

    for step in sorted(side_aux.steps.keys()):
        aux: AuxTensorsForStep = side_aux.steps[step]

        input_ids_2d: torch.Tensor = aux.input_ids  # [B, S]
        positions_2d: torch.Tensor = aux.positions  # [B, S]
        seq_lens_list: list[int] = aux.seq_lens.tolist()
        batch_size: int = input_ids_2d.shape[0]
        max_seq_len: int = input_ids_2d.shape[1]

        for batch_idx in range(batch_size):
            seq_id: int = next_seq_id + batch_idx
            if seq_id not in accum:
                accum[seq_id] = _SeqAccumulator()
            acc: _SeqAccumulator = accum[seq_id]

            slen: int = seq_lens_list[batch_idx]
            for s in range(slen):
                acc.input_ids.append(int(input_ids_2d[batch_idx, s].item()))
                acc.positions.append(int(positions_2d[batch_idx, s].item()))
                acc.steps.append(step)
                acc.indices.append(batch_idx * max_seq_len + s)

        next_seq_id += batch_size

    return {
        seq_id: SequenceRecord(
            input_ids=tuple(acc.input_ids),
            positions=tuple(acc.positions),
            steps=tuple(acc.steps),
            indices=tuple(acc.indices),
        )
        for seq_id, acc in accum.items()
    }


def _build_megatron_thd_index(
    side_aux: SideAux,
) -> dict[int, SequenceRecord]:
    """Build token index for Megatron thd layout (static batching, segment-based)."""
    accum: dict[int, _SeqAccumulator] = {}
    next_seq_id: int = 0

    for step in sorted(side_aux.steps.keys()):
        aux: AuxTensorsForStep = side_aux.steps[step]

        input_ids_flat: list[int] = aux.input_ids.flatten().tolist()
        positions_flat: list[int] = aux.positions.flatten().tolist()
        seq_lens_list: list[int] = aux.seq_lens.tolist()

        offset: int = 0
        for seg_idx, slen in enumerate(seq_lens_list):
            seq_id: int = next_seq_id + seg_idx
            if seq_id not in accum:
                accum[seq_id] = _SeqAccumulator()
            acc: _SeqAccumulator = accum[seq_id]

            for j in range(slen):
                acc.input_ids.append(input_ids_flat[offset + j])
                acc.positions.append(positions_flat[offset + j])
                acc.steps.append(step)
                acc.indices.append(offset + j)

            offset += slen

        next_seq_id += len(seq_lens_list)

    return {
        seq_id: SequenceRecord(
            input_ids=tuple(acc.input_ids),
            positions=tuple(acc.positions),
            steps=tuple(acc.steps),
            indices=tuple(acc.indices),
        )
        for seq_id, acc in accum.items()
    }


# ---------------------------------------------------------------------------
# Sequence matching
# ---------------------------------------------------------------------------


def _match_sequences(
    *,
    seqs_a: dict[int, SequenceRecord],
    seqs_b: dict[int, SequenceRecord],
) -> list[tuple[int, int]]:
    """Two-pass sequence matching: exact then prefix."""
    matched: list[tuple[int, int]] = []
    unmatched_a: set[int] = set(seqs_a.keys())
    unmatched_b: set[int] = set(seqs_b.keys())

    b_lookup: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for seq_id, rec in seqs_b.items():
        b_lookup[rec.input_ids].append(seq_id)

    for seq_id_a in sorted(seqs_a.keys()):
        if seq_id_a not in unmatched_a:
            continue
        ids_a: tuple[int, ...] = seqs_a[seq_id_a].input_ids
        candidates: list[int] = b_lookup.get(ids_a, [])
        for candidate in candidates:
            if candidate in unmatched_b:
                matched.append((seq_id_a, candidate))
                unmatched_a.discard(seq_id_a)
                unmatched_b.discard(candidate)
                break

    remaining_a: list[int] = sorted(
        unmatched_a, key=lambda s: len(seqs_a[s].input_ids), reverse=True
    )
    remaining_b_by_len: list[tuple[int, tuple[int, ...]]] = sorted(
        [(s, seqs_b[s].input_ids) for s in unmatched_b],
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for seq_id_a in remaining_a:
        ids_a = seqs_a[seq_id_a].input_ids
        best_match: int | None = None
        best_len: int = 0

        for seq_id_b, ids_b in remaining_b_by_len:
            if seq_id_b not in unmatched_b:
                continue

            shorter: tuple[int, ...] = ids_a if len(ids_a) <= len(ids_b) else ids_b
            longer: tuple[int, ...] = ids_b if len(ids_a) <= len(ids_b) else ids_a

            if longer[: len(shorter)] == shorter and len(shorter) > best_len:
                best_match = seq_id_b
                best_len = len(shorter)

        if best_match is not None:
            matched.append((seq_id_a, best_match))
            unmatched_a.discard(seq_id_a)
            unmatched_b.discard(best_match)

    if len(matched) > len(set(m[0] for m in matched)):
        warnings.warn(
            "Ambiguous sequence matching: some sequences matched multiple times"
        )

    return matched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SeqAccumulator:
    """Mutable accumulator for building SequenceRecord incrementally."""

    __slots__ = ("input_ids", "positions", "steps", "indices")

    def __init__(self) -> None:
        self.input_ids: list[int] = []
        self.positions: list[int] = []
        self.steps: list[int] = []
        self.indices: list[int] = []


def _make_side_info(index: SideTokenIndex) -> SideInfo:
    all_steps: set[int] = set()
    total_tokens: int = 0
    for rec in index.sequences.values():
        total_tokens += len(rec.positions)
        all_steps.update(rec.steps)

    return SideInfo(
        framework=index.framework,
        layout=index.layout,
        num_sequences=len(index.sequences),
        num_tokens=total_tokens,
        num_steps=len(all_steps),
    )
