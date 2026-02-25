from __future__ import annotations

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    ExternalSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
    TokenAlignerStepAux,
)


def build_seqs_info(global_aux: TokenAlignerGlobalAux) -> TokenAlignerSeqsInfo:
    """Build sequence info for one side from its auxiliary tensors."""
    return TokenAlignerSeqsInfo(
        sequences=_build_token_index(global_aux),
        layout=global_aux.layout,
    )


class _SeqAccumulator:
    """Mutable accumulator for building SeqInfo incrementally."""

    __slots__ = ("input_ids", "positions", "steps", "indices")

    def __init__(self) -> None:
        self.input_ids: list[int] = []
        self.positions: list[int] = []
        self.steps: list[int] = []
        self.indices: list[int] = []


def _build_token_index(
    global_aux: TokenAlignerGlobalAux,
) -> dict[int, TokenAlignerSeqInfo]:
    """Build token index for any framework/layout using seq_ids for identity tracking."""
    external_to_internal: dict[ExternalSeqId, int] = {}
    next_internal_id: int = 0
    accum: dict[int, _SeqAccumulator] = {}

    for step in sorted(global_aux.step_auxs.keys()):
        aux: TokenAlignerStepAux = global_aux.step_auxs[step]

        input_ids_flat: list[int] = aux.input_ids.flatten().tolist()
        positions_flat: list[int] = aux.positions.flatten().tolist()
        seq_lens_list: list[int] = aux.seq_lens.tolist()

        offset: int = 0
        for seq_index, slen in enumerate(seq_lens_list):
            ext_seq_id: ExternalSeqId = aux.seq_ids[seq_index]

            if ext_seq_id not in external_to_internal:
                external_to_internal[ext_seq_id] = next_internal_id
                accum[next_internal_id] = _SeqAccumulator()
                next_internal_id += 1

            internal_id: int = external_to_internal[ext_seq_id]
            acc: _SeqAccumulator = accum[internal_id]

            for j in range(slen):
                acc.input_ids.append(input_ids_flat[offset + j])
                acc.positions.append(positions_flat[offset + j])
                acc.steps.append(step)
                acc.indices.append(offset + j)

            offset += slen

    return {
        sid: TokenAlignerSeqInfo(
            input_ids=acc.input_ids,
            positions=acc.positions,
            steps=acc.steps,
            indices=acc.indices,
        )
        for sid, acc in accum.items()
    }
