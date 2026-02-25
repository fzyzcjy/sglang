from __future__ import annotations

from dataclasses import dataclass, field

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    ExternalSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
    TokenAlignerStepAux,
)


@dataclass
class _SeqInfoAccumulator:
    """Mutable accumulator for building TokenAlignerSeqInfo without per-step validation."""

    input_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)

    def extend(
        self,
        *,
        input_ids: list[int],
        positions: list[int],
        steps: list[int],
        indices: list[int],
    ) -> None:
        self.input_ids.extend(input_ids)
        self.positions.extend(positions)
        self.steps.extend(steps)
        self.indices.extend(indices)

    def build(self) -> TokenAlignerSeqInfo:
        return TokenAlignerSeqInfo(
            input_ids=self.input_ids,
            positions=self.positions,
            steps=self.steps,
            indices=self.indices,
        )


def build_seqs_info(global_aux: TokenAlignerGlobalAux) -> TokenAlignerSeqsInfo:
    """Build sequence info for one side from its auxiliary tensors."""
    return TokenAlignerSeqsInfo(
        sequences=_build_token_aligner_seq_infos(global_aux),
        layout=global_aux.layout,
    )


def _build_token_aligner_seq_infos(
    global_aux: TokenAlignerGlobalAux,
) -> dict[int, TokenAlignerSeqInfo]:
    """Build token index for any framework/layout using seq_ids for identity tracking."""
    external_to_internal_seq_id: dict[ExternalSeqId, int] = {}
    next_internal_id: int = 0
    accum: dict[int, _SeqInfoAccumulator] = {}

    for step in sorted(global_aux.step_auxs.keys()):
        aux: TokenAlignerStepAux = global_aux.step_auxs[step]

        offset: int = 0
        for seq_index, seq_len in enumerate(aux.seq_lens):
            external_seq_id: ExternalSeqId = aux.seq_ids[seq_index]

            if external_seq_id not in external_to_internal_seq_id:
                external_to_internal_seq_id[external_seq_id] = next_internal_id
                accum[next_internal_id] = _SeqInfoAccumulator()
                next_internal_id += 1

            internal_id: int = external_to_internal_seq_id[external_seq_id]

            accum[internal_id].extend(
                input_ids=aux.input_ids[offset : offset + seq_len],
                positions=aux.positions[offset : offset + seq_len],
                steps=[step] * seq_len,
                indices=list(range(offset, offset + seq_len)),
            )

            offset += seq_len

    return {seq_id: acc.build() for seq_id, acc in accum.items()}
