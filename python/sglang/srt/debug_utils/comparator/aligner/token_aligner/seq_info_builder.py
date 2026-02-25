from __future__ import annotations

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    ExternalSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
    TokenAlignerStepAux,
)

_EMPTY_SEQ_INFO = TokenAlignerSeqInfo(input_ids=[], positions=[], steps=[], indices=[])


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
    accum: dict[int, TokenAlignerSeqInfo] = {}

    for step in sorted(global_aux.step_auxs.keys()):
        aux: TokenAlignerStepAux = global_aux.step_auxs[step]

        input_ids_flat: list[int] = aux.input_ids.tolist()
        positions_flat: list[int] = aux.positions.tolist()
        seq_lens_list: list[int] = aux.seq_lens.tolist()

        offset: int = 0
        for seq_index, seq_len in enumerate(seq_lens_list):
            external_seq_id: ExternalSeqId = aux.seq_ids[seq_index]

            if external_seq_id not in external_to_internal_seq_id:
                external_to_internal_seq_id[external_seq_id] = next_internal_id
                accum[next_internal_id] = _EMPTY_SEQ_INFO
                next_internal_id += 1

            internal_id: int = external_to_internal_seq_id[external_seq_id]

            accum[internal_id] = accum[internal_id] + TokenAlignerSeqInfo(
                input_ids=input_ids_flat[offset : offset + seq_len],
                positions=positions_flat[offset : offset + seq_len],
                steps=[step] * seq_len,
                indices=list(range(offset, offset + seq_len)),
            )

            offset += seq_len

    return accum
