from __future__ import annotations

from typing import List, Optional

from sglang.srt.managers.io_struct import BatchStrOutput
from sglang.srt.managers.request_state import ReqState


def accumulate_recv(
    state: ReqState,
    *,
    recv_obj: BatchStrOutput,
    recv_obj_index: int,
    top_logprobs_num: int,
    token_ids_logprob: Optional[List[int]],
) -> None:
    """Extend per-state logprob accumulators with this recv_obj chunk.

    Caller MUST gate on ``recv_obj.input_token_logprobs_val is not None``
    before calling — this function assumes scheduler logprob payload is
    present (it would otherwise IndexError on the unconditional output
    extends below).
    """
    if (
        len(recv_obj.input_token_logprobs_val) > 0
        and recv_obj.input_token_logprobs_val[recv_obj_index] is not None
    ):
        state.input_token_logprobs_val.extend(
            recv_obj.input_token_logprobs_val[recv_obj_index]
        )
        state.input_token_logprobs_idx.extend(
            recv_obj.input_token_logprobs_idx[recv_obj_index]
        )
    state.output_token_logprobs_val.extend(
        recv_obj.output_token_logprobs_val[recv_obj_index]
    )
    state.output_token_logprobs_idx.extend(
        recv_obj.output_token_logprobs_idx[recv_obj_index]
    )

    if top_logprobs_num > 0:
        if len(recv_obj.input_top_logprobs_val) > 0:
            state.input_top_logprobs_val.extend(
                recv_obj.input_top_logprobs_val[recv_obj_index]
            )
            state.input_top_logprobs_idx.extend(
                recv_obj.input_top_logprobs_idx[recv_obj_index]
            )
        state.output_top_logprobs_val.extend(
            recv_obj.output_top_logprobs_val[recv_obj_index]
        )
        state.output_top_logprobs_idx.extend(
            recv_obj.output_top_logprobs_idx[recv_obj_index]
        )

    if token_ids_logprob is not None:
        if len(recv_obj.input_token_ids_logprobs_val) > 0:
            state.input_token_ids_logprobs_val.extend(
                recv_obj.input_token_ids_logprobs_val[recv_obj_index]
            )
            state.input_token_ids_logprobs_idx.extend(
                recv_obj.input_token_ids_logprobs_idx[recv_obj_index]
            )
        state.output_token_ids_logprobs_val.extend(
            recv_obj.output_token_ids_logprobs_val[recv_obj_index]
        )
        state.output_token_ids_logprobs_idx.extend(
            recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
        )
