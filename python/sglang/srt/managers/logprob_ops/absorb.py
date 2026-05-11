from __future__ import annotations

from typing import Any, List, Optional

from sglang.srt.managers.io_struct import BatchStrOutput
from sglang.srt.managers.logprob_ops.fill import fill_meta_info
from sglang.srt.managers.request_state import ReqState


def absorb_recv(
    meta_info: dict,
    state: ReqState,
    *,
    top_logprobs_num: int,
    token_ids_logprob: Optional[List[int]],
    return_text_in_logprobs: bool,
    recv_obj: BatchStrOutput,
    recv_obj_index: int,
    tokenizer: Optional[Any],
) -> None:
    if recv_obj.input_token_logprobs_val is None:
        return

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

    fill_meta_info(
        meta_info,
        state,
        top_logprobs_num=state.obj.top_logprobs_num,
        token_ids_logprob=state.obj.token_ids_logprob,
        return_text_in_logprobs=return_text_in_logprobs,
        tokenizer=tokenizer,
    )
