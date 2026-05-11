from __future__ import annotations

from typing import Any, List, Optional

from sglang.srt.managers.logprob_ops.detokenize import (
    _detokenize_logprob_tokens,
    _detokenize_top_logprobs_tokens,
)
from sglang.srt.managers.request_state import ReqState


def fill_meta_info(
    meta_info: dict,
    state: ReqState,
    *,
    top_logprobs_num: int,
    token_ids_logprob: Optional[List[int]],
    return_text_in_logprobs: bool,
    tokenizer: Optional[Any],
) -> None:
    # 1. Handle regular logprobs
    if len(state.input_token_logprobs_val) > len(state.input_token_logprobs):
        state.input_token_logprobs.extend(
            _detokenize_logprob_tokens(
                state.input_token_logprobs_val[len(state.input_token_logprobs) :],
                state.input_token_logprobs_idx[len(state.input_token_logprobs) :],
                decode_to_text=return_text_in_logprobs,
                tokenizer=tokenizer,
            )
        )

    if len(state.output_token_logprobs_val) > len(state.output_token_logprobs):
        state.output_token_logprobs.extend(
            _detokenize_logprob_tokens(
                state.output_token_logprobs_val[len(state.output_token_logprobs) :],
                state.output_token_logprobs_idx[len(state.output_token_logprobs) :],
                decode_to_text=return_text_in_logprobs,
                tokenizer=tokenizer,
            )
        )

    meta_info["input_token_logprobs"] = state.input_token_logprobs
    meta_info["output_token_logprobs"] = state.output_token_logprobs
    meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs)

    # 2. Handle top logprobs
    if top_logprobs_num > 0:
        if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):
            state.input_top_logprobs.extend(
                _detokenize_top_logprobs_tokens(
                    state.input_top_logprobs_val[len(state.input_top_logprobs) :],
                    state.input_top_logprobs_idx[len(state.input_top_logprobs) :],
                    decode_to_text=return_text_in_logprobs,
                    tokenizer=tokenizer,
                )
            )
        if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
            state.output_top_logprobs.extend(
                _detokenize_top_logprobs_tokens(
                    state.output_top_logprobs_val[len(state.output_top_logprobs) :],
                    state.output_top_logprobs_idx[len(state.output_top_logprobs) :],
                    decode_to_text=return_text_in_logprobs,
                    tokenizer=tokenizer,
                )
            )

        meta_info["input_top_logprobs"] = state.input_top_logprobs
        meta_info["output_top_logprobs"] = state.output_top_logprobs

    # 3. Handle token_ids_logprob
    if token_ids_logprob is not None:
        if len(state.input_token_ids_logprobs_val) > len(
            state.input_token_ids_logprobs
        ):
            state.input_token_ids_logprobs.extend(
                _detokenize_top_logprobs_tokens(
                    state.input_token_ids_logprobs_val[
                        len(state.input_token_ids_logprobs) :
                    ],
                    state.input_token_ids_logprobs_idx[
                        len(state.input_token_ids_logprobs) :
                    ],
                    decode_to_text=return_text_in_logprobs,
                    tokenizer=tokenizer,
                )
            )
        if len(state.output_token_ids_logprobs_val) > len(
            state.output_token_ids_logprobs
        ):
            state.output_token_ids_logprobs.extend(
                _detokenize_top_logprobs_tokens(
                    state.output_token_ids_logprobs_val[
                        len(state.output_token_ids_logprobs) :
                    ],
                    state.output_token_ids_logprobs_idx[
                        len(state.output_token_ids_logprobs) :
                    ],
                    decode_to_text=return_text_in_logprobs,
                    tokenizer=tokenizer,
                )
            )

        meta_info["input_token_ids_logprobs"] = state.input_token_ids_logprobs
        meta_info["output_token_ids_logprobs"] = state.output_token_ids_logprobs
