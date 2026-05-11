from __future__ import annotations

from typing import Any, List, Optional, Tuple


def detokenize_logprob_tokens(
    token_logprobs_val: List[float],
    token_logprobs_idx: List[int],
    *,
    decode_to_text: bool,
    tokenizer: Optional[Any],
) -> List[Tuple[float, int, Optional[str]]]:
    if not decode_to_text:
        return [
            (logprob, token_id, None)
            for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
        ]
    else:
        assert tokenizer is not None
        # In transformers v5, batch_decode([1, 2, 3]) concatenates all tokens
        # into one string. Wrap each ID in its own list so they decode separately.
        token_texts = tokenizer.batch_decode([[idx] for idx in token_logprobs_idx])
        return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))


def detokenize_top_logprobs_tokens(
    token_logprobs_val: List[List[float]],
    token_logprobs_idx: List[List[int]],
    *,
    decode_to_text: bool,
    tokenizer: Optional[Any],
) -> List[Optional[List[Tuple[float, int, Optional[str]]]]]:
    # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
    # We should batch all top-k tokens in all positions.
    ret = []
    for i in range(len(token_logprobs_val)):
        if token_logprobs_val[i]:
            ret.append(
                detokenize_logprob_tokens(
                    token_logprobs_val[i],
                    token_logprobs_idx[i],
                    decode_to_text=decode_to_text,
                    tokenizer=tokenizer,
                )
            )
        else:
            ret.append(None)
    return ret
