"""Shared reference-capture and near-tie alignment helpers for DSpark losslessness.

The strict losslessness check compares a DSpark greedy completion against a
non-spec greedy reference token-for-token. DSpark does not support
``return_logprob`` (the server rejects it), so the spec server only returns
text. The non-spec reference, however, does support logprobs, and a greedy
completion is exactly the argmax sequence, so we capture from the reference:

  * the detokenized text PIECE of every chosen (argmax) token, and
  * the top-2 logprob gap at every position.

Re-tokenizing the spec text to recover token ids is unreliable -- the tokenizer
re-splits the prompt/output boundary and shifts every position. Instead we walk
the reference token pieces, accumulate the running reference text, and find the
first reference token boundary at which the spec output stops sharing the
reference prefix. That boundary is the per-token divergence position, and the
reference top-2 gap there classifies the divergence:

  * gap <= eps  -> near-tie: the verify-path forward (a gamma+1 token window
    through the verify-mask attention kernel) and the non-spec decode forward
    (one token through the decode kernel) are different kernels whose logits
    differ by accumulated fp rounding; at a near-tie that rounding flips the
    argmax. This is benign and unavoidable (the non-spec reference is itself not
    bit-reproducible once a long greedy run accumulates such near-ties), the
    same effect EAGLE/DFlash live with.
  * gap > eps   -> confident divergence: the argmax differed at a token the
    reference was confident about. A lossless accept path cannot produce that,
    so it is a real losslessness bug and the caller must FAIL.
"""

from typing import List, NamedTuple, Optional

import requests


class ReferenceCapture(NamedTuple):
    """A non-spec greedy reference completion plus per-token logprob structure."""

    text: str
    pieces: List[str]
    top2_gaps: List[float]


class Divergence(NamedTuple):
    """First per-token divergence of a spec output from the reference."""

    token_index: int
    top2_gap: float


def greedy_request(url: str, prompt: str, max_new_tokens: int) -> str:
    """Send a greedy (temperature 0) generation request; return the output text."""
    resp = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
    )
    resp.raise_for_status()
    return resp.json()["text"]


def capture_reference(url: str, prompt: str, max_new_tokens: int) -> ReferenceCapture:
    """Capture a non-spec greedy completion with per-token text pieces and top-2 gaps."""
    resp = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
            "top_logprobs_num": 2,
            "return_text_in_logprobs": True,
        },
    )
    resp.raise_for_status()
    meta = resp.json()["meta_info"]

    pieces = [entry[2] for entry in meta["output_token_logprobs"]]
    top2_gaps: List[float] = []
    for entry in meta["output_top_logprobs"]:
        if len(entry) >= 2:
            top2_gaps.append(abs(entry[0][0] - entry[1][0]))
        else:
            top2_gaps.append(float("inf"))

    return ReferenceCapture(
        text=resp.json()["text"], pieces=pieces, top2_gaps=top2_gaps
    )


def first_token_divergence(
    reference: ReferenceCapture, spec_text: str
) -> Optional[Divergence]:
    """Return the first reference token boundary where spec_text leaves the prefix.

    Returns None when the spec text agrees with the reference along every
    reference token boundary (spec is a prefix of, equal to, or an extension of
    the reference text up to the captured length).
    """
    accumulated = ""
    for index, piece in enumerate(reference.pieces):
        candidate = accumulated + piece
        if spec_text.startswith(candidate):
            accumulated = candidate
            continue
        if candidate.startswith(spec_text):
            return None
        gap = (
            reference.top2_gaps[index]
            if index < len(reference.top2_gaps)
            else float("inf")
        )
        return Divergence(token_index=index, top2_gap=gap)
    return None
