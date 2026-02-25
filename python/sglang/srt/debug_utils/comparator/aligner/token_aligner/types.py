from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Union

from pydantic import model_validator

from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase


class SGLangSeqId(NamedTuple):
    rid: str


class PositionalSeqId(NamedTuple):
    step: int
    seq_index: int


ExternalSeqId = Union[SGLangSeqId, PositionalSeqId]


@dataclass(frozen=True)
class TokenAlignerStepAux:
    """Normalized auxiliary tensors for a single step (framework-agnostic)."""

    input_ids: list[int]  # [num_tokens]
    positions: list[int]  # [num_tokens]
    seq_lens: list[int]  # [num_seqs]
    seq_ids: list[ExternalSeqId]  # [num_seqs] — sequence identity


@dataclass(frozen=True)
class TokenAlignerGlobalAux:
    """Auxiliary tensors for one side across all steps + side-level metadata."""

    step_auxs: dict[int, TokenAlignerStepAux]
    framework: str  # "sglang" | "megatron"
    layout: str  # "thd"


class TokenAlignerSeqInfo(_FrozenBase):
    """Information for a sequence, containing information to locate all the tokens inside the sequence."""

    # All these fields are of shape (num_tokens_in_seq,)
    input_ids: list[int]
    positions: list[int]
    steps: list[int]
    indices: list[int]

    @model_validator(mode="after")
    def _validate_fields(self) -> TokenAlignerSeqInfo:
        n: int = len(self.input_ids)
        assert (
            len(self.positions) == n
        ), f"positions length {len(self.positions)} != {n}"
        assert len(self.steps) == n, f"steps length {len(self.steps)} != {n}"
        assert len(self.indices) == n, f"indices length {len(self.indices)} != {n}"
        assert self.positions == list(
            range(n)
        ), f"positions must be [0, 1, ..., {n - 1}], got {self.positions}"
        return self


class TokenAlignerSeqsInfo(_FrozenBase):
    """All sequences for one side across all steps."""

    sequences: dict[int, TokenAlignerSeqInfo]
    layout: str


class TokenAlignerPlan(_FrozenBase):
    """Token alignment plan.

    (match_steps.x[i], match_indices.x[i]) and (match_steps.y[i], match_indices.y[i])
    correspond to the same logical token.
    """

    match_steps: Pair[list[int]]
    match_indices: Pair[list[int]]
