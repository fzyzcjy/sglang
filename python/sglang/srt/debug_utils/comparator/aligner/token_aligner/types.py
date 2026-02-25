from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Union

import torch

from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase


class SGLangSeqId(NamedTuple):
    rid: str


class MegatronSeqId(NamedTuple):
    step: int
    seq_index: int


ExternalSeqId = Union[SGLangSeqId, MegatronSeqId]


@dataclass(frozen=True)
class TokenAlignerStepAux:
    """Normalized auxiliary tensors for a single step (framework-agnostic)."""

    input_ids: torch.Tensor  # [num_tokens] (1D flat)
    positions: torch.Tensor  # [num_tokens] (1D flat)
    seq_lens: torch.Tensor  # [num_seqs]
    seq_ids: list[ExternalSeqId]  # [num_seqs] — sequence identity


@dataclass(frozen=True)
class TokenAlignerGlobalAux:
    """Auxiliary tensors for one side across all steps + side-level metadata."""

    step_auxs: dict[int, TokenAlignerStepAux]
    framework: str  # "sglang" | "megatron"
    layout: str  # "thd"


class TokenAlignerSeqInfo(_FrozenBase):
    """Information for a sequence, containing information to locate all the tokens inside the sequence."""

    input_ids: list[int]
    positions: list[int]
    steps: list[int]
    indices: list[int]

    def __add__(self, other: TokenAlignerSeqInfo) -> TokenAlignerSeqInfo:
        return TokenAlignerSeqInfo(
            input_ids=self.input_ids + other.input_ids,
            positions=self.positions + other.positions,
            steps=self.steps + other.steps,
            indices=self.indices + other.indices,
        )


class TokenAlignerSeqsInfo(_FrozenBase):
    """All sequences for one side across all steps."""

    sequences: dict[int, TokenAlignerSeqInfo]
    layout: str


class TokenAlignerPlan(_FrozenBase):
    """Token alignment plan.

    match_steps.x[i] + match_indices.x[i] and match_steps.y[i] + match_indices.y[i]
    correspond to the same logical token.
    """

    match_steps: Pair[tuple[int, ...]]
    match_indices: Pair[tuple[int, ...]]
