from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase

# seq_id type: str (SGLang rid) or tuple[int, int] (Megatron (step, seq_index))
ExternalSeqId = Union[str, tuple[int, int]]


@dataclass(frozen=True)
class AuxTensorsForStep:
    """Normalized auxiliary tensors for a single step (framework-agnostic)."""

    input_ids: torch.Tensor  # [T] (1D flat)
    positions: torch.Tensor  # [T] (1D flat)
    seq_lens: torch.Tensor  # [num_seqs]
    seq_ids: tuple[ExternalSeqId, ...]  # [num_seqs] — sequence identity


@dataclass(frozen=True)
class SideAux:
    """Auxiliary tensors for one side across all steps + side-level metadata."""

    steps: dict[int, AuxTensorsForStep]
    framework: str  # "sglang" | "megatron"
    layout: str  # "thd"


class SeqInfo(_FrozenBase):
    """Information for a sequence, containing information to locate all the tokens inside the sequence."""

    input_ids: list[int]
    positions: list[int]
    steps: list[int]
    indices: list[int]


class SeqsInfo(_FrozenBase):
    """All sequences for one side across all steps."""

    sequences: dict[int, SeqInfo]
    layout: str


class AlignmentPlan(_FrozenBase):
    """Token alignment plan.

    match_steps.x[i] + match_indices.x[i] and match_steps.y[i] + match_indices.y[i]
    correspond to the same logical token.
    """

    match_steps: Pair[tuple[int, ...]]
    match_indices: Pair[tuple[int, ...]]
