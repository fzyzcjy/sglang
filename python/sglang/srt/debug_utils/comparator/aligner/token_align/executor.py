from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.types import TokenAlignmentPlan
from sglang.srt.debug_utils.comparator.utils import Pair


def execute_alignment(
    plan: TokenAlignmentPlan,
    tensors: Pair[dict[int, torch.Tensor]],
) -> Pair[torch.Tensor]:
    """Extract aligned token pairs from step-keyed tensors according to the plan.

    Returns two tensors of shape [num_matched_tokens, ...] with matching tokens
    at corresponding indices.
    """
    tensors_a: dict[int, torch.Tensor] = tensors.x
    tensors_b: dict[int, torch.Tensor] = tensors.y

    if not plan.match_steps.x:
        dummy: torch.Tensor = next(iter(tensors_a.values()))
        empty_shape: list[int] = [0] + list(dummy.shape[1:])
        empty: torch.Tensor = torch.empty(empty_shape, dtype=dummy.dtype)
        return Pair(x=empty, y=empty.clone())

    tokens_a: list[torch.Tensor] = [
        tensors_a[s][i] for s, i in zip(plan.match_steps.x, plan.match_indices.x)
    ]
    tokens_b: list[torch.Tensor] = [
        tensors_b[s][i] for s, i in zip(plan.match_steps.y, plan.match_indices.y)
    ]

    return Pair(x=torch.stack(tokens_a), y=torch.stack(tokens_b))
