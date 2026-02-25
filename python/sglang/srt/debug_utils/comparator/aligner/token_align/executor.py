from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.types import TokenAlignPlan
from sglang.srt.debug_utils.comparator.utils import Pair


def execute_token_align(
    plan: TokenAlignPlan,
    tensors: Pair[dict[int, torch.Tensor]],
) -> Pair[torch.Tensor]:
    """Extract aligned token pairs from step-keyed tensors according to the plan.

    Returns two tensors of shape [num_matched_tokens, ...] with matching tokens
    at corresponding indices.
    """
    if not plan.match_steps.x:
        dummy: torch.Tensor = next(iter(tensors.x.values()))
        empty_shape: list[int] = [0] + list(dummy.shape[1:])
        empty: torch.Tensor = torch.empty(empty_shape, dtype=dummy.dtype)
        return Pair(x=empty, y=empty.clone())

    tokens_x: list[torch.Tensor] = [
        tensors.x[s][i] for s, i in zip(plan.match_steps.x, plan.match_indices.x)
    ]
    tokens_y: list[torch.Tensor] = [
        tensors.y[s][i] for s, i in zip(plan.match_steps.y, plan.match_indices.y)
    ]

    return Pair(x=torch.stack(tokens_x), y=torch.stack(tokens_y))
