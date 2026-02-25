from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.types import TokenAlignPlan
from sglang.srt.debug_utils.comparator.utils import Pair


def execute_token_align(
    plan: TokenAlignPlan,
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
) -> Pair[torch.Tensor]:
    if not plan.match_steps.x:
        dummy: torch.Tensor = next(iter(tensor_of_step_pair.x.values()))
        empty_shape: list[int] = [0] + list(dummy.shape[1:])
        empty: torch.Tensor = torch.empty(empty_shape, dtype=dummy.dtype)
        return Pair(x=empty, y=empty.clone())

    tokens_x: list[torch.Tensor] = [
        tensor_of_step_pair.x[s][i] for s, i in zip(plan.match_steps.x, plan.match_indices.x)
    ]
    tokens_y: list[torch.Tensor] = [
        tensor_of_step_pair.y[s][i] for s, i in zip(plan.match_steps.y, plan.match_indices.y)
    ]

    return Pair(x=torch.stack(tokens_x), y=torch.stack(tokens_y))
