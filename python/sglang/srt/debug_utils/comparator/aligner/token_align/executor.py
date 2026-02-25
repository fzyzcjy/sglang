from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.types import AlignmentPlan
from sglang.srt.debug_utils.comparator.utils import Pair


def execute_alignment(
    plan: AlignmentPlan,
    tensors: Pair[dict[int, torch.Tensor]],
) -> Pair[torch.Tensor]:
    """Extract aligned token pairs from step-keyed tensors according to the plan.

    Returns two tensors of shape [num_matched_tokens, ...] with matching tokens
    at corresponding indices.
    """
    tensors_a: dict[int, torch.Tensor] = tensors.a
    tensors_b: dict[int, torch.Tensor] = tensors.b

    if plan.layouts.a == "bshd":
        tensors_a = {s: t.flatten(0, 1) for s, t in tensors_a.items()}
    if plan.layouts.b == "bshd":
        tensors_b = {s: t.flatten(0, 1) for s, t in tensors_b.items()}

    if not plan.match_steps.a:
        dummy: torch.Tensor = next(iter(tensors_a.values()))
        empty_shape: list[int] = [0] + list(dummy.shape[1:])
        empty: torch.Tensor = torch.empty(empty_shape, dtype=dummy.dtype)
        return Pair(a=empty, b=empty.clone())

    tokens_a: list[torch.Tensor] = [
        tensors_a[s][i] for s, i in zip(plan.match_steps.a, plan.match_indices.a)
    ]
    tokens_b: list[torch.Tensor] = [
        tensors_b[s][i] for s, i in zip(plan.match_steps.b, plan.match_indices.b)
    ]

    return Pair(a=torch.stack(tokens_a), b=torch.stack(tokens_b))
