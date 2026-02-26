from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.dims import TokenLayout
from sglang.srt.debug_utils.comparator.utils import Pair


def execute_token_aligner(
    plan: TokenAlignerPlan,
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
    *,
    token_dims: Pair[int] = Pair(x=0, y=0),
) -> Pair[torch.Tensor]:
    if not plan.locators.x.steps:
        return Pair(
            x=_make_empty(
                tensor_of_step=tensor_of_step_pair.x,
                token_dim=token_dims.x,
                layout=plan.layouts.x,
            ),
            y=_make_empty(
                tensor_of_step=tensor_of_step_pair.y,
                token_dim=token_dims.y,
                layout=plan.layouts.y,
            ),
        )

    return Pair(
        x=_extract_and_stack_tokens(
            tensor_of_step=tensor_of_step_pair.x,
            locator=plan.locators.x,
            token_dim=token_dims.x,
            layout=plan.layouts.x,
        ),
        y=_extract_and_stack_tokens(
            tensor_of_step=tensor_of_step_pair.y,
            locator=plan.locators.y,
            token_dim=token_dims.y,
            layout=plan.layouts.y,
        ),
    )


def _make_empty(
    *, tensor_of_step: dict[int, torch.Tensor], token_dim: int, layout: TokenLayout
) -> torch.Tensor:
    dummy: torch.Tensor = next(iter(tensor_of_step.values()))
    shape: list[int] = list(dummy.shape)

    if layout == TokenLayout.BS and dummy.ndim >= 2:
        # [B, S, ...] → collapse B*S into token_dim, drop batch dim
        shape = [shape[0] * shape[1]] + shape[2:]

    shape[token_dim] = 0
    return torch.empty(shape, dtype=dummy.dtype)


def _resolve_bshd(
    *, tensor_of_step: dict[int, torch.Tensor], layout: TokenLayout
) -> dict[int, torch.Tensor]:
    """BSHD: reshape [B, S, ...] → [B*S, ...] so token indices are flat."""
    if layout != TokenLayout.BS:
        return tensor_of_step

    resolved: dict[int, torch.Tensor] = {}
    for step, tensor in tensor_of_step.items():
        if tensor.ndim >= 2:
            resolved[step] = tensor.reshape(-1, *tensor.shape[2:])
        else:
            resolved[step] = tensor
    return resolved


def _extract_and_stack_tokens(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    locator: TokenLocator,
    token_dim: int,
    layout: TokenLayout,
) -> torch.Tensor:
    resolved: dict[int, torch.Tensor] = _resolve_bshd(
        tensor_of_step=tensor_of_step, layout=layout
    )
    tokens: list[torch.Tensor] = [
        resolved[s].select(dim=token_dim, index=i)
        for s, i in zip(locator.steps, locator.token_index_in_step)
    ]
    return torch.stack(tokens, dim=token_dim)
