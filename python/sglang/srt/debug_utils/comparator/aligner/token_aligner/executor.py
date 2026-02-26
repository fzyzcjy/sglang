from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.dims import (
    TokenDimInfo,
    TokenLayout,
    resolve_dim_by_name,
    strip_dim_names,
)
from sglang.srt.debug_utils.comparator.utils import Pair

_UNNAMED_TOKEN_DIM_FALLBACK: int = 0


def _resolve_dim_or_fallback(tensor: torch.Tensor, name: str) -> int:
    if tensor.names[0] is None:
        return _UNNAMED_TOKEN_DIM_FALLBACK
    return resolve_dim_by_name(tensor, name)


def execute_token_aligner(
    plan: TokenAlignerPlan,
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
    *,
    token_dim_info: Pair[TokenDimInfo] = Pair(
        x=TokenDimInfo(token_dim_name="t"), y=TokenDimInfo(token_dim_name="t")
    ),
) -> Pair[torch.Tensor]:
    if not plan.locators.x.steps:
        return Pair(
            x=_make_empty(
                tensor_of_step=tensor_of_step_pair.x,
                token_dim_info=token_dim_info.x,
                layout=plan.layouts.x,
            ),
            y=_make_empty(
                tensor_of_step=tensor_of_step_pair.y,
                token_dim_info=token_dim_info.y,
                layout=plan.layouts.y,
            ),
        )

    return Pair(
        x=_extract_and_stack_tokens(
            tensor_of_step=tensor_of_step_pair.x,
            locator=plan.locators.x,
            token_dim_info=token_dim_info.x,
            layout=plan.layouts.x,
        ),
        y=_extract_and_stack_tokens(
            tensor_of_step=tensor_of_step_pair.y,
            locator=plan.locators.y,
            token_dim_info=token_dim_info.y,
            layout=plan.layouts.y,
        ),
    )


def _make_empty(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    token_dim_info: TokenDimInfo,
    layout: TokenLayout,
) -> torch.Tensor:
    dummy: torch.Tensor = next(iter(tensor_of_step.values()))

    if layout == TokenLayout.BS:
        batch_dim: int = _resolve_dim_or_fallback(dummy, token_dim_info.token_dim_name)
        assert token_dim_info.seq_dim_name is not None
        seq_dim: int = _resolve_dim_or_fallback(dummy, token_dim_info.seq_dim_name)
        lo, hi = min(batch_dim, seq_dim), max(batch_dim, seq_dim)

        shape: list[int] = list(dummy.shape)
        shape = shape[:lo] + [shape[lo] * shape[hi]] + shape[hi + 1 :]
        shape[lo] = 0
        return torch.empty(shape, dtype=dummy.dtype)

    token_dim: int = _resolve_dim_or_fallback(dummy, token_dim_info.token_dim_name)
    shape = list(dummy.shape)
    shape[token_dim] = 0
    return torch.empty(shape, dtype=dummy.dtype)


def _resolve_bs_layout(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    layout: TokenLayout,
    token_dim_info: TokenDimInfo,
) -> Tuple[dict[int, torch.Tensor], int]:
    """BS layout: collapse B and S dims into a single flat token dim.

    Returns (resolved_tensors, token_dim_index_after_collapse).
    """
    if layout != TokenLayout.BS:
        some_tensor: torch.Tensor = next(iter(tensor_of_step.values()))
        token_dim: int = _resolve_dim_or_fallback(some_tensor, token_dim_info.token_dim_name)
        return tensor_of_step, token_dim

    some_tensor = next(iter(tensor_of_step.values()))
    batch_dim: int = _resolve_dim_or_fallback(some_tensor, token_dim_info.token_dim_name)
    assert token_dim_info.seq_dim_name is not None
    seq_dim: int = _resolve_dim_or_fallback(some_tensor, token_dim_info.seq_dim_name)

    if abs(batch_dim - seq_dim) != 1:
        raise ValueError(
            f"BS dims must be adjacent: "
            f"{token_dim_info.token_dim_name}={batch_dim}, "
            f"{token_dim_info.seq_dim_name}={seq_dim}"
        )

    lo: int = min(batch_dim, seq_dim)
    hi: int = max(batch_dim, seq_dim)

    resolved: dict[int, torch.Tensor] = {}
    for step, tensor in tensor_of_step.items():
        plain: torch.Tensor = strip_dim_names(tensor)
        shape: list[int] = list(plain.shape)
        new_shape: list[int] = shape[:lo] + [shape[lo] * shape[hi]] + shape[hi + 1 :]
        resolved[step] = plain.reshape(new_shape)

    return resolved, lo


def _extract_and_stack_tokens(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    locator: TokenLocator,
    token_dim_info: TokenDimInfo,
    layout: TokenLayout,
) -> torch.Tensor:
    resolved: dict[int, torch.Tensor]
    token_dim: int
    resolved, token_dim = _resolve_bs_layout(
        tensor_of_step=tensor_of_step,
        layout=layout,
        token_dim_info=token_dim_info,
    )

    tokens: list[torch.Tensor] = [
        strip_dim_names(resolved[s]).select(dim=token_dim, index=i)
        for s, i in zip(locator.steps, locator.token_index_in_step)
    ]
    return torch.stack(tokens, dim=token_dim)
