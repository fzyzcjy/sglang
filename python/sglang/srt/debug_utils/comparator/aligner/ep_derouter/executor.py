from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.entrypoint import (
    _PLUGIN_REGISTRY,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.types import DeRouterPlan

if TYPE_CHECKING:
    from sglang.srt.debug_utils.comparator.raw_aux_loader import RawAuxLoader


def execute_de_router_plan(
    plan: DeRouterPlan,
    tensor: torch.Tensor,
    *,
    aux_loader: Optional[RawAuxLoader],
    meta: dict[str, Any],
) -> torch.Tensor:
    """Execute a de-router plan on a single tensor.

    Resolves the plugin from ``plan.dispatch_path``, loads the required
    auxiliary tensors via *aux_loader*, then calls
    ``flatten_routed_tensor`` + ``compute_forward_permutation`` and
    applies the unified scatter.
    """
    plugin_cls: type[DeRouterPlugin] | None = _PLUGIN_REGISTRY.get(plan.dispatch_path)
    if plugin_cls is None:
        raise ValueError(
            f"Unknown dispatch_path {plan.dispatch_path!r}. "
            f"Available: {sorted(_PLUGIN_REGISTRY.keys())}"
        )

    plugin: DeRouterPlugin = plugin_cls()

    ep_meta_names: frozenset[str] = frozenset(
        {
            f"{plan.dispatch_path}_ep_num_tokens",
            f"{plan.dispatch_path}_ep_top_k",
        }
    )
    all_required: frozenset[str] = plugin.required_aux_dump_names | ep_meta_names

    aux_tensors: dict[str, torch.Tensor] = _load_aux_tensors(
        required_names=all_required,
        cross_rank_names=plugin.cross_rank_aux_names,
        aux_loader=aux_loader,
        meta=meta,
    )

    num_tokens: int = int(aux_tensors.pop(f"{plan.dispatch_path}_ep_num_tokens").item())
    top_k: int = int(aux_tensors.pop(f"{plan.dispatch_path}_ep_top_k").item())

    num_tokens = plugin.resolve_num_tokens(
        num_tokens=num_tokens, aux_tensors=aux_tensors
    )

    flat_tensor: torch.Tensor = plugin.flatten_routed_tensor(
        routed_tensor=tensor, aux_tensors=aux_tensors
    )
    forward_perm: torch.Tensor = plugin.compute_forward_permutation(
        aux_tensors=aux_tensors,
        num_tokens=num_tokens,
        top_k=top_k,
        num_routed=flat_tensor.shape[0],
    )

    return _apply_forward_permutation(
        flat_routed=flat_tensor,
        forward_perm=forward_perm,
        total_slots=num_tokens * top_k,
    )


def _load_aux_tensors(
    *,
    required_names: frozenset[str],
    cross_rank_names: frozenset[str],
    aux_loader: Optional[RawAuxLoader],
    meta: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Load auxiliary tensors required by a de-router plugin."""
    if not required_names:
        return {}

    if aux_loader is None:
        raise ValueError(
            f"De-router requires auxiliary tensors {sorted(required_names)}, "
            f"but no aux_loader was provided."
        )

    result: dict[str, torch.Tensor] = aux_loader.load(
        step=int(meta["step"]),
        rank=int(meta["rank"]),
        layer_id=int(meta["layer_id"]),
        aux_names=required_names,
    )

    missing: set[str] = set()
    for name in required_names:
        if name not in result:
            missing.add(name)

    empty_cross_rank: set[str] = {
        name
        for name in cross_rank_names
        if name in result and result[name].numel() == 0
    }

    cross_rank_needed: set[str] = (missing & cross_rank_names) | empty_cross_rank
    if cross_rank_needed:
        result.update(
            _load_cross_rank(
                names=frozenset(cross_rank_needed),
                aux_loader=aux_loader,
                meta=meta,
            )
        )
        missing -= set(result.keys())

    if missing:
        raise ValueError(
            f"De-router requires auxiliary tensor(s) {sorted(missing)}, "
            f"but they were not found in the dump. "
            f"Available: {sorted(result.keys())}"
        )

    return result


def _load_cross_rank(
    *,
    names: frozenset[str],
    aux_loader: RawAuxLoader,
    meta: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Try loading aux tensors from any rank (for DP-attention scenarios)."""
    step: int = int(meta["step"])
    layer_id: int = int(meta["layer_id"])
    world_size: int = int(meta.get("world_size", 1))

    for rank in range(world_size):
        result: dict[str, torch.Tensor] = aux_loader.load(
            step=step,
            rank=rank,
            layer_id=layer_id,
            aux_names=names,
        )
        found: dict[str, torch.Tensor] = {
            name: tensor
            for name, tensor in result.items()
            if tensor.numel() > 0
        }
        if found:
            return found

    return {}


def _apply_forward_permutation(
    flat_routed: torch.Tensor,
    forward_perm: torch.Tensor,
    *,
    total_slots: int,
) -> torch.Tensor:
    """Scatter flat_routed into output using forward_perm indices.

    Positions where ``forward_perm[i] == -1`` are discarded (padding).
    """
    flat_unnamed: torch.Tensor = flat_routed.rename(None)
    forward_perm = forward_perm[: flat_unnamed.shape[0]]
    valid_mask: torch.Tensor = forward_perm >= 0
    trailing_shape: list[int] = list(flat_unnamed.shape[1:])

    output: torch.Tensor = torch.zeros(
        [total_slots] + trailing_shape,
        dtype=flat_unnamed.dtype,
        device=flat_unnamed.device,
    )
    output[forward_perm[valid_mask]] = flat_unnamed[valid_mask]

    return output
