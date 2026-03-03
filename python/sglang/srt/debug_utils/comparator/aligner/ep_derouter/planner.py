from __future__ import annotations

from typing import Optional

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.entrypoint import (
    _PLUGIN_REGISTRY,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.types import DeRouterPlan
from sglang.srt.debug_utils.comparator.dims_spec import (
    EP_LIKE_AXES,
    DimsSpec,
)


def maybe_compute_de_router_plan(
    *,
    dims_spec: Optional[DimsSpec],
    available_aux_names: frozenset[str],
) -> list[DeRouterPlan]:
    """Generate a DeRouterPlan if dims contain ``[ep]`` (without reduction) and aux tensors match a plugin.

    ``[ep:partial]`` does NOT trigger the derouter — it indicates that the dump
    has already been scattered to canonical order and only needs a reduce-sum
    across EP ranks (handled by the unsharder).
    """
    if dims_spec is None:
        return []

    has_ep: bool = any(
        mod.axis in EP_LIKE_AXES and mod.reduction is None
        for dim in dims_spec.dims
        for mod in dim.parallel_modifiers
    )
    if not has_ep:
        return []

    dispatch_path: Optional[str] = _infer_dispatch_path(available_aux_names)
    if dispatch_path is None:
        return []

    return [DeRouterPlan(dispatch_path=dispatch_path)]


def _infer_dispatch_path(available_aux_names: frozenset[str]) -> Optional[str]:
    """Infer dispatch path from which plugin's required aux tensors are available."""
    for path, plugin_cls in _PLUGIN_REGISTRY.items():
        if plugin_cls().required_aux_dump_names <= available_aux_names:
            return path
    return None
