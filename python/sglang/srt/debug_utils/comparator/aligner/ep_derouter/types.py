from __future__ import annotations

from typing import Literal

from sglang.srt.debug_utils.comparator.utils import _FrozenBase


class DeRouterPlan(_FrozenBase):
    """Plan for de-routing dispatched MoE tensors back to global token order.

    Each tensor in a step independently carries enough metadata to perform
    de-routing, so this plan operates on individual tensors (not groups).
    """

    type: Literal["de_router"] = "de_router"
    dispatch_path: Literal[
        "fused_moe",
        "deepep_normal",
        "deepep_normal_output_index",
        "deepep_ll",
        "megatron_a2a",
    ]
