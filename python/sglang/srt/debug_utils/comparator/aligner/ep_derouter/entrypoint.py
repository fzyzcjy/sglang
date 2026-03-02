from __future__ import annotations

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_ll import (
    DeepEPLLDeRouter,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_normal import (
    DeepEPNormalDeRouter,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_normal_output_index import (
    DeepEPNormalOutputIndexDeRouter,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.fused_moe import (
    FusedMoEDeRouter,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.megatron_a2a import (
    MegatronA2ADeRouter,
)

_PLUGIN_REGISTRY: dict[str, type[DeRouterPlugin]] = {
    "fused_moe": FusedMoEDeRouter,
    "deepep_normal": DeepEPNormalDeRouter,
    "deepep_normal_output_index": DeepEPNormalOutputIndexDeRouter,
    "deepep_ll": DeepEPLLDeRouter,
    "megatron_a2a": MegatronA2ADeRouter,
}
