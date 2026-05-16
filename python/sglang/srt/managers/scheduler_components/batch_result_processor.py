from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerBatchResultProcessor:
    is_generation: bool
    disaggregation_mode: Any
    enable_overlap: bool
    enable_overlap_mlx: bool
    server_args: Any
    model_config: Any
    token_to_kv_pool_allocator: Any
    tree_cache: Any
    hisparse_coordinator: Any
    req_to_token_pool: Any
    decode_offload_manager: Any
    metrics_collector: Any
    draft_worker: Any
    model_worker: Any
    logprob_result_processor: Any
    output_streamer: Any
    abort_request: Any
    report_prefill_stats: Any
    report_decode_stats: Any
    update_spec_metrics: Any
    increment_generated_tokens: Any
    advance_forward_ct_decode: Any
