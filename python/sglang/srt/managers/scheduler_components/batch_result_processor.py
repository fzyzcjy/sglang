from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Union  # noqa: F401

import torch  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.environ import envs  # noqa: F401
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: F401
from sglang.srt.managers.io_struct import AbortReq  # noqa: F401
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # noqa: F401
from sglang.srt.mem_cache.common import (  # noqa: F401
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.server_args import get_global_server_args  # noqa: F401
from sglang.srt.state_capturer.indexer_topk import (  # noqa: F401
    get_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (  # noqa: F401
    get_global_experts_capturer,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (  # noqa: F401
        EmbeddingBatchResult,
        GenerationBatchResult,
    )

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerBatchResultProcessor:
    """``Scheduler.process_batch_result`` hot-path main body. Composition
    target on Scheduler (``self.batch_result_processor``)."""

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
    logprob_computer: Any
    output_streamer: Any
    abort_request: Any
    report_prefill_stats: Any
    report_decode_stats: Any
    update_spec_metrics: Any
    increment_generated_tokens: Any
    advance_forward_ct_decode: Any
