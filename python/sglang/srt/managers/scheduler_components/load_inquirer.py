from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
import time  # noqa: F401
from dataclasses import dataclass
from typing import Any, Callable, Optional  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.managers.io_struct import (  # noqa: F401
    DisaggregationMetrics,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    """``/v1/loads`` RPC handler. Composition target on Scheduler
    (``self.load_inquirer``)."""

    disaggregation_mode: Any
    ps: Any
    server_args: Any
    max_total_num_tokens: int
    max_running_requests: int
    pool_stats_observer: Any
    tp_worker: Any
    token_to_kv_pool_allocator: Any
    spec_algorithm: Any
    get_running_batch: Callable
    get_waiting_queue: Callable
    get_stats: Callable
    get_chunked_req: Callable
    get_disagg_prefill_bootstrap_queue: Callable
    get_disagg_prefill_inflight_queue: Callable
    get_disagg_decode_prealloc_queue: Callable
    get_disagg_decode_transfer_queue: Callable
    get_spec_total_num_accepted_tokens: Callable
    get_spec_total_num_forward_ct: Callable
