from __future__ import annotations  # noqa: F401

import dataclasses  # noqa: F401
import time  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from typing import Any, Callable, Optional  # noqa: F401

from sglang.srt.disaggregation.kv_events import (  # noqa: F401
    EventPublisherFactory,
    KVEventBatch,
)


# ``SchedulerStats`` referenced only as a type hint in ``emit_kv_metrics`` —
# leave a forward-ref placeholder.
class SchedulerStats: ...  # type: ignore[no-redef]


@dataclasses.dataclass
class KvMetrics:
    request_active_slots: int = 0
    request_total_slots: int = 0
    kv_active_blocks: int = 0
    kv_total_blocks: int = 0
    num_requests_waiting: int = 0
    gpu_cache_usage_perc: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0
    data_parallel_rank: int = 0


@dataclass(kw_only=True, slots=True)
class SchedulerKvEventsPublisher:
    """KV cache event / metrics publication channel. Composition target on
    Scheduler (``self.kv_events_publisher``)."""

    kv_events_config: Optional[str]
    attn_tp_rank: int
    attn_cp_rank: int
    attn_dp_rank: int
    dp_rank: Optional[int]
    tree_cache: Any
    send_metrics_from_scheduler: Any
    max_running_requests: int
    max_total_num_tokens: int
    get_stats: Callable
    enable_kv_cache_events: bool = False
    kv_event_publisher: Any = None
