from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerInvariantChecker:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    disaggregation_mode: Any
    page_size: int
    full_tokens_per_layer: Any
    swa_tokens_per_layer: Any
    max_total_num_tokens: int
    server_args: Any
    tree_cache: Any
    token_to_kv_pool_allocator: Any
    req_to_token_pool: Any
    pool_stats_observer: SchedulerPoolStatsObserver
    get_last_batch: Callable
    get_running_batch: Callable
    get_pool_stats: Callable
    count_req_pool_leak_warnings: int = 0
    count_memory_leak_warnings: int = 0
