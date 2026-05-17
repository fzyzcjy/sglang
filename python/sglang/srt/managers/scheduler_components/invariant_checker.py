from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# NOTE: this dataclass is intentionally NOT ``frozen=True`` (the project
# default per ``feedback_dataclass_defaults``): ``raise_error_or_warn``
# increments ``count_*_warnings`` via ``setattr`` on ``self``, which would
# raise ``FrozenInstanceError`` against a frozen dataclass. The two
# warning counters are mutable internal state.
@dataclass(kw_only=True, slots=True)
class SchedulerInvariantChecker:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    disaggregation_mode: DisaggregationMode
    page_size: int
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    max_total_num_tokens: int
    server_args: ServerArgs
    tree_cache: BasePrefixCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    req_to_token_pool: ReqToTokenPool
    pool_stats_observer: SchedulerPoolStatsObserver
    get_last_batch: Callable
    get_running_batch: Callable
    get_pool_stats: Callable
    count_req_pool_leak_warnings: int = 0
    count_memory_leak_warnings: int = 0
