from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
import warnings  # noqa: F401
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.environ import envs  # noqa: F401
from sglang.srt.managers.scheduler_components.pool_stats_observer import (  # noqa: F401
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.srt.utils.common import ceil_align, raise_error_or_warn  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerInvariantChecker:
    """KV pool / req pool / tree_cache memory invariant checks.
    Composition target on Scheduler (``self.invariant_checker``)."""

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
