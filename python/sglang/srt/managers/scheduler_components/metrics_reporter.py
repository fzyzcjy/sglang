from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.observability.scheduler_metrics_mixin import (  # noqa: F401
    SchedulerMetricsMixin,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler  # noqa: F401


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SchedulerMetricsReporter:
    """Prometheus / Stats hot-path. Composition target on Scheduler
    (``self.metrics_reporter``)."""

    ps: Any
    server_args: Any
    disaggregation_mode: Any
    spec_algorithm: Any
    metrics_collector: Any
    enable_priority_scheduling: bool
    enable_lora: Any
    enable_hierarchical_cache: bool
    max_running_requests: int
    max_total_num_tokens: int
    tp_rank: int
    pp_rank: int
    dp_rank: Any
    attn_tp_rank: int
    moe_ep_rank: int
    device: str
    model_config: Any
    max_running_requests_under_SLO: Any
    waiting_queue: Any
    grammar_manager: Any
    mm_receiver: Any
    tree_cache: Any
    tp_worker: Any
    draft_worker: Any
    disagg_prefill_bootstrap_queue: Any
    disagg_prefill_inflight_queue: Any
    disagg_decode_prealloc_queue: Any
    disagg_decode_transfer_queue: Any
    kv_events_publisher: Any
    pool_stats_observer: Any
    get_running_batch: Callable
    get_forward_ct: Callable
    get_running_mbs: Callable
    get_last_batch: Callable
    get_grammar_manager: Callable
    get_disaggregation_mode: Callable
    get_disagg_prefill_bootstrap_queue: Callable
    get_disagg_prefill_inflight_queue: Callable
    get_disagg_decode_prealloc_queue: Callable
    get_disagg_decode_transfer_queue: Callable

    def __post_init__(self) -> None:
        # Owned counters (ownership migration from Scheduler).
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        # Run the original init_metrics body via the qualified staticmethod
        # form — methods still live on SchedulerMetricsMixin during prep;
        # the upcoming ``-move`` commit cuts + pastes them into this class
        # and the qualified prefix collapses to ``self.init_metrics(...)``.
        SchedulerMetricsMixin.init_metrics(
            self, self.tp_rank, self.pp_rank, self.dp_rank
        )
        # ``install_device_timer_on_runners`` was originally called from
        # Scheduler.__init__ right after init_model_worker; we invoke it
        # here so callers don't need a separate hook.
        SchedulerMetricsMixin.install_device_timer_on_runners(self)
