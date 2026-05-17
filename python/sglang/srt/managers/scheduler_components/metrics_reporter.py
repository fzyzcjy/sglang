from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.scheduler_metrics_mixin import (
    SchedulerMetricsMixin,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SchedulerMetricsReporter:
    scheduler: "Scheduler"
    tp_rank: int
    pp_rank: int
    dp_rank: Optional[int]
    metrics_collector: SchedulerMetricsCollector
    num_retracted_reqs: int = 0
    num_paused_reqs: int = 0

    def __post_init__(self) -> None:
        self.enable_metrics = self.scheduler.server_args.enable_metrics
        self.is_stats_logging_rank = self.scheduler.ps.attn_tp_rank == 0
        self.current_scheduler_metrics_enabled = self.enable_metrics and (
            self.is_stats_logging_rank
            or self.scheduler.server_args.enable_metrics_for_all_schedulers
        )
        self.enable_kv_cache_events = bool(
            self.scheduler.server_args.kv_events_config
            and self.scheduler.ps.attn_tp_rank == 0
            and self.scheduler.ps.attn_cp_rank == 0
        )
        SchedulerMetricsMixin._init_metrics(
            self, self.tp_rank, self.pp_rank, self.dp_rank
        )
        SchedulerMetricsMixin._install_device_timer_on_runners(self)
