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
        # The 4 "metrics enablement" booleans now live on the reporter (was
        # Scheduler in main). ``enable_metrics`` / ``is_stats_logging_rank`` /
        # ``current_scheduler_metrics_enabled`` get re-assigned inside
        # ``_init_metrics`` too (kept in REPORTER_OWNED_ATTRS so the body
        # rewrite leaves them on ``self``); we set them up-front here so any
        # reader between ``__post_init__`` and ``_init_metrics`` is correct.
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
        # Bootstrap calls below were previously emitted as separate Scheduler
        # statements at construction time; consolidating them into
        # ``__post_init__`` tightens ownership (the reporter is responsible for
        # its own bring-up).
        SchedulerMetricsMixin._init_metrics(
            self, self.tp_rank, self.pp_rank, self.dp_rank
        )
        SchedulerMetricsMixin._install_device_timer_on_runners(self)
