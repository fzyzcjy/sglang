from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

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
