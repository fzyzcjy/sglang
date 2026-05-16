from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.observability.scheduler_metrics_mixin import (  # noqa: F401
    SchedulerMetricsMixin,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler  # noqa: F401


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SchedulerMetricsReporter:
    scheduler: "Scheduler"
    tp_rank: int
    pp_rank: int
    dp_rank: Optional[int]
    metrics_collector: Any

    def __post_init__(self) -> None:
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        SchedulerMetricsMixin.init_metrics(
            self, self.tp_rank, self.pp_rank, self.dp_rank
        )
        SchedulerMetricsMixin.install_device_timer_on_runners(self)
