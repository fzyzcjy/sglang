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
    """Prometheus / Stats hot-path. Composition target on Scheduler
    (``self.metrics_reporter``).

    R4 exception: this is the ONLY sister class that holds a
    ``scheduler: Scheduler`` back-reference. Metric emission is a
    read-only panoramic observer over Scheduler state — forcing every
    field access through Callable getters (the alternative) is
    back-reference in disguise and obscures the genuine "see most of
    Scheduler" requirement. Other sister classes do not get this
    exemption."""

    scheduler: "Scheduler"
    tp_rank: int
    pp_rank: int
    dp_rank: Optional[int]
    metrics_collector: Any

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
