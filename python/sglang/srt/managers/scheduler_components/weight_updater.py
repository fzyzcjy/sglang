from __future__ import annotations  # noqa: F401

from dataclasses import dataclass, field  # noqa: F401
from typing import Any, Callable, Optional  # noqa: F401


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    """Hot weight-update / memory-occupation / model-save / weight-inspection
    control surface. Composition target on Scheduler
    (``self.weight_updater``)."""

    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
