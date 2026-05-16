from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLogprobResultProcessor:
    server_args: Any
    model_config: Any
