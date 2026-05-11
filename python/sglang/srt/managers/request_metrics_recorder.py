from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.server_args import ServerArgs


@dataclass(slots=True, kw_only=True)
class RequestMetricsRecorder:
    """Per-request Prometheus metrics emission."""

    server_args: ServerArgs
    enable_metrics: bool
    enable_priority_scheduling: bool
    metrics_collector: Optional[TokenizerMetricsCollector] = None
