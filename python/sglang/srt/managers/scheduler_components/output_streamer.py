from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from typing import Any, Callable, Dict, List, Optional  # noqa: F401

import torch  # noqa: F401
import zmq  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.environ import envs  # noqa: F401
from sglang.srt.managers.io_struct import (  # noqa: F401
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    GetLoadsReqInput,
    GetLoadsReqOutput,
)
from sglang.srt.managers.schedule_batch import BaseFinishReason, Req  # noqa: F401

logger = logging.getLogger(__name__)


# Module-level constant copied from the original output_processor mixin.
DEFAULT_FORCE_STREAM_INTERVAL = envs.SGLANG_FORCE_STREAM_INTERVAL.get()


@dataclass(kw_only=True, slots=True)
class SchedulerOutputStreamer:
    """Output adapter — serialize finished/sampling-complete reqs into
    ``BatchTokenIDOutput`` / ``BatchEmbeddingOutput`` and write to the
    detokenizer IPC. Composition target on Scheduler
    (``self.output_streamer``)."""

    send_to_detokenizer: Any
    tree_cache: Any
    ps: Any
    server_args: Any
    is_generation: bool
    spec_algorithm: Any
    disaggregation_mode: Any
    enable_hicache_storage: Callable[[], bool]
    load_inquirer_get_loads: Callable[..., Any]
    _test_stream_output_count: int = 0
