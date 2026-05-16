from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable


from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


DEFAULT_FORCE_STREAM_INTERVAL = envs.SGLANG_FORCE_STREAM_INTERVAL.get()


@dataclass(kw_only=True, slots=True)
class SchedulerOutputStreamer:
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
