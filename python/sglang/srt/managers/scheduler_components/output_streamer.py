from __future__ import annotations

import logging
from typing import Any, Callable


from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


# Module-level constant copied from the original output_processor mixin.
DEFAULT_FORCE_STREAM_INTERVAL = envs.SGLANG_FORCE_STREAM_INTERVAL.get()


class SchedulerOutputStreamer:
    """Output adapter — serialize finished/sampling-complete reqs into
    ``BatchTokenIDOutput`` / ``BatchEmbeddingOutput`` and write to the
    detokenizer IPC. Composition target on Scheduler
    (``self.output_streamer``)."""

    def __init__(
        self,
        *,
        send_to_detokenizer,
        tree_cache,
        ps,
        server_args,
        is_generation: bool,
        stream_interval: int,
        spec_algorithm,
        disaggregation_mode,
        enable_hicache_storage: Callable[[], bool],
        load_inquirer_get_loads: Callable[..., Any],
    ) -> None:
        self.send_to_detokenizer = send_to_detokenizer
        self.tree_cache = tree_cache
        self.ps = ps
        self.server_args = server_args
        self.is_generation = is_generation
        self.stream_interval = stream_interval
        self.spec_algorithm = spec_algorithm
        self.disaggregation_mode = disaggregation_mode
        self.enable_hicache_storage = enable_hicache_storage
        self.load_inquirer_get_loads = load_inquirer_get_loads
        self._test_stream_output_count: int = 0
