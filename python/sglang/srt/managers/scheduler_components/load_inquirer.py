from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SchedulerLoadInquirer:
    """``/v1/loads`` RPC handler. Composition target on Scheduler
    (``self.load_inquirer``)."""

    def __init__(
        self,
        *,
        disaggregation_mode,
        ps,
        max_total_num_tokens: int,
        max_running_requests: int,
        enable_lora: bool,
        pool_stats_observer,
        tp_worker,
        token_to_kv_pool_allocator,
        spec_algorithm,
    ) -> None:
        self.disaggregation_mode = disaggregation_mode
        self.ps = ps
        self.max_total_num_tokens = max_total_num_tokens
        self.max_running_requests = max_running_requests
        self.enable_lora = enable_lora
        self.pool_stats_observer = pool_stats_observer
        self.tp_worker = tp_worker
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.spec_algorithm = spec_algorithm
