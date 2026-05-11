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

    def get_loads(
        self,
        req: GetLoadsReqInput = None,
        *,
        running_batch,
        waiting_queue,
        stats,
        spec_total_num_accepted_tokens: int,
        spec_total_num_forward_ct: int,
        disagg_prefill_bootstrap_queue,
        disagg_prefill_inflight_queue,
        disagg_decode_prealloc_queue,
        disagg_decode_transfer_queue,
    ) -> GetLoadsReqOutput:
        """
        Get comprehensive load metrics for /v1/loads endpoint.

        Args:
            req: Request containing include list and optional dp_rank filter

        Returns:
            GetLoadsReqOutput with core metrics and optional detailed sections
        """
        if req is None:
            req = GetLoadsReqInput()

        include = set(req.include) if req.include else {"core"}
        include_all = "all" in include

        num_running_reqs = len(running_batch.reqs)

        waiting_queues = [waiting_queue]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            waiting_queues.append(disagg_decode_prealloc_queue.queue)
            waiting_queues.append(disagg_decode_transfer_queue.queue)
            waiting_queues.append(disagg_decode_prealloc_queue.retracted_queue)

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_used_tokens, kv_token_usage = self.pool_stats_observer.get_pool_stats(
            last_batch=None,
            running_batch=running_batch,
        ).get_kv_token_stats()
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in waiting_queues for req in queue
        )

        memory = None
        if include_all or "memory" in include:
            try:
                memory = MemoryMetrics(
                    weight_gb=round(
                        self.tp_worker.model_runner.weight_load_mem_usage, 3
                    ),
                    kv_cache_gb=round(
                        self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                    ),
                    graph_gb=round(self.tp_worker.model_runner.graph_mem_usage, 3),
                    token_capacity=int(self.max_total_num_tokens),
                )
            except AttributeError as e:
                logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if include_all or "spec" in include:
            if not self.spec_algorithm.is_none() and spec_total_num_forward_ct > 0:
                speculative = SpeculativeMetrics(
                    accept_length=(
                        spec_total_num_accepted_tokens / spec_total_num_forward_ct
                    ),
                    accept_rate=stats.spec_accept_rate,
                )

        lora = None
        if include_all or "lora" in include:
            if self.enable_lora:
                lora = LoRAMetrics(
                    slots_used=stats.lora_pool_slots_used,
                    slots_total=stats.lora_pool_slots_total,
                    utilization=stats.lora_pool_utilization,
                )

        disaggregation = None
        if include_all or "disagg" in include:
            mode_str = "null"
            prefill_bootstrap = 0
            prefill_inflight = 0
            decode_prealloc = 0
            decode_transfer = 0
            decode_retracted = 0

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                mode_str = "prefill"
                prefill_bootstrap = len(disagg_prefill_bootstrap_queue.queue)
                prefill_inflight = len(disagg_prefill_inflight_queue)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                mode_str = "decode"
                decode_prealloc = len(disagg_decode_prealloc_queue.queue)
                decode_transfer = len(disagg_decode_transfer_queue.queue)
                decode_retracted = len(disagg_decode_prealloc_queue.retracted_queue)

            disaggregation = DisaggregationMetrics(
                mode=mode_str,
                prefill_bootstrap_queue_reqs=prefill_bootstrap,
                prefill_inflight_queue_reqs=prefill_inflight,
                decode_prealloc_queue_reqs=decode_prealloc,
                decode_transfer_queue_reqs=decode_transfer,
                decode_retracted_queue_reqs=decode_retracted,
                kv_transfer_speed_gb_s=stats.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=stats.kv_transfer_latency_ms,
            )

        queues = None
        if include_all or "queues" in include:
            queues = QueueMetrics(
                waiting=len(waiting_queue),
                grammar=stats.num_grammar_queue_reqs,
                paused=stats.num_paused_reqs,
                retracted=stats.num_retracted_reqs,
            )

        return GetLoadsReqOutput(
            dp_rank=self.ps.dp_rank,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(stats.gen_throughput, 2),
            cache_hit_rate=round(stats.cache_hit_rate, 4),
            utilization=round(stats.utilization, 4),
            max_running_requests=self.max_running_requests,
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
        )
