from __future__ import annotations

import warnings
from typing import List, Tuple

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
)
from sglang.srt.utils.common import ceil_align, raise_error_or_warn


class SchedulerInvariantChecker:
    """KV pool / req pool / tree_cache memory invariant checks.
    Composition target on Scheduler (``self.invariant_checker``)."""

    def __init__(
        self,
        *,
        is_hybrid_swa: bool,
        is_hybrid_ssm: bool,
        disaggregation_mode,
        page_size: int,
        full_tokens_per_layer,
        swa_tokens_per_layer,
        max_total_num_tokens: int,
        server_args,
        tree_cache,
        token_to_kv_pool_allocator,
        req_to_token_pool,
        pool_stats_observer,
    ) -> None:
        self.is_hybrid_swa = is_hybrid_swa
        self.is_hybrid_ssm = is_hybrid_ssm
        self.disaggregation_mode = disaggregation_mode
        self.page_size = page_size
        self.full_tokens_per_layer = full_tokens_per_layer
        self.swa_tokens_per_layer = swa_tokens_per_layer
        self.max_total_num_tokens = max_total_num_tokens
        self.server_args = server_args
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.pool_stats_observer = pool_stats_observer
        self.count_req_pool_leak_warnings: int = 0
        self.count_memory_leak_warnings: int = 0

    @staticmethod
    def _check_pool_invariant(
        *,
        pool_name: str,
        available: int,
        evictable: int,
        protected: int,
        session_held: int,
        total: int,
        uncached: int = 0,
    ) -> Tuple[bool, str]:
        """Check that available + evictable + protected + session_held + uncached == total."""
        accounted = available + evictable + protected + session_held + uncached
        if accounted != total:
            msg = (
                f"{pool_name} pool size mismatch: "
                f"available={available}, evictable={evictable}, "
                f"protected={protected}, session_held={session_held}, "
                f"uncached={uncached}, total={total}, accounted={accounted}, "
                f"diff={total - accounted}"
            )
            return True, msg
        return False, ""

    def _check_full_pool(
        self,
        *,
        ps: PoolStats,
        last_batch,
        running_batch,
        uncached: int = 0,
    ) -> Tuple[bool, str]:
        if self.is_hybrid_swa:
            available = ps.full_available_size
            evictable = ps.full_evictable_size
            protected = self.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_full_tokens(
                last_batch=last_batch, running_batch=running_batch
            )
            total = self.full_tokens_per_layer
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            available = ps.full_available_size
            evictable = ps.full_evictable_size
            protected = self.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_tokens(
                last_batch=last_batch, running_batch=running_batch
            )
            total = self.token_to_kv_pool_allocator.size
        else:
            available = ps.full_available_size
            evictable = ps.full_evictable_size
            protected = self.tree_cache.protected_size()
            session_held = self.pool_stats_observer.session_held_tokens(
                last_batch=last_batch, running_batch=running_batch
            )
            total = self.max_total_num_tokens
        return self._check_pool_invariant(
            pool_name="full",
            available=available,
            evictable=evictable,
            protected=protected,
            session_held=session_held,
            total=total,
            uncached=uncached,
        )

    def _check_swa_pool(
        self,
        *,
        ps: PoolStats,
        last_batch,
        running_batch,
        uncached: int = 0,
    ) -> Tuple[bool, str]:
        available = ps.swa_available_size
        evictable = ps.swa_evictable_size
        protected = self.tree_cache.swa_protected_size()
        session_held = self.pool_stats_observer.session_held_swa_tokens(
            last_batch=last_batch, running_batch=running_batch
        )
        total = self.swa_tokens_per_layer
        return self._check_pool_invariant(
            pool_name="swa",
            available=available,
            evictable=evictable,
            protected=protected,
            session_held=session_held,
            total=total,
            uncached=uncached,
        )

    def _check_mamba_pool(
        self, *, ps: PoolStats, last_batch, running_batch
    ) -> Tuple[bool, str]:
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        if is_mamba_radix_cache:
            mamba_available = self.req_to_token_pool.mamba_pool.available_size()
            mamba_evictable = self.tree_cache.mamba_evictable_size()
            mamba_protected = self.tree_cache.mamba_protected_size()
            mamba_total = self.req_to_token_pool.mamba_pool.size
            session_held = self.pool_stats_observer.session_held_mamba_slots(
                last_batch=last_batch, running_batch=running_batch
            )
            return self._check_pool_invariant(
                pool_name="mamba",
                available=mamba_available,
                evictable=mamba_evictable,
                protected=mamba_protected,
                session_held=session_held,
                total=mamba_total,
            )
        return False, ""

    def _get_total_uncached_sizes(
        self, *, last_batch, running_batch
    ) -> Tuple[int, int]:
        """Sum uncached tokens for full and SWA pools across all active batches.

        Returns (full_uncached, swa_uncached). For non-SWA models, swa_uncached is 0.

        For full pool: uncached = allocated - cache_protected_len
        For SWA pool:  uncached = allocated - max(cache_protected_len, swa_evicted_seqlen)
        """
        # After decode: running_batch IS last_batch (same object), count once.
        # After prefill: they differ, both hold uncached tokens.
        batches = [last_batch]
        if running_batch not in (None, last_batch) and not running_batch.is_empty():
            batches.append(running_batch)

        full_uncached = 0
        swa_uncached = 0
        for batch in batches:
            for req in batch.reqs:
                assert req.kv_committed_freed == req.kv_overallocated_freed
                if req.kv_committed_freed or req.req_pool_idx is None:
                    continue

                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0

                full_uncached += allocated_len - req.cache_protected_len
                if self.is_hybrid_swa:
                    swa_uncached += allocated_len - max(
                        req.cache_protected_len, req.swa_evicted_seqlen
                    )

        return full_uncached, swa_uncached

    def self_check_during_busy(self, *, last_batch, running_batch) -> None:
        """Check memory invariants during busy state (hot-path adjacent)."""
        if last_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return
        ps = self.pool_stats_observer.get_pool_stats(
            last_batch=last_batch, running_batch=running_batch
        )
        full_uncached, swa_uncached = self._get_total_uncached_sizes(
            last_batch=last_batch, running_batch=running_batch
        )
        full_leak, full_msg = self._check_full_pool(
            ps=ps,
            last_batch=last_batch,
            running_batch=running_batch,
            uncached=full_uncached,
        )
        if full_leak:
            self._report_leak("full", full_msg)
        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(
                ps=ps,
                last_batch=last_batch,
                running_batch=running_batch,
                uncached=swa_uncached,
            )
            if swa_leak:
                self._report_leak("swa", swa_msg)

    def _check_req_pool(self) -> None:
        session_req_count = self.pool_stats_observer.session_held_req_count()
        req_total_size = self.req_to_token_pool.size
        if len(self.req_to_token_pool.free_slots) + session_req_count != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def _report_leak(self, pool_name: str, token_msg: str) -> None:
        msg = f"{pool_name} memory leak detected! {token_msg}"
        raise_error_or_warn(
            self,
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
            "count_memory_leak_warnings",
            msg,
        )

    def _check_all_pools(
        self,
        *,
        ps: PoolStats,
        last_batch,
        running_batch,
        uncached: int = 0,
    ) -> Tuple[bool, List[str]]:
        """Check memory invariant across all pools. Returns (has_leak, messages)."""
        has_leak = False
        messages = []

        full_leak, full_msg = self._check_full_pool(
            ps=ps, last_batch=last_batch, running_batch=running_batch, uncached=uncached
        )
        has_leak |= full_leak
        messages.append(full_msg)

        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(
                ps=ps, last_batch=last_batch, running_batch=running_batch
            )
            has_leak |= swa_leak
            messages.append(swa_msg)

        if self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            mamba_leak, mamba_msg = self._check_mamba_pool(
                ps=ps, last_batch=last_batch, running_batch=running_batch
            )
            has_leak |= mamba_leak
            messages.append(mamba_msg)

        return has_leak, messages

    def _check_tree_cache(self) -> None:
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()
