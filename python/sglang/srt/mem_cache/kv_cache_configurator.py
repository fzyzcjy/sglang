from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheConfigResult:
    """Configurator output — caller writes back to ModelRunner fields."""

    max_total_num_tokens: int
    max_running_requests: int
    full_max_total_num_tokens: Optional[int]
    swa_max_total_num_tokens: Optional[int]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: KVCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    memory_pool_config: "MemoryPoolConfig"


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheConfigurator:
    """KV cache pipeline (profile -> resolve -> constrain -> init pools).

    Replaces ``ModelRunnerKVCacheMixin`` via composition. ``frozen=True``
    blocks any stale ``self.X = Y`` writes left over from the mixin
    migration; ``slots=True`` blocks attribute typos at runtime;
    ``kw_only=True`` forces named-kwargs construction at the caller.

    Pipeline intermediate state (profiled bytes / resolved configs / pool
    objects) flows through local variables + return values, not via
    attribute writes on ``self``.
    """

    # 部署 env
    device: str
    gpu_id: int
    mem_fraction_static: float
    page_size: int
    # 并行 rank / size
    tp_rank: int
    tp_size: int
    pp_size: int
    dp_size: int
    attention_tp_size: int
    # 模型 / dtype
    model_config: ModelConfig
    server_args: ServerArgs
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    # 推测
    spec_algorithm: SpeculativeAlgorithm
    is_draft_worker: bool
    # 架构 flag
    is_hybrid_swa: bool
    is_hybrid_swa_compress: bool
    use_mla_backend: bool
    enable_hisparse: bool
    mambaish_config: Optional[Any]
    hybrid_gdn_config: Optional[Any]
    # PP 切片
    start_layer: int
    end_layer: int
    num_effective_layers: int
    # 可选预注入（draft worker 复用 target 的 pool）
    req_to_token_pool: Optional[ReqToTokenPool]
    token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator]
    # draft worker 预算
    memory_pool_config: Optional["MemoryPoolConfig"]

    def configure(self, *, pre_model_load_memory: int) -> KVCacheConfigResult:
        raise NotImplementedError("populated in kvc-migrate-method-bodies")
