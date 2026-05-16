from __future__ import annotations  # noqa: F401

from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerDPAttnAdapter:
    tp_group: Any
    req_to_token_pool: Any
    token_to_kv_pool_allocator: Any
    tree_cache: Any
    offload_tags: Any
    ps: Any
    server_args: Any
    model_config: Any
    enable_overlap: bool
    spec_algorithm: Any
    require_mlp_sync: bool
