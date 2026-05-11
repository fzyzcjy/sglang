from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.server_args import ServerArgs


@dataclass(slots=True, kw_only=True)
class LoraControllerConfig:
    enable_lora: bool
    max_loaded_loras: Optional[int]
    dp_size: int
    initial_lora_paths: Optional[list]


@dataclass(slots=True, kw_only=True)
class LoraController:
    """LoRA load/unload/LRU + per-request acquire/release."""

    server_args: ServerArgs
    auto_create_handle_loop: Callable[[], None]
    update_lora_adapter_communicator: Any = None  # set after facade.init_communicators
    config: LoraControllerConfig = None  # type: ignore[assignment]
    lora_registry: LoRARegistry = None  # type: ignore[assignment]
    lora_update_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lora_ref_cache: Dict[str, LoRARef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lora_registry = LoRARegistry(self.server_args.lora_paths)
        if self.server_args.lora_paths is not None:
            for lora_ref in self.server_args.lora_paths:
                self.lora_ref_cache[lora_ref.lora_name] = lora_ref
