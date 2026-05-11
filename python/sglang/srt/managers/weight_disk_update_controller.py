from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

from sglang.srt.managers.pause_controller import PauseController
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.utils import TypeBasedDispatcher


@dataclass(slots=True, kw_only=True)
class WeightDiskUpdateControllerConfig:
    dp_size: int
    initial_load_format: str
    checkpoint_engine_wait_weights_before_ready: bool


@dataclass(slots=True, kw_only=True)
class WeightDiskUpdateController:
    """update_weights_from_disk endpoint + UpdateWeightFromDiskReqOutput dispatcher handler."""

    send_to_scheduler: Any
    dispatcher: TypeBasedDispatcher
    pause_controller: PauseController
    model_update_lock: RWLock
    server_args: ServerArgs
    auto_create_handle_loop: Callable[[], None]
    config: WeightDiskUpdateControllerConfig
    initial_weights_loaded: bool = True
    model_update_result: Optional[Awaitable[Any]] = None
    model_update_tmp: List[Any] = field(default_factory=list)
