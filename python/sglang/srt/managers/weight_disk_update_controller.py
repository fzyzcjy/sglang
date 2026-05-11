from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqOutput
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

    def __post_init__(self) -> None:
        if self.config.checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False
        # Lambda forwarder: until the move commit relocates the @staticmethod
        # body onto this class, the actual handler lives on TokenizerManager.
        # Late import avoids the TokenizerManager <-> WeightDiskUpdateController
        # import cycle. Move-time flips this to a direct method reference.
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        self.dispatcher._mapping[UpdateWeightFromDiskReqOutput] = (
            lambda x: TokenizerManager._handle_update_weights_from_disk_req_output(
                self, x
            )
        )
