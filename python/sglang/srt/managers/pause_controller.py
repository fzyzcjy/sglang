from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.request_state import ReqState
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.utils import TypeBasedDispatcher


@dataclass(slots=True, kw_only=True)
class PauseControllerConfig:
    enable_metrics: bool
    skip_tokenizer_init: bool
    weight_version: Optional[str]


@dataclass(slots=True, kw_only=True)
class PauseController:
    """Pause / resume / abort state machine + AbortReq dispatcher handler."""

    send_to_scheduler: Any
    dispatcher: TypeBasedDispatcher
    rid_to_state: Dict[str, ReqState]
    model_update_lock: RWLock
    metrics_collector: Optional[Any]
    tokenizer: Optional[Any]
    config: PauseControllerConfig
    is_pause: bool = False
    is_pause_cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    def __post_init__(self) -> None:
        # Forward to the still-on-TM staticmethod _handle_abort_req via lambda.
        # The next commit cuts the method here and flips this to a direct
        # ``self._handle_abort_req`` reference.
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        # TypeBasedDispatcher has no public register(); poke private _mapping.
        self.dispatcher._mapping[AbortReq] = (
            lambda x: TokenizerManager._handle_abort_req(self, x)
        )
