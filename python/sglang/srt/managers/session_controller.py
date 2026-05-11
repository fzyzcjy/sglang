from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from sglang.srt.managers.io_struct import OpenSessionReqOutput
from sglang.utils import TypeBasedDispatcher


@dataclass(slots=True, kw_only=True)
class SessionControllerConfig:
    enable_streaming_session: bool


@dataclass(slots=True, kw_only=True)
class SessionController:
    """open_session / close_session endpoints + OpenSessionReqOutput dispatcher handler."""

    send_to_scheduler: Any
    dispatcher: TypeBasedDispatcher
    auto_create_handle_loop: Callable[[], None]
    config: SessionControllerConfig
    session_futures: Dict[str, asyncio.Future] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Lambda forwarder: during prep the handler still lives on TokenizerManager
        # as a @staticmethod with ``self: "SessionController"`` typing. The
        # follow-up -move commit cuts the method into this class and flips this
        # registration to a direct method reference.
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        self.dispatcher._mapping[OpenSessionReqOutput] = (
            lambda recv_obj: TokenizerManager._handle_open_session_req_output(
                self, recv_obj
            )
        )
