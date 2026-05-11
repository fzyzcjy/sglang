from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict

from sglang.utils import TypeBasedDispatcher


@dataclass(slots=True, kw_only=True)
class SessionControllerConfig:
    enable_streaming_session: bool


@dataclass(slots=True, kw_only=True)
class SessionController:
    """open_session / close_session endpoints + OpenSessionReqOutput dispatcher handler."""

    send_to_scheduler: Any
    dispatcher: TypeBasedDispatcher
    config: SessionControllerConfig
    session_futures: Dict[str, asyncio.Future] = field(default_factory=dict)
