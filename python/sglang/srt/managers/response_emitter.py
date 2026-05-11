from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from sglang.srt.managers.lora_controller import LoraController
from sglang.srt.managers.request_log_manager import RequestLogManager
from sglang.srt.managers.request_state import ReqState


@dataclass(slots=True, kw_only=True)
class ResponseEmitterConfig:
    incremental_streaming_output: bool
    enable_lora: bool


@dataclass(slots=True, kw_only=True)
class ResponseEmitter:
    """Drains rid_to_state[rid].out_list and yields per-request dicts to HTTP clients."""

    rid_to_state: Dict[str, ReqState]
    lora_controller: LoraController
    request_log_manager: RequestLogManager
    abort_request: Callable[..., None]
    config: ResponseEmitterConfig
