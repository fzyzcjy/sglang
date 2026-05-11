from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.request_state import ReqState


@dataclass(frozen=True, slots=True)
class ScoreResult:
    scores: List[List[float]]
    prompt_tokens: int = 0
    pooled_hidden_states: Optional[List[Optional[torch.Tensor]]] = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreRequestHandlerConfig:
    is_generation: bool
    enable_mis: bool
    model_config: ModelConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreRequestHandler:
    tokenizer: Optional[Any]
    rid_to_state: Dict[str, ReqState]
    generate_request: Callable[..., AsyncIterator[dict]]
    config: ScoreRequestHandlerConfig
