from __future__ import annotations  # noqa: F401

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple  # noqa: F401

import torch  # noqa: F401

from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: F401
from sglang.srt.managers.schedule_batch import Req  # noqa: F401
from sglang.srt.server_args import MIS_DELIMITER_TOKEN_ID  # noqa: F401


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLogprobComputer:
    """Pure-compute logprob accumulator helpers. Composition target on
    Scheduler (``self.logprob_computer``)."""

    server_args: Any
    model_config: Any
