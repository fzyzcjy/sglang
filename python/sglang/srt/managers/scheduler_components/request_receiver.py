from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    recv_from_tokenizer: Any
    recv_from_rpc: Any
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    ps: Any
    tp_group: Any
    tp_cpu_group: Any
    attn_tp_group: Any
    attn_tp_cpu_group: Any
    attn_cp_group: Any
    attn_cp_cpu_group: Any
    world_group: Any
    server_args: Any
    model_config: Any
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_forward_mode: Callable[[], Any]
