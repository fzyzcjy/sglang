from __future__ import annotations  # noqa: F401

from dataclasses import dataclass
from http import HTTPStatus  # noqa: F401
from typing import Any, Callable, List, Optional, Union  # noqa: F401

import zmq  # noqa: F401
from torch.distributed import barrier  # noqa: F401

from sglang.srt.disaggregation.utils import prepare_abort  # noqa: F401
from sglang.srt.managers.io_struct import (  # noqa: F401
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import (  # noqa: F401
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.utils import broadcast_pyobj, point_to_point_pyobj  # noqa: F401


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    """Wire-level request receiver: pulls ``recv_req`` lists from zmq /
    pipeline upstream, applies recv_skipper / input_blocker guards, broadcasts
    across TP/DP/CP groups, runs MM-receiver pre-processing, and unwraps shm
    features. Owns no mutable state."""

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
