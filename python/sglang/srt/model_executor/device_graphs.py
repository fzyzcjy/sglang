from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.platforms import current_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


def init_device_graphs(
    *,
    is_generation: bool,
    server_args: ServerArgs,
    device: str,
    gpu_id: int,
    make_graph_runner: Callable[[], Any],
) -> tuple[Any, float]:
    """Capture device graphs.

    Returns (graph_runner, graph_mem_usage). Both are None / 0 on the bail
    paths so the caller's tuple-unpack writeback still works.

    `make_graph_runner` is a 0-arg callable supplied by the caller — it owns
    the ``GraphRunnerCls(self)`` / ``graph_runners[device](self)`` selection
    + ctor (which still need a ModelRunner ref). Keeping it in the caller
    avoids an R4 concession kwarg here.
    """
    if not is_generation:
        # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
        return None, 0

    if server_args.model_impl.lower() == ModelImpl.MINDSPORE:
        return None, 0

    if device != "cpu" and server_args.disable_cuda_graph:
        return None, 0

    if device == "cpu" and not server_args.enable_torch_compile:
        return None, 0

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(device, gpu_id)
    graph_backend = defaultdict(
        lambda: f"{current_platform.device_name} graph",
        {
            "cuda": "cuda graph",
            "musa": "cuda graph",
            "cpu": "cpu graph",
            "npu": "npu graph",
        },
    )
    logger.info(
        f"Capture {graph_backend[device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
    )
    graph_runner = make_graph_runner()

    after_mem = get_available_gpu_memory(device, gpu_id)
    graph_mem_usage = before_mem - after_mem
    logger.info(
        f"Capture {graph_backend[device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
        f"mem usage={graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
    )
    return graph_runner, graph_mem_usage
