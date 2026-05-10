from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from torch import nn

from sglang.srt.configs.model_config import ModelConfig, ModelImpl
from sglang.srt.platforms import current_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0

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


@dataclass(frozen=True, slots=True, kw_only=True)
class PiecewiseCudaGraphsResult:
    piecewise_cuda_graph_runner: Any
    attention_layers: Optional[list[Any]]
    moe_layers: Optional[list[Any]]
    moe_fusions: Optional[list[Any]]


_PIECEWISE_BAIL = PiecewiseCudaGraphsResult(
    piecewise_cuda_graph_runner=None,
    attention_layers=None,
    moe_layers=None,
    moe_fusions=None,
)


def init_piecewise_cuda_graphs(
    *,
    server_args: ServerArgs,
    is_draft_worker: bool,
    model: nn.Module,
    model_config: ModelConfig,
    device: str,
    gpu_id: int,
    resolve_language_model: Callable[[nn.Module], nn.Module],
    make_runner: Callable[[], Any],
) -> PiecewiseCudaGraphsResult:
    """Initialize piecewise CUDA graph runner."""
    if server_args.disable_piecewise_cuda_graph:
        logger.info(
            "Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set"
        )
        return _PIECEWISE_BAIL

    # Draft models use decode CUDA graphs, not PCG
    if is_draft_worker:
        return _PIECEWISE_BAIL

    # Disable piecewise CUDA graph for non-language models
    if not hasattr(model, "model"):
        logger.warning(
            "Disable piecewise CUDA graph because the model is not a language model"
        )
        return _PIECEWISE_BAIL

    # Disable piecewise CUDA graph for non capture size
    if not server_args.piecewise_cuda_graph_tokens:
        logger.warning(
            "Disable piecewise CUDA graph because the capture size is not set"
        )
        return _PIECEWISE_BAIL

    # Collect attention layers and moe layers from the model
    model.model = resolve_language_model(model)
    language_model = getattr(model, "language_model", model)

    # Resolve model with layers: handle CausalLM wrapper (.model.layers) and direct TextModel (.layers)
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        layer_model = language_model.model
    elif hasattr(language_model, "layers"):
        layer_model = language_model
    else:
        logger.warning(
            "Disable piecewise CUDA graph because the model does not have a 'layers' attribute"
        )
        return _PIECEWISE_BAIL

    attention_layers: list[Any] = []
    moe_layers: list[Any] = []
    moe_fusions: list[Any] = []
    for layer in layer_model.layers:
        attn_layer = None
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "attn"):
                attn_layer = layer.self_attn.attn
            elif hasattr(layer.self_attn, "attn_mqa"):
                # For DeepSeek model
                attn_layer = layer.self_attn.attn_mqa
        # For hybrid model
        elif hasattr(layer, "attn"):
            attn_layer = layer.attn
        elif hasattr(layer, "linear_attn"):
            if hasattr(layer.linear_attn, "attn"):
                attn_layer = layer.linear_attn.attn
            else:
                attn_layer = layer.linear_attn
        # For InternVL model
        elif hasattr(layer, "attention"):
            if hasattr(layer.attention, "attn"):
                attn_layer = layer.attention.attn
        # For NemotronH and similar hybrid models using 'mixer' attribute
        elif hasattr(layer, "mixer"):
            if hasattr(layer.mixer, "attn"):
                attn_layer = layer.mixer.attn
            elif hasattr(layer, "_forward_mamba"):
                # Mamba layer with split op support - store the layer itself
                attn_layer = layer

        if attn_layer is not None:
            attention_layers.append(attn_layer)
        elif hasattr(layer, "mixer"):
            attention_layers.append(None)

        moe_block = None
        moe_fusion = None
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            moe_block = layer.mlp.experts
            moe_fusion = layer.mlp
        if hasattr(layer, "block_sparse_moe") and hasattr(
            layer.block_sparse_moe, "experts"
        ):
            moe_block = layer.block_sparse_moe.experts
            moe_fusion = layer.block_sparse_moe
        if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
            moe_block = layer.moe.experts
            moe_fusion = layer.moe
        # For NemotronH MoE layers using 'mixer' attribute
        if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
            moe_block = layer.mixer.experts
            moe_fusion = layer.mixer
        moe_layers.append(moe_block)
        moe_fusions.append(moe_fusion)

    if len(attention_layers) < model_config.num_hidden_layers:
        # TODO(yuwei): support Non-Standard GQA
        log_info_on_rank0(
            logger,
            "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
        )
        return PiecewiseCudaGraphsResult(
            piecewise_cuda_graph_runner=None,
            attention_layers=attention_layers,
            moe_layers=moe_layers,
            moe_fusions=moe_fusions,
        )

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(device, gpu_id)
    logger.info(f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB")

    piecewise_cuda_graph_runner = make_runner()

    after_mem = get_available_gpu_memory(device, gpu_id)
    mem_usage = before_mem - after_mem
    logger.info(
        f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
        f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
    )
    return PiecewiseCudaGraphsResult(
        piecewise_cuda_graph_runner=piecewise_cuda_graph_runner,
        attention_layers=attention_layers,
        moe_layers=moe_layers,
        moe_fusions=moe_fusions,
    )
