from __future__ import annotations

import torch

from sglang.srt.configs.model_config import ModelConfig, is_deepseek_nsa
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import is_hip

_is_hip = is_hip()


def calculate_mla_kv_cache_dim(
    *,
    model_config: ModelConfig,
    kv_cache_dtype: torch.dtype,
    server_args: ServerArgs,
) -> int:
    is_nsa_model = is_deepseek_nsa(model_config.hf_config)
    kv_lora_rank = model_config.kv_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # default mla kv cache dim

    # For non-NSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim
    if not is_nsa_model:
        return kv_cache_dim

    # TRTLLM backend does not override kv_cache_dim for MLA kv cache
    # Assuming nsa prefill and decode backends are the same when using trtllm MLA backend,
    # since it is not compatible for trtllm and other mla attn backend due to the different
    # kv cache layout.
    if (
        server_args.nsa_prefill_backend == "trtllm"
        or server_args.nsa_decode_backend == "trtllm"
    ):
        return kv_cache_dim

    # On HIP with TileLang backend, keep the default MLA KV cache dimension.
    # FP8 attention uses the nope(512 fp8) + rope(64 fp8) layout, without extra per-block scales.
    if _is_hip and (
        server_args.nsa_prefill_backend == "tilelang"
        or server_args.nsa_decode_backend == "tilelang"
    ):
        return kv_cache_dim

    quant_block_size = NSATokenToKVPool.quant_block_size
    rope_storage_dtype = NSATokenToKVPool.rope_storage_dtype
    # Calculate override_kv_cache_dim for FP8 storage in backends that use scaled KV layout (excluding TRTLLM and HIP+TileLang).
    # kv_lora_rank + scale storage (kv_lora_rank // quant_block_size * 4 bytes) + rope dimension storage
    # Note: rope dimension is stored in original dtype (bf16), not quantized to fp8
    if kv_cache_dtype == torch.float8_e4m3fn:
        assert (
            kv_lora_rank % quant_block_size == 0
        ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"

        return (
            kv_lora_rank
            + kv_lora_rank // quant_block_size * 4
            + qk_rope_head_dim * rope_storage_dtype.itemsize
        )

    return kv_cache_dim
