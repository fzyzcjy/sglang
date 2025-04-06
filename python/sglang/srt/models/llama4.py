# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama4.py
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import Llama4TextConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    dp_gather_partial,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    tp_all_gather,
    tp_reduce_scatter,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)
from sglang.srt.utils import DeepEPMode, add_prefix, make_layers
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class Llama4MoE(nn.Module):

    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        router_scores_aK, router_indices_aK = torch.topk(gating_output, topk, dim=-1)
        router_scores_aK = torch.sigmoid(router_scores_aK.float()).to(
            hidden_states.dtype
        )
        return (
            router_scores_aK.view(-1).reshape(router_scores_aK.shape),
            router_indices_aK.to(torch.int32),
        )

    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok

        intermediate_size_moe = config.intermediate_size
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("router", prefix),
        )

        MoEImpl = (
            DeepEPMoE
            if global_server_args_dict["enable_deepep_moe"]
            else (EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE)
        )
        self.experts = MoEImpl(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            custom_routing_function=Llama4MoE.custom_routing_function,
            intermediate_size=intermediate_size_moe,
            renormalize=False,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            **(
                dict(deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]])
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
        )

        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            prefix=add_prefix("shared_expert", prefix),
            reduce_results=False,  # We need to do scatter before reduce
            # disable tp for shared experts when enable deepep moe
            **(
                dict(tp_rank=0, tp_size=1)
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
        )

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok

            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=config.num_local_experts,
                num_local_experts=config.num_local_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,
                return_recv_hook=True,
            )

    def forward(
        self, hidden_states: torch.Tensor, forward_mode: Optional[ForwardMode] = None
    ) -> torch.Tensor:
        if not global_server_args_dict["enable_deepep_moe"]:
            return self.forward_normal(hidden_states)
        else:
            return self.forward_deepep(hidden_states, forward_mode)

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # router_scores: [num_tokens, num_experts]
        router_logits, _ = self.router(hidden_states)
        shared_out = self.shared_expert(hidden_states)
        routed_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        out_aD = routed_out + shared_out

        if self.tp_size > 1:
            out_aD = tensor_model_parallel_all_reduce(out_aD)

        return out_aD

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_mode: ForwardMode
    ) -> torch.Tensor:
        shared_output = None
        topk_idx = torch.full(
            (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
        )
        topk_weights = torch.empty(
            (0, self.top_k), dtype=torch.float32, device=hidden_states.device
        )
        if (
            forward_mode is not None
            and not forward_mode.is_idle()
            and hidden_states.shape[0] > 0
        ):
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.router(hidden_states)
            shared_output = self.shared_expert(hidden_states)
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=False,
                renormalize=False,
                topk_group=None,
                num_expert_group=None,
                correction_bias=None,
            )
        if self.ep_size > 1:
            (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states,
                topk_idx,
                topk_weights,
                self.num_experts,
                forward_mode=forward_mode,
            )
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            forward_mode=forward_mode,
        )
        if self.ep_size > 1:
            final_hidden_states = self.deepep_dispatcher.combine(
                final_hidden_states,
                topk_idx,
                topk_weights,
                forward_mode,
            )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states


class Llama4Attention(nn.Module):

    def __init__(
        self,
        config: Llama4TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.no_rope_layers = config.no_rope_layers
        self.nope = self.no_rope_layers[self.layer_id] == 0
        self.use_qk_norm = config.use_qk_norm and not self.nope
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_temperature_tuning = (
            getattr(config, "attn_temperature_tuning", False) and self.nope
        )
        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        self.q_norm = (
            RMSNorm(
                hidden_size=self.q_size,
                eps=config.rms_norm_eps,
            )
            if self.use_qk_norm
            else None
        )
        self.k_norm = (
            RMSNorm(
                hidden_size=self.kv_size,
                eps=config.rms_norm_eps,
            )
            if self.use_qk_norm
            else None
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            reduce_results=reduce_results,
        )
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_theta),
                rope_scaling=rope_scaling if rope_scaling != "default" else None,
                is_neox_style=is_neox_style,
            )
            if not self.nope
            else None
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0

        return attn_scale.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        if self.attn_temperature_tuning and self.nope:
            attn_scale = self._get_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        if self.q_norm is not None:
            # TODO: support float
            q = self.q_norm(q.bfloat16().contiguous()).to(q.dtype)
        if self.k_norm is not None:
            k = self.k_norm(k.bfloat16().contiguous()).to(k.dtype)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.dp_size = get_attention_dp_size()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        def compute_is_moe_layer(l: int):
            return (l + 1) % config.interleave_moe_layer_step == 0

        def compute_input_is_scattered(l: int):
            return (
                l > 0
                and compute_is_moe_layer(l)
                and compute_is_moe_layer(l - 1)
                and global_server_args_dict["enable_deepep_moe"]
            )

        self.self_attn = Llama4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
        )
        if compute_is_moe_layer(layer_id):
            self.feed_forward = Llama4MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
            self.is_sparse = True
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
            self.is_sparse = False
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1
        self.input_is_scattered = compute_input_is_scattered(layer_id)
        self.output_is_scattered = (
            not self.is_last_layer
        ) and compute_input_is_scattered(layer_id + 1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if global_server_args_dict["enable_deepep_moe"] and self.is_sparse:
            return self.forward_deepep(
                positions, hidden_states, forward_batch, residual
            )
        else:
            return self.forward_normal(
                positions, hidden_states, forward_batch, residual
            )

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        assert not (self.attn_tp_size != 1 and self.input_is_scattered)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Gather
        if get_tensor_model_parallel_world_size() > 1:
            # all gather and all reduce
            if self.dp_size != 1:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                hidden_states, local_hidden_states = (
                    forward_batch.gathered_buffer,
                    hidden_states,
                )
                dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
                dp_scatter(residual, hidden_states, forward_batch)
                hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        # Fully Connected
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual

    def forward_deepep(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:

        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.attn_tp_size != 1 and self.input_is_scattered:
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        if self.attn_tp_size != 1:
            if self.input_is_scattered:
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                tp_reduce_scatter(hidden_states, tensor_list)
                if hidden_states.shape[0] != 0:
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
            else:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                tp_reduce_scatter(hidden_states, tensor_list)
                residual = hidden_states
                if hidden_states.shape[0] != 0:
                    hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
        hidden_states = self.feed_forward(hidden_states, forward_batch.forward_mode)

        if (not self.output_is_scattered) and self.attn_tp_size != 1:
            hidden_states += residual
            residual = None
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        return hidden_states, residual


class Llama4Model(LlamaModel):
    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.num_experts = getattr(config, "num_local_experts", 0)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Llama4DecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix="model.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def load_moe_expert_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: Dict[str, nn.Parameter],
        loaded_params: Set[str],
        expert_params_mapping: List[Tuple[str, str, int, str]],
        fused: bool = True,
    ) -> bool:
        expert_param_loaded = False
        if "experts.gate_up_proj" in name:
            loaded_weight = loaded_weight.chunk(2, dim=-1)
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            new_loaded_weight = loaded_weight
            if fused:
                e_str, _, proj_str, _ = weight_name.split(".")
                weight_name = f"{e_str}.{proj_str}"
                param_name = f"{param_name}weight"
            if weight_name not in name:
                continue
            full_param_name = name.replace(weight_name, param_name)
            if full_param_name not in params_dict:
                continue
            param = params_dict[full_param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if fused:
                if "w13" in full_param_name:
                    shard_idx = 0 if shard_id == "w1" else 1
                    new_loaded_weight = new_loaded_weight[shard_idx]
                new_loaded_weight = new_loaded_weight.transpose(-1, -2)
                layer_idx = int(name.split(".")[2])
                # EP mapping
                expert_map = self.layers[layer_idx].feed_forward.experts.expert_map
                if expert_map is not None:
                    local_expert_indices = (
                        (expert_map != -1)
                        .nonzero()
                        .flatten()
                        .to(new_loaded_weight.device)
                    )
                    new_loaded_weight = new_loaded_weight[local_expert_indices]
                    expert_id = local_expert_indices[0].item()
            else:
                # TODO: add EP support for non fused weights
                pass
            weight_loader(
                param,
                new_loaded_weight,
                full_param_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )

            loaded_params.add(full_param_name)
            expert_param_loaded = True
        return expert_param_loaded


class Llama4ForCausalLM(LlamaForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)

    def _init_model(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return Llama4Model(config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        return
        # Process weights for rotary embeddings
        processed_weights = []
        for name, loaded_weight in weights:
            name, loaded_weight = self.permute_qk_weight_for_rotary(name, loaded_weight)
            processed_weights.append((name, loaded_weight))

        # Load weights using the same pattern as in LlamaForCausalLM
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        # Setup MoE expert parameters mapping
        fused_experts_params = False
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.model.num_experts,
        )
        expert_params_mapping_fused = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="gate_up_proj",
            num_experts=1,
        )

        for name, loaded_weight in processed_weights:
            # Handle MoE expert weights
            if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                fused_experts_params = True
                expert_params_mapping = expert_params_mapping_fused

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Handle FP8 kv-scale remapping
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # Handle quantization scales for KV cache
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name or "experts" in name:
                    continue
                full_param_name = name.replace(weight_name, param_name)
                if full_param_name not in params_dict:
                    continue
                param = params_dict[full_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, full_param_name, shard_id=shard_id)
                loaded_params.add(full_param_name)
                break
            else:
                # Handle MoE expert weights
                if "experts" in name:
                    moe_loaded = self.model.load_moe_expert_weights(
                        name,
                        loaded_weight,
                        params_dict,
                        loaded_params,
                        expert_params_mapping,
                        fused=fused_experts_params,
                    )
                    if moe_loaded:
                        continue

                # Handle regular weights
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

        return loaded_params

    def permute_qk_weight_for_rotary(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        modules = name.split(".")

        # rotary embeds should be sliced
        if ("wk" in modules or "k_proj" in modules) and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight, self.config.num_key_value_heads)
        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight, self.config.num_attention_heads)

        return name, loaded_weight


EntryClass = [Llama4ForCausalLM]
