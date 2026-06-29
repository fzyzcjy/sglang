from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dspark import DSparkDraftMixin, _DSPARK_SKIPPED_WEIGHT_PREFIXES
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.dspark_utils import parse_dspark_draft_config

logger = logging.getLogger(__name__)


def _get_gemma4_rope_params(config) -> Tuple[float, str, float]:
    """Return (rope_theta, rope_type, partial_rotary_factor) for full_attention layers."""
    rope_parameters = getattr(config, "rope_parameters", {})
    if "full_attention" in rope_parameters:
        params = dict(rope_parameters["full_attention"])
    else:
        params = {}
    rope_theta = float(params.get("rope_theta", 10000.0))
    rope_type = str(params.get("rope_type", "default"))
    partial_rotary_factor = float(params.get("partial_rotary_factor", 1.0))
    return rope_theta, rope_type, partial_rotary_factor


class Gemma4DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_parallel().tp_size)

        total_num_heads = int(config.num_attention_heads)
        head_dim = int(config.global_head_dim)
        use_alternative_attention = bool(getattr(config, "attention_k_eq_v", False))
        if use_alternative_attention:
            total_num_kv_heads = int(config.num_global_key_value_heads)
        else:
            total_num_kv_heads = int(
                getattr(config, "num_key_value_heads", total_num_heads)
            )

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.use_alternative_attention = use_alternative_attention

        assert self.total_num_heads % tp_size == 0, (
            f"Gemma4DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"Gemma4DFlashAttention requires total_num_kv_heads divisible by tp_size "
                f"when >= tp_size. total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"Gemma4DFlashAttention requires tp_size divisible by total_num_kv_heads "
                f"when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            prefix="o_proj",
        )

        self.q_norm = Gemma4RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(head_dim, eps=rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(head_dim, eps=rms_norm_eps, with_scale=False)

        rope_theta, rope_type, partial_rotary_factor = _get_gemma4_rope_params(config)
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        rope_scaling = None if rope_type in ("default", "") else {"rope_type": rope_type}
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            is_neox_style=True,
        )

        self.scaling = 1.0
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_alternative_attention:
            v = k
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        v = self.apply_v_norm(v)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        Used by the DSpark worker to materialize ctx tokens into the draft KV
        cache. When attention_k_eq_v, v == k (raw, before norms).
        """
        can_slice, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice]
                if self.qkv_proj.bias is not None
                else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_alternative_attention:
            v = k
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k

    def apply_v_norm(self, v: torch.Tensor) -> torch.Tensor:
        v_by_head = v.reshape(-1, self.head_dim)
        v_by_head = self.v_norm(v_by_head)
        return v_by_head.view_as(v)


class Gemma4DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for Gemma4DFlashMLP."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma4DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.self_attn = Gemma4DFlashAttention(config=config, layer_id=layer_id)
        self.mlp = Gemma4DFlashMLP(config=config)
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.numel() == 0:
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4DFlashDraftModel(nn.Module):
    """Gemma4-style KV-injection draft backbone (no embedding / lm_head weights).

    Mirrors DFlashDraftModel's public interface so the same DSpark worker drives
    both Qwen3 and Gemma4 draft backbones. Key Gemma4 differences vs Qwen3/DFlash:
    - GeGLU MLP (gelu_pytorch_tanh), not SiluAndMul
    - Gemma4RMSNorm with (1+weight) scaling on all hidden-state norms
    - v_norm applied per-head (apply_v_norm is a real norm, not identity)
    - scaling = 1.0 (not head_dim**-0.5)
    - head_dim = config.global_head_dim
    - attention_k_eq_v: when True, v == k (raw) before separate norms
    - 4-norm sandwich decoder layer with layer_scalar
    - full_attention (ENCODER_ONLY) for all draft layers
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [
                Gemma4DFlashDecoderLayer(config=config, layer_id=i)
                for i in range(num_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)

        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "Gemma4 DSpark target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}."
            )
        return self.hidden_norm(self.fc(target_hidden))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "Gemma4DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            hidden_states = self.norm(hidden_states)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        use_alternative_attention = bool(getattr(self.config, "attention_k_eq_v", False))
        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped = name[len("model."):]
                if stripped in params_dict:
                    return stripped
            else:
                prefixed = f"model.{name}"
                if prefixed in params_dict:
                    return prefixed
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)

                if (
                    use_alternative_attention
                    and weight_name == "k_proj"
                    and shard_id == "k"
                ):
                    for v_shard_id in ("k", "v"):
                        v_resolved = resolve_param_name(mapped_name)
                        if v_resolved is None:
                            continue
                        param = params_dict[v_resolved]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight, v_shard_id)
                    break

                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if use_alternative_attention and ".v_proj." in name:
                    continue

                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "Gemma4 DSpark fc.weight shape mismatch. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, "
                        f"hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class Gemma4DSparkModel(DSparkDraftMixin, Gemma4DFlashDraftModel):
    """Gemma4 DSpark dense draft: Gemma4 KV-injection backbone + serial Markov head."""

    pass


EntryClass = [Gemma4DSparkModel]
