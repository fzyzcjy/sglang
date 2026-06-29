# Upstream source: deepspec/modeling/dspark/gemma4/modeling.py
# Copied verbatim except:
# - `from deepspec.modeling.dspark.common import ...` -> local ref import
# - `from deepspec.modeling.dspark.markov_head import build_markov_head` -> local ref
# - `from deepspec.utils.sampling import sample_tokens` -> local ref
# This module requires `transformers` (Gemma4 model classes). It is used only
# by the GPU-tier parity test (test_dspark_model_parity.py) and is not imported
# by CPU-only tests.
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextMLP,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    apply_rotary_pos_emb as apply_gemma4_rotary_pos_emb,
)

from test.srt.speculative._dspark_reference.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    extract_context_feature,
    log_sampler_stats,
    sample_anchor_positions,
)
from test.srt.speculative._dspark_reference.markov_head import build_markov_head
from test.srt.speculative._dspark_reference.sampling import sample_tokens


class Gemma4DSparkAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_attention_heads = int(config.num_attention_heads)
        self.head_dim = int(config.global_head_dim)
        self.use_alternative_attention = bool(config.attention_k_eq_v)
        if self.use_alternative_attention:
            self.num_key_value_heads = int(config.num_global_key_value_heads)
        else:
            self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.scaling = 1.0
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.v_proj = None
        if not self.use_alternative_attention:
            self.v_proj = nn.Linear(
                config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=bool(config.attention_bias),
            )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=bool(config.attention_bias),
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            with_scale=False,
        )

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return hidden_states
        return hidden_states.repeat_interleave(self.num_key_value_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(
            bsz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        )
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        if self.use_alternative_attention:
            v_ctx = k_ctx
            v_noise = k_noise
        else:
            v_ctx = self.v_proj(target_hidden_states)
            v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz,
            ctx_len + q_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz,
            ctx_len + q_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)
        cos, sin = position_embeddings
        q = apply_gemma4_rotary_pos_emb(
            q,
            cos[:, -q_len:, :],
            sin[:, -q_len:, :],
            unsqueeze_dim=1,
        )
        k = apply_gemma4_rotary_pos_emb(k, cos, sin, unsqueeze_dim=1)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        if (
            attention_mask is not None
            and self.config._attn_implementation == "flex_attention"
        ):
            attn_output = flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                scale=self.scaling,
            )
        else:
            attn_is_causal = bool(kwargs.get("is_causal", False))
            self.is_causal = attn_is_causal
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=attn_is_causal,
                scale=self.scaling,
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), None


class Gemma4DSparkDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4DSparkAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, output_attentions, use_cache
        assert hidden_states is not None
        assert target_hidden_states is not None
        assert position_embeddings is not None
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states * self.layer_scalar


class Gemma4DSparkModel(Gemma4PreTrainedModel):
    config_class = Gemma4TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4DSparkDecoderLayer"]
    _supports_flex_attn = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.target_layer_ids = config.target_layer_ids

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", None),
            embed_scale=float(config.hidden_size) ** 0.5,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4DSparkDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(
            config,
            layer_type="full_attention",
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)

        self.markov_head = build_markov_head(config)
        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))
        self.confidence_head = None
        self.post_init()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        softcap = getattr(self.config, "final_logit_softcapping", None)
        if softcap is not None:
            softcap = float(softcap)
            assert softcap > 0.0
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        return None

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
            layer_type="full_attention",
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)


__all__ = [
    "Gemma4DSparkModel",
]
