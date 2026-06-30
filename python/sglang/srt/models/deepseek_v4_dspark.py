# Adapted from the DeepSeek-V4-Flash DSpark reference (DSparkBlock / DSparkAttention /
# DSparkMarkovHead / Transformer.forward_spec) but implemented with SGLang primitives.
# The V4 DSpark draft is a block draft with its own sliding-window MLA KV (fed from the
# target hidden states), a target-hidden projection (main_proj/main_norm), and a serial
# Markov head. It reuses the target model's token embedding / lm_head for the base logits.
# When the draft config enables it, the draft also carries an opt-in confidence head that
# consumes the post-hc_head PRE-norm draft hidden (reference model.py:862/873: x=hc_head(x),
# confidence=confidence_head(x, markov_embed); the norm only feeds the logits).

from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v4 import (
    DEEPSEEK_V4_STACKED_PARAMS_MAPPING,
    DeepseekV4DecoderLayer,
    hc_head_torch,
    make_hc_head_params,
)
from sglang.srt.models.dspark import DSparkConfidenceHead
from sglang.srt.speculative.dspark_utils import parse_dspark_draft_config
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

# A per-step sampler: (step_logits [bs, vocab], step_idx) -> sampled tokens [bs].
StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place rotary embedding, mirroring the DSpark reference (model.py:238)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def build_dspark_topk_idxs(
    *, window_size: int, bsz: int, block_size: int, start_pos: int
) -> torch.Tensor:
    """Non-causal full-block draft index layout (window positions + block positions).

    Mirrors the reference ``get_dspark_topk_idxs`` (model.py:744): every one of the
    ``block_size`` draft queries attends the full sliding window (the target-hidden KV)
    plus the whole draft block (non-causal), so the index matrix is shared across rows.
    The corresponding paged ``kv_indices``/``kv_indptr`` builder lives in the worker
    backend (plan §2.b); this returns the dense per-row index matrix used by the
    self-contained reference attention path.
    """
    if start_pos <= 0:
        raise ValueError(
            f"DSpark draft attention requires start_pos > 0, got {start_pos}."
        )
    window = torch.arange(min(window_size, start_pos + 1))
    block = window_size + torch.arange(block_size)
    matrix = torch.cat([window, block])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(nn.Module):
    """Sliding-window sparse MLA for the V4 DSpark draft (compress_ratio == 0).

    Mirrors the reference ``DSparkAttention`` (model.py:750). The draft KV is the
    target hidden state (``main_x``) projected through ``wkv``/``kv_norm``/rope and
    written into a per-request sliding-window ring; the draft queries attend that ring
    plus the current full draft block (non-causal). Kept separate from the base
    ``MQALayer`` (RadixAttention + paged pool + compressor/indexer): this draft path is
    compress_ratio == 0, non-causal full-block, and owns its own KV ring (the production
    paged pool is wired by the worker in Phase b).
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        assert self.head_dim == config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.window_size = int(
            getattr(config, "sliding_window", None) or config.window_size
        )
        self.eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5

        self.compress_ratio = 0
        assert (
            self.compress_ratio == 0
        ), "DSpark draft attention requires compress_ratio == 0."

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = ReplicatedLinear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_a", prefix),
        )
        self.wkv = ReplicatedLinear(
            self.dim,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_a", prefix),
            params_dtype=torch.bfloat16,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_b", prefix),
        )

        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis
        from sglang.srt.utils.hf_transformers.common import get_rope_config

        rope_theta, rope_scaling = get_rope_config(config)
        rope_scaling = rope_scaling or {}
        original_seq_len = rope_scaling.get("original_max_position_embeddings", 0)
        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=0 if self.compress_ratio == 0 else original_seq_len,
            base=rope_theta,
            factor=rope_scaling.get("factor", 1.0),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        self._kv_ring: Optional[torch.Tensor] = None

    def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:
        kv, _ = self.wkv(x)
        return kv

    def apply_kv_norm_rope(
        self, kv: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        rd = self.rope_head_dim
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        return kv

    def _ensure_ring(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._kv_ring is None
            or self._kv_ring.shape[0] < bsz
            or self._kv_ring.device != device
        ):
            self._kv_ring = torch.zeros(
                bsz, self.window_size, self.head_dim, device=device, dtype=dtype
            )

    def write_window_kv(self, main_kv: torch.Tensor, start_pos: int) -> None:
        """Write the prefill window of target-hidden KV into the ring (model.py:763)."""
        bsz, seqlen, _ = main_kv.shape
        self._ensure_ring(bsz, main_kv.device, main_kv.dtype)
        win = self.window_size
        if seqlen <= win:
            self._kv_ring[:bsz, :seqlen] = main_kv
        else:
            cutoff = seqlen % win
            tail = main_kv[:, -win:]
            self._kv_ring[:bsz, cutoff:win], self._kv_ring[:bsz, :cutoff] = tail.split(
                [win - cutoff, cutoff], dim=1
            )

    def write_step_kv(self, main_kv_step: torch.Tensor, start_pos: int) -> None:
        """Write one accepted target-hidden KV into the ring slot (model.py:783)."""
        bsz = main_kv_step.shape[0]
        self._ensure_ring(bsz, main_kv_step.device, main_kv_step.dtype)
        self._kv_ring[:bsz, start_pos % self.window_size] = main_kv_step

    def project_and_store_main_kv(
        self, main_x: torch.Tensor, start_pos: int, is_prefill: bool
    ) -> torch.Tensor:
        """Project the target hidden into draft KV and write it into the ring.

        Shared by ``forward`` and the model-level ``inject_target_hidden`` (plan §2.g):
        ``wkv`` proj -> ``kv_norm`` + rope -> sliding-window ring store (full window on
        prefill, single accepted slot on decode). Returns the projected ``main_kv``.
        """
        seqlen = main_x.shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        main_kv = self.apply_kv_norm_rope(self.kv_proj_only(main_x), freqs_cis)
        if is_prefill:
            self.write_window_kv(main_kv, start_pos)
        else:
            self.write_step_kv(main_kv.squeeze(1), start_pos)
        return main_kv

    def _sparse_attn(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-torch reference for the reference ``sparse_attn`` (model.py:533/785).

        ``q``: [b, s, h, d]; ``kv``: [b, t, d] (single latent head, MQA-broadcast);
        ``topk_idxs``: [b, s, k] gather indices into ``kv`` (-1 marks padding). The
        production sparse kernel is wired by the worker backend (plan §2.b/§2.e); this
        keeps the model self-contained for parity (plan §1).
        """
        bsz, seqlen, n_heads, d = q.shape
        kv_gathered = kv.unsqueeze(1).expand(bsz, seqlen, kv.shape[1], d)
        idx = topk_idxs.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, d).long()
        sel = torch.gather(kv_gathered, 2, idx)
        scores = torch.einsum("bshd,bskd->bshk", q.float(), sel.float())
        scores = scores * self.softmax_scale
        mask = (topk_idxs < 0).unsqueeze(2)
        scores = scores.masked_fill(mask, float("-inf"))
        sink = self.attn_sink.view(1, 1, n_heads, 1).expand(bsz, seqlen, n_heads, 1)
        scores = torch.cat([scores, sink], dim=-1)
        probs = scores.softmax(dim=-1)[..., :-1]
        out = torch.einsum("bshk,bskd->bshd", probs.to(sel.dtype), sel)
        return out

    def forward(
        self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor
    ) -> torch.Tensor:
        rd = self.rope_head_dim
        seqlen = main_x.shape[1]

        if start_pos == 0:
            self.project_and_store_main_kv(main_x, start_pos, is_prefill=True)
            return x

        self.project_and_store_main_kv(main_x, start_pos, is_prefill=False)

        bsz, block_size, _ = x.shape
        freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]

        q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q, _ = self.wq_b(q)
        q = q.view(bsz, block_size, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(
            q.dtype
        )
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.apply_kv_norm_rope(self.kv_proj_only(x), freqs_cis)

        topk_idxs = build_dspark_topk_idxs(
            window_size=self.window_size,
            bsz=bsz,
            block_size=block_size,
            start_pos=start_pos,
        ).to(x.device)
        kv = torch.cat([self._kv_ring[:bsz], kv], dim=1)
        o = self._sparse_attn(q, kv, topk_idxs)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float()).to(x.dtype)
        x, _ = self.wo_b(o.flatten(2))
        return x


class DSparkV4MarkovHead(nn.Module):
    """V4 DSpark Markov head: full-logits bias = w2(w1(prev_token)), w2 in fp32.

    Mirrors the reference ``DSparkMarkovHead`` (model.py:795): ``markov_w1`` is a full
    (non-vocab-parallel) embedding, ``markov_w2`` a full fp32 head producing the
    whole-vocab logits bias. Exposes the same ``apply_step_logits``/``sample_block``
    interface as the dense ``VanillaMarkov`` so the shared serial-Markov loop can reuse it.
    """

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"DSparkV4MarkovHead requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = VocabParallelEmbedding(
            self.vocab_size, self.markov_rank, enable_tp=False
        )
        self.markov_w2 = nn.Linear(
            self.markov_rank, self.vocab_size, bias=False, dtype=torch.float32
        )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return F.linear(latent_states.float(), self.markov_w2.weight)

    def compute_step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.get_prev_embeddings(token_ids)
        logits = self.project_bias(embed)
        return logits, embed

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty, base_logits

        sampled_tokens: List[torch.Tensor] = []
        corrected_logits: List[torch.Tensor] = []
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :],
                token_ids=prev_tokens,
                hidden_states=None,
            )
            next_tokens = sampler(step_logits, step_idx)
            sampled_tokens.append(next_tokens)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


def _greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return step_logits.argmax(dim=-1)


def build_dspark_v4_confidence_head(
    *, config: DeepSeekV4Config, markov_rank: int
) -> Optional[DSparkConfidenceHead]:
    """Build the V4 DSpark confidence head when the draft config enables it.

    Mirrors the dense ``build_confidence_head`` gating and interface (the head emits a
    RAW accept-rate logit; ``with_markov`` concatenates the per-step markov_embed). The
    V4 hidden size comes from ``config.hidden_size`` and the rank from the parsed draft
    config; returns ``None`` for the static-verify checkpoints that omit the head.
    """
    if not bool(getattr(config, "enable_confidence_head", False)):
        return None
    with_markov = bool(getattr(config, "confidence_head_with_markov", markov_rank > 0))
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark V4 confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=int(config.hidden_size),
        markov_rank=int(markov_rank),
        with_markov=with_markov,
    )


class DSparkV4Stage(DeepseekV4DecoderLayer):
    """One DSpark MTP stage (reference ``DSparkBlock`` model.py:818).

    Subclasses ``DeepseekV4DecoderLayer`` to reuse its MoE, mHC mixing params, norms,
    and the ``hc_pre``/``hc_post`` math; only the attention submodule (``DSparkAttention``
    via ``_build_self_attn``) and the forward contract differ. The base ``hc_pre``/
    ``hc_post`` operate on token-flattened ``[N, hc, d]``; the draft bridges its 4D
    ``[b, s, hc, d]`` reference tensors to that contract. On GPU these reuse the target
    model's fused mHC kernels (same math); without the fused env flags they fall back to
    the identical pure-torch path.
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        stage_id: int,
        num_stages: int,
        num_target_layers: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.stage_id = stage_id
        self.dim = config.hidden_size

        if stage_id == 0:
            if num_target_layers <= 0:
                raise ValueError(
                    "DSpark needs target layers for the target-hidden projection."
                )
            self.main_proj = ReplicatedLinear(
                config.hidden_size * num_target_layers,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("main_proj", prefix),
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if stage_id == num_stages - 1:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            (
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
            ) = make_hc_head_params(config.hc_mult, config.hidden_size)

    def _build_self_attn(
        self,
        *,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[List[torch.cuda.Stream]],
        compress_ratio_override: Optional[int],
    ) -> nn.Module:
        del alt_streams, compress_ratio_override
        return DSparkAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _hc_pre_block(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bridge the 4D draft tensor to the base ``hc_pre`` (token-flattened ``[N, hc, d]``)."""
        bsz, block_size = x.shape[:2]
        y, post, comb, _ = self.hc_pre(
            x.reshape(bsz * block_size, self.hc_mult, self.dim),
            hc_fn,
            hc_scale,
            hc_base,
        )
        return (
            y.reshape(bsz, block_size, self.dim),
            post.reshape(bsz, block_size, self.hc_mult),
            comb.reshape(bsz, block_size, self.hc_mult, self.hc_mult),
        )

    def _hc_post_block(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Bridge the 4D draft tensor to the base ``hc_post`` (token-flattened ``[N, hc, d]``)."""
        bsz, block_size = x.shape[:2]
        y = self.hc_post(
            x.reshape(bsz * block_size, self.dim),
            residual.reshape(bsz * block_size, self.hc_mult, self.dim),
            post.reshape(bsz * block_size, self.hc_mult),
            comb.reshape(bsz * block_size, self.hc_mult, self.hc_mult),
        )
        return y.reshape(bsz, block_size, self.hc_mult, self.dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
    ) -> torch.Tensor:
        if start_pos == 0:
            # Prefill only fills the draft KV window; the full block runs on decode.
            return self.self_attn(x, start_pos, main_x)

        residual = x
        x, post, comb = self._hc_pre_block(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        x = self.self_attn(x, start_pos, main_x)
        x = self._hc_post_block(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre_block(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self._run_ffn(x, input_ids)
        x = self._hc_post_block(x, residual, post, comb)
        return x

    def _run_ffn(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        y = self.mlp(x)
        return y.view(shape)


class DeepseekV4ForCausalLMDSpark(nn.Module):
    """V4 DSpark draft model: block draft with target-hidden-fed sliding-window MLA KV.

    Owns the target-hidden projection (main_proj/main_norm), the DSpark MTP stages, the
    Markov head, an opt-in confidence head, and the head finish (hc_head + the target's
    shared lm_head). The token embedding and lm_head are supplied by the target model and
    attached via ``attach_shared_modules`` (the worker wires them in Phase b).

    When the confidence head is enabled the draft computes the confidence inside its own
    forward at the correct tap (post-hc_head, PRE-norm draft hidden) and stashes it on
    ``self._last_confidence``; the worker relay reads it via ``last_confidence`` (the dense
    relay's post-norm ``draft_hidden`` would be the wrong tap for V4, reference
    model.py:862/873).
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark V4 draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(
            dspark_config.resolve_gamma(default=int(config.num_hidden_layers))
        )
        self.block_size = self.gamma
        self.num_stages = int(getattr(config, "num_nextn_predict_layers", 1) or 1)

        target_num_layers = (
            int(dspark_config.num_target_layers)
            if dspark_config.num_target_layers is not None
            else int(getattr(config, "num_hidden_layers", 1))
        )
        if dspark_config.target_layer_ids is not None:
            self.num_target_features = len(dspark_config.target_layer_ids)
        else:
            self.num_target_features = target_num_layers
        self.noise_token_id = int(getattr(config, "dspark_noise_token_id", 0))
        self.temperature = float(getattr(config, "temperature", 1.0))

        base_layer_id = int(config.num_hidden_layers)
        self.stages = nn.ModuleList(
            [
                DSparkV4Stage(
                    config=config,
                    layer_id=base_layer_id + stage_id,
                    stage_id=stage_id,
                    num_stages=self.num_stages,
                    num_target_layers=self.num_target_features,
                    quant_config=quant_config,
                    prefix=add_prefix(f"stages.{stage_id}", prefix),
                )
                for stage_id in range(self.num_stages)
            ]
        )
        self.markov_head = DSparkV4MarkovHead(
            vocab_size=int(config.vocab_size),
            markov_rank=int(dspark_config.markov_rank),
        )
        self.confidence_head = build_dspark_v4_confidence_head(
            config=config, markov_rank=int(dspark_config.markov_rank)
        )
        self.hc_mult = int(config.hc_mult)
        self.norm_eps = float(config.rms_norm_eps)
        self.hc_eps = float(config.hc_eps)

        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self._last_confidence: Optional[torch.Tensor] = None

    @property
    def enable_confidence_head(self) -> bool:
        return self.confidence_head is not None

    def last_confidence(self) -> Optional[torch.Tensor]:
        """Confidence stashed by the most recent ``forward_head`` (worker relay, Phase b).

        Returns the post-STS confidence ``[bs, gamma]`` in ``(0, 1)`` computed from the
        post-hc_head PRE-norm tap, or ``None`` when the confidence head is disabled.
        """
        return self._last_confidence

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        """Attach the target model's shared embedding and lm_head (worker, Phase b)."""
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def project_target_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hc-mean features -> draft hidden (main_proj)."""
        stage0 = self.stages[0]
        projected, _ = stage0.main_proj(main_hidden)
        return stage0.main_norm(projected)

    def inject_target_hidden(
        self,
        *,
        main_hidden: torch.Tensor,
        start_pos: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        """Project the target hidden and write its KV into every stage's draft ring.

        Owns the full per-stage injection orchestration (plan §2.g): project ->
        per-stage wkv proj -> kv_norm + rope -> sliding-window ring store. The worker
        passes the captured target hidden / positions / commit signal; the model owns
        the pool API. Returns the projected ``main_x`` for reuse in the draft forward.
        """
        main_x = self.project_target_hidden(main_hidden)
        for stage in self.stages:
            stage.self_attn.project_and_store_main_kv(
                main_x, start_pos, is_prefill=is_prefill
            )
        return main_x

    def forward_embed(
        self, main_hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.embed_tokens is None:
            raise ValueError(
                "DeepseekV4ForCausalLMDSpark requires the target embed_tokens "
                "(call attach_shared_modules first)."
            )
        main_x = self.project_target_hidden(main_hidden)
        draft_input_ids = input_ids.new_full(
            [input_ids.size(0), self.block_size], self.noise_token_id
        )
        draft_input_ids[:, 0] = input_ids
        x = self.embed_tokens(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def _collapse_hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse the draft mHC tensor through the last stage's hc_head (PRE-norm).

        Reference model.py:862 ``x = hc_head(x)``. The returned tensor is the
        post-hc_head PRE-norm hidden that feeds the confidence head; the LM-head path
        applies ``norm`` to it separately (model.py:863, the norm only feeds the logits).
        """
        last = self.stages[-1]
        return hc_head_torch(
            x,
            last.hc_head_fn,
            last.hc_head_scale,
            last.hc_head_base,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def _logits_from_x_post_hc(self, x_post_hc: torch.Tensor) -> torch.Tensor:
        """LM-head logits from the post-hc_head hidden: norm -> matmul -> TP all-gather.

        ``lm_head.weight`` is the target's local vocab shard under TP, so the local
        matmul yields rank-local vocab logits; ``tensor_model_parallel_all_gather`` then
        assembles the full vocab on every rank (a no-op at tp=1), sliced back to
        ``org_vocab_size`` to drop the TP vocab padding. Mirrors the dense worker's
        ``_compute_base_logits`` (dspark_worker_v2.py).
        """
        if self.lm_head is None:
            raise ValueError(
                "DeepseekV4ForCausalLMDSpark requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        last = self.stages[-1]
        x = last.norm(x_post_hc)
        local_logits = F.linear(x.float(), self.lm_head.weight.float())
        full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
        org_vocab_size = int(self.lm_head.org_vocab_size)
        return full_logits[..., :org_vocab_size]

    def forward_head(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        sampler: Optional[StepSampler] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Serial Markov head finish (reference ``forward_head`` model.py:860).

        Produces ``output_ids[block_size + 1]`` (anchor + gamma sampled) and the
        Markov-corrected per-step logits. ``sampler`` defaults to greedy argmax. When the
        confidence head is enabled it is evaluated on the post-hc_head PRE-norm tap
        ``x_post_hc`` (reference model.py:873) and the result is stashed on
        ``self._last_confidence`` for the worker relay.
        """
        if sampler is None:
            sampler = _greedy_step_sampler
        x_post_hc = self._collapse_hc_head(x)
        base_logits = self._logits_from_x_post_hc(x_post_hc)
        sampled_tokens, corrected_logits = self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=input_ids,
            hidden_states=None,
            sampler=sampler,
        )
        output_ids = torch.cat(
            [input_ids.unsqueeze(1), sampled_tokens.to(input_ids.dtype)], dim=1
        )
        self._last_confidence = self._compute_confidence(
            x_post_hc=x_post_hc, anchor_tokens=input_ids, sampled_tokens=sampled_tokens
        )
        return output_ids, corrected_logits

    def _compute_confidence(
        self,
        *,
        x_post_hc: torch.Tensor,
        anchor_tokens: torch.Tensor,
        sampled_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Confidence on the post-hc_head PRE-norm tap; stashed for the worker relay.

        Returns ``None`` when the confidence head is disabled (keeping a+b unaffected).
        Otherwise feeds ``x_post_hc`` (the same tap the LM head's ``norm`` consumes, but
        BEFORE the norm) and, for with_markov heads, the per-step markov_embed stack built
        from the prev-token sequence ``[anchor, s_0, ..., s_{gamma-2}]`` (the off-by-one
        shared with the worker). STS calibration is identity in the MVP, so the raw logit
        is mapped to ``(0, 1)`` by sigmoid; losslessness does not depend on the value.
        """
        confidence_head = self.confidence_head
        if confidence_head is None:
            return None
        if confidence_head.with_markov:
            prev_seq = torch.cat(
                [anchor_tokens.view(-1, 1), sampled_tokens[:, : self.gamma - 1]], dim=1
            )
            markov_embed_stack = self.markov_head.get_prev_embeddings(prev_seq)
        else:
            markov_embed_stack = None
        confidence_raw = confidence_head(x_post_hc, markov_embed_stack)
        confidence = torch.sigmoid(confidence_raw.float())
        assert bool(
            ((confidence >= 0) & (confidence <= 1)).all()
        ), "DSpark confidence must lie in [0, 1]."
        return confidence

    @torch.no_grad()
    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        start_pos: int = 0,
        sampler: Optional[StepSampler] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Reference ``forward_spec`` (model.py:928): embed -> stages -> head finish.

        On prefill (``start_pos == 0``) only the draft KV window is filled (returns
        None); on decode the full block runs and the Markov head emits the draft tokens.
        """
        x, main_x = self.forward_embed(main_hidden, input_ids)
        for stage in self.stages:
            x = stage(x, start_pos, input_ids, main_x)
        if start_pos == 0:
            self._last_confidence = None
            return None
        return self.forward_head(x, input_ids, sampler=sampler)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load DSpark draft weights from the V4 ``mtp.{i}.*`` checkpoint namespace.

        Remaps the reference ``mtp.{stage}.*`` names to the draft module tree
        (``stages.{stage}.*``), loads the confidence head when enabled (drops it
        otherwise), drops the shared embed/lm_head (supplied by the target), and routes
        MoE/attention weights through the V4 name conventions. Never goes through the
        NextN loader (plan §2.d). A confidence head that is enabled but absent from the
        checkpoint is identity-initialized with a warning (mirrors the dense head).
        """
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        stacked_params_mapping = DEEPSEEK_V4_STACKED_PARAMS_MAPPING
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_weight_name(name)
            if mapped is None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped:
                    continue
                candidate = mapped.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(candidate)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in mapped:
                        continue
                    candidate = mapped.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    param = params_dict[candidate]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        candidate,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(candidate)
                    break
                else:
                    if mapped not in params_dict:
                        logger.warning(
                            "DSpark V4 draft: unexpected weight %r -> %r", name, mapped
                        )
                        continue
                    param = params_dict[mapped]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(mapped)

        self._maybe_identity_init_confidence_head(
            params_dict=params_dict, loaded_params=loaded_params
        )

    def _maybe_identity_init_confidence_head(
        self, *, params_dict: dict, loaded_params: set
    ) -> None:
        if self.confidence_head is None:
            return
        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_params
        if missing:
            logger.warning(
                "DSpark V4 confidence head present but checkpoint is missing %s; "
                "identity-initializing to a constant accept probability of 0.5 "
                "(advisory-only; does not affect losslessness).",
                sorted(missing),
            )
            with torch.no_grad():
                self.confidence_head.proj.weight.zero_()
                if self.confidence_head.proj.bias is not None:
                    self.confidence_head.proj.bias.zero_()

    def _remap_dspark_weight_name(self, name: str) -> Optional[str]:
        """Map a reference ``mtp.{stage}.*`` checkpoint name to a draft param name."""
        if name.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
            return None
        if "rotary_emb.inv_freq" in name:
            return None

        if not name.startswith("mtp."):
            return None
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        stage_id, rest = parts[1], parts[2]

        if rest.startswith("markov_head."):
            return f"markov_head.{rest[len('markov_head.'):]}"

        if rest.startswith("confidence_head."):
            if self.confidence_head is None:
                return None
            return f"confidence_head.{rest[len('confidence_head.'):]}"

        mapped_rest = rest
        mapped_rest = mapped_rest.replace("attn.", "self_attn.", 1)
        mapped_rest = mapped_rest.replace("ffn.", "mlp.", 1)
        mapped_rest = mapped_rest.replace("attn_norm.", "input_layernorm.", 1)
        mapped_rest = mapped_rest.replace("ffn_norm.", "post_attention_layernorm.", 1)
        mapped_rest = mapped_rest.replace(".w1.", ".gate_proj.")
        mapped_rest = mapped_rest.replace(".w2.", ".down_proj.")
        mapped_rest = mapped_rest.replace(".w3.", ".up_proj.")
        mapped_rest = mapped_rest.replace(".gate.bias", ".gate.e_score_correction_bias")
        return f"stages.{stage_id}.{mapped_rest}"


EntryClass = [DeepseekV4ForCausalLMDSpark]
