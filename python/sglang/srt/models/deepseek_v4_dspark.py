# Adapted from the DeepSeek-V4-Flash DSpark reference (DSparkBlock / DSparkAttention /
# DSparkMarkovHead / Transformer.forward_spec) but implemented with SGLang primitives.
# The V4 DSpark draft is a block draft with its own sliding-window MLA KV (fed from the
# target hidden states), a target-hidden projection (main_proj/main_norm), and a serial
# Markov head. It carries no confidence head (out of scope for the static-verify MVP) and
# reuses the target model's token embedding / lm_head for the base logits.

from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models import deepseek_v2
from sglang.srt.models.dbrx import ReplicatedLinear
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
    plus the current full draft block (non-causal). Reuses the V4 MLA projection
    submodules but owns its own KV ring (the production paged pool is wired by the
    worker in Phase b).
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
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
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

        rope_scaling = config.rope_scaling or {}
        original_seq_len = rope_scaling.get("original_max_position_embeddings", 0)
        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=0 if self.compress_ratio == 0 else original_seq_len,
            base=config.rope_theta,
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
        bsz, seqlen, _ = main_x.shape
        main_freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        main_kv = self.apply_kv_norm_rope(self.kv_proj_only(main_x), main_freqs_cis)

        if start_pos == 0:
            self.write_window_kv(main_kv, start_pos)
            return x

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
        self.write_step_kv(main_kv.squeeze(1), start_pos)
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


class DSparkV4Stage(nn.Module):
    """One DSpark MTP stage (reference ``DSparkBlock`` model.py:818).

    Reuses the V4 MoE and HC parameters; swaps in ``DSparkAttention``. The pure-torch
    HC pre/post/head mirror the reference ``Block``; on GPU these align with the target
    model's mHC numerics (the target uses fused kernels for the same math).
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
        super().__init__()
        self.layer_id = layer_id
        self.stage_id = stage_id
        self.dim = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters

        self.attn = DSparkAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.ffn = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("ffn", prefix),
            layer_id=layer_id,
            is_deepseek_v4=True,
        )
        self.attn_norm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=self.norm_eps)

        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

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
            self.main_norm = RMSNorm(config.hidden_size, eps=self.norm_eps)

        if stage_id == num_stages - 1:
            self.norm = RMSNorm(config.hidden_size, eps=self.norm_eps)
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from sglang.srt.layers.mhc import hc_split_sinkhorn

        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
    ) -> torch.Tensor:
        if start_pos == 0:
            # Prefill only fills the draft KV window; the full block runs on decode.
            return self.attn(x, start_pos, main_x)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, main_x)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self._run_ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def _run_ffn(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        y = self.ffn(x)
        return y.view(shape)


class DeepseekV4ForCausalLMDSpark(nn.Module):
    """V4 DSpark draft model: block draft with target-hidden-fed sliding-window MLA KV.

    Owns the target-hidden projection (main_proj/main_norm), the DSpark MTP stages, the
    Markov head, and the head finish (hc_head + the target's shared lm_head). The token
    embedding and lm_head are supplied by the target model and attached via
    ``attach_shared_modules`` (the worker wires them in Phase b).
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
        self.hc_mult = int(config.hc_mult)

        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None

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
        bsz, seqlen, _ = main_x.shape
        for stage in self.stages:
            attn = stage.attn
            freqs_cis = attn.freqs_cis[start_pos : start_pos + seqlen]
            main_kv = attn.apply_kv_norm_rope(attn.kv_proj_only(main_x), freqs_cis)
            if is_prefill:
                attn.write_window_kv(main_kv, start_pos)
            else:
                attn.write_step_kv(main_kv.squeeze(1), start_pos)
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

    def _base_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise ValueError(
                "DeepseekV4ForCausalLMDSpark requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        last = self.stages[-1]
        x = last.hc_head(x, last.hc_head_fn, last.hc_head_scale, last.hc_head_base)
        x = last.norm(x)
        return F.linear(x.float(), self.lm_head.weight.float())

    def forward_head(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        sampler: Optional[StepSampler] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Serial Markov head finish (reference ``forward_head`` model.py:860).

        Produces ``output_ids[block_size + 1]`` (anchor + gamma sampled) and the
        Markov-corrected per-step logits. ``sampler`` defaults to greedy argmax.
        """
        if sampler is None:
            sampler = _greedy_step_sampler
        logits = self._base_logits(x)
        bsz = input_ids.size(0)
        output_ids = input_ids.new_empty(bsz, self.block_size + 1)
        output_ids[:, 0] = input_ids
        corrected_logits: List[torch.Tensor] = []
        for i in range(self.block_size):
            step_logits = self.markov_head.apply_step_logits(
                logits[:, i, :],
                token_ids=output_ids[:, i],
                hidden_states=None,
            )
            output_ids[:, i + 1] = sampler(step_logits, i)
            corrected_logits.append(step_logits.unsqueeze(1))
        return output_ids, torch.cat(corrected_logits, dim=1)

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
            return None
        return self.forward_head(x, input_ids, sampler=sampler)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load DSpark draft weights from the V4 ``mtp.{i}.*`` checkpoint namespace.

        Remaps the reference ``mtp.{stage}.*`` names to the draft module tree
        (``stages.{stage}.*``), drops the confidence head and the shared embed/lm_head
        (supplied by the target), and routes MoE/attention weights through the V4
        name conventions. Never goes through the NextN loader (plan §2.d).
        """
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
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

    def _remap_dspark_weight_name(self, name: str) -> Optional[str]:
        """Map a reference ``mtp.{stage}.*`` checkpoint name to a draft param name."""
        if name.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
            return None
        if "confidence_head" in name or "rotary_emb.inv_freq" in name:
            return None

        if not name.startswith("mtp."):
            return None
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        stage_id, rest = parts[1], parts[2]

        if rest.startswith("markov_head."):
            return f"markov_head.{rest[len('markov_head.'):]}"

        mapped_rest = rest
        mapped_rest = mapped_rest.replace(".w1.", ".gate_proj.")
        mapped_rest = mapped_rest.replace(".w2.", ".down_proj.")
        mapped_rest = mapped_rest.replace(".w3.", ".up_proj.")
        mapped_rest = mapped_rest.replace(".gate.bias", ".gate.e_score_correction_bias")
        return f"stages.{stage_id}.{mapped_rest}"


EntryClass = [DeepseekV4ForCausalLMDSpark]
