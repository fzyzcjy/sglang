# Vendored DSpark V4 (DeepSeek-V4-Flash-DSpark) source-of-truth (SoT) modeling.
#
# Upstream source:
#   lab/code/scratch/2026-06-29-spark/DeepSeek-V4-Flash-DSpark-inference/model.py
#   lab/code/scratch/2026-06-29-spark/DeepSeek-V4-Flash-DSpark-inference/kernel.py
#
# This is the EXTERNAL standard-answer for the dsv4 DSpark draft parity tests
# (category 1). The SGLang production draft + this oracle are fed the SAME random
# weights and the SAME inputs; their outputs (KV projections, hc_head collapse,
# base logits, markov-corrected logits, confidence) must agree. It is intentionally
# NOT the in-repo modules: the production path and this oracle must derive the math
# independently, so a shared bug cannot make a wrong production path pass.
#
# Copied verbatim from model.py except, in order:
# - The TP parallel layers (`ParallelEmbedding` / `ColumnParallelLinear` /
#   `RowParallelLinear` / `ParallelHead`) are reduced to their world_size==1,
#   rank==0 path: the `dist.all_reduce` / `dist.all_gather` branches and the
#   vocab-shard masking are dropped (they are no-ops at world_size 1). The math on
#   the remaining path is verbatim. The parity tests run tp=1.
# - `act_quant` (fp8 nope-latent QAT) is dropped: this float oracle keeps the latent
#   in float, so the parity test sets the through-sparse-attention tolerance ABOVE
#   the production fp8 noise floor (see Ch8). The pure-linear components (Ch6) carry
#   no fp8 and are compared tight.
# - `sparse_attn` is the independent pure-torch re-expression in `sot_attention.py`
#   (the SoT `kernel.py` version is a tilelang JIT kernel unavailable here). NEVER
#   import the in-repo `_sparse_attn` scaffold (shared-index-semantics blindspot).
# - `hc_split_sinkhorn` (the SoT mHC pre/post/comb kernel, kernel.py:430) is
#   re-expressed in pure torch below from the kernel body verbatim.
# - `apply_rotary_emb`, `precompute_freqs_cis`, `get_dspark_topk_idxs` are reused
#   verbatim from `sot_attention.py`.
# - The compress / indexer / MoE / hash-routing machinery of the full model is NOT
#   vendored: the DSpark draft attention is compress_ratio == 0 (sliding window
#   only) and the parity tests drive `forward_spec`, not the dense `forward`. The
#   draft FFN is the standard SoT `Block` FFN reused via a simple SwiGLU expert.
# - `DSparkMarkovHead` / `DSparkConfidenceHead` / the `Block` hc_pre/hc_post/hc_head
#   math / `DSparkBlock.forward_embed` / `forward_head` / `Transformer.forward_spec`
#   are copied verbatim.

from __future__ import annotations

from dataclasses import dataclass
from test.srt.speculative._dspark_reference.deepseek_v4.sot_attention import (
    SoTDSparkAttentionOracle,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RefModelArgs:
    """DSpark-relevant subset of the SoT ``ModelArgs`` (model.py:34).

    Field names match the SoT keys; the full-model fields the DSpark draft path does
    not touch (moe routing, compress ratios, indexer) are omitted.
    """

    vocab_size: int = 256
    dim: int = 64
    n_layers: int = 2
    n_mtp_layers: int = 1
    n_heads: int = 4
    head_dim: int = 16
    rope_head_dim: int = 8
    q_lora_rank: int = 32
    o_lora_rank: int = 32
    o_groups: int = 2
    window_size: int = 16
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 0
    max_seq_len: int = 512
    temperature: float = 0.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    moe_inter_dim: int = 128
    dspark_block_size: int = 3
    dspark_noise_token_id: int = 1
    dspark_markov_rank: int = 8
    dspark_target_layer_ids: Tuple[int, ...] = (0,)
    enable_confidence_head: bool = False


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Verbatim from model.py:939 (Gumbel-max sampler; temperature 0 = greedy)."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


class RefEmbedding(nn.Module):
    """``ParallelEmbedding`` (model.py:89) reduced to the world_size==1 path."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class RefLinear(nn.Module):
    """``Linear`` (model.py:129) reduced to the plain bf16/float ``F.linear`` path.

    The SoT fp8/fp4 quant branches are dropped: this float oracle never quantizes
    weights, so ``linear`` degenerates to ``F.linear`` (model.py:126 else-branch).
    ``ColumnParallelLinear`` / ``RowParallelLinear`` collapse onto this at
    world_size==1 (no shard, no all_reduce), so all three share this module.
    """

    def __init__(
        self, in_features: int, out_features: int, dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RefRMSNorm(nn.Module):
    """Verbatim from model.py:189 (fp32 RMSNorm, weight stored fp32)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class RefHead(nn.Module):
    """``ParallelHead`` (model.py:719) reduced to the world_size==1 path.

    The vocab is not sharded and there is no all_gather, so the head is a plain fp32
    ``F.linear``. ``full_logits`` keeps the whole sequence; otherwise only the last
    position is scored (verbatim model.py:733).
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
        if not full_logits:
            x = x[:, -1]
        return F.linear(x.float(), self.weight)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch re-expression of the SoT ``hc_split_sinkhorn`` kernel (kernel.py:430).

    Reproduces the kernel body verbatim per row: ``pre`` / ``post`` are independent
    sigmoid gates; ``comb`` is a Sinkhorn-normalized [hc, hc] coupling. ``mixes`` is
    ``[n, mix_hc]`` with ``mix_hc = (2 + hc) * hc``; the first ``hc`` entries are the
    pre gates, the next ``hc`` are the post gates, the remaining ``hc * hc`` are the
    comb logits (row-major). Matches the kernel's softmax-row + iterated
    row/col normalization exactly.
    """
    n = mixes.shape[0]
    hc = hc_mult
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mixes[:, 2 * hc :].view(n, hc, hc) * hc_scale[2] + hc_base[2 * hc :].view(
        hc, hc
    )

    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class RefMoE(nn.Module):
    """Minimal SwiGLU FFN standing in for the SoT ``MoE`` on the draft path.

    The full model's hash/score routing + sharded experts (model.py:614) is not
    exercised by ``forward_spec`` parity in the tiny config (single shared expert);
    this reproduces the per-token SwiGLU the DSpark draft FFN reduces to with one
    expert, in float for stability (verbatim ``Expert.forward`` math, model.py:601).
    """

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = RefLinear(dim, inter_dim)
        self.w2 = RefLinear(inter_dim, dim)
        self.w3 = RefLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        y = F.silu(gate) * up
        return self.w2(y.to(dtype))


class RefDSparkAttention(nn.Module):
    """SoT ``DSparkAttention`` (model.py:750), weights owned, sparse_attn pure-torch.

    Holds the MLA-LoRA weight modules (wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/wo_b +
    attn_sink) and its own dense KV ring, and runs the SoT decode-block forward via
    the independent ``SoTDSparkAttentionOracle``. compress_ratio == 0 (sliding window
    only): no compressor/indexer.
    """

    def __init__(self, args: RefModelArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = args.o_groups
        self.window_size = args.window_size
        self.eps = args.norm_eps
        self.softmax_scale = self.head_dim**-0.5

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = RefLinear(self.dim, self.q_lora_rank)
        self.q_norm = RefRMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = RefLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = RefLinear(self.dim, self.head_dim)
        self.kv_norm = RefRMSNorm(self.head_dim, self.eps)
        self.wo_a = RefLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = RefLinear(self.n_groups * self.o_lora_rank, self.dim)

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_head_dim,
            seqlen=args.max_seq_len,
            original_seq_len=args.original_seq_len,
            base=args.rope_theta,
            factor=args.rope_factor,
            beta_fast=args.beta_fast,
            beta_slow=args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.kv_cache: Optional[torch.Tensor] = None

    def _oracle(
        self, device: torch.device, dtype: torch.dtype
    ) -> SoTDSparkAttentionOracle:
        oracle = SoTDSparkAttentionOracle(
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            n_heads=self.n_local_heads,
            n_groups=self.n_local_groups,
            o_lora_rank=self.o_lora_rank,
            window_size=self.window_size,
            eps=self.eps,
            softmax_scale=self.softmax_scale,
            freqs_cis=self.freqs_cis,
            attn_sink=self.attn_sink,
            device=device,
            dtype=dtype,
        )
        oracle.kv_cache = self.kv_cache
        return oracle

    def forward(
        self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor
    ) -> torch.Tensor:
        """SoT ``DSparkAttention.forward`` (model.py:752), split prefill / decode."""
        bsz, seqlen, _ = main_x.size()
        rd = self.rope_head_dim
        device, dtype = x.device, x.dtype

        main_freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)

        oracle = self._oracle(device, dtype)
        if start_pos == 0:
            oracle.write_window(main_kv, start_pos=0, bsz=bsz)
            self.kv_cache = oracle.kv_cache
            return x

        block_size = x.size(1)
        freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]

        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_normed = q.clone()

        kv = self.kv_norm(self.wkv(x))
        kv_normed = kv.clone()

        out = oracle.forward_block(
            q_normed=q_normed,
            kv_normed=kv_normed,
            main_kv_step=main_kv,
            start_pos=start_pos,
            seqlen=seqlen,
            wo_a_weight=self.wo_a.weight,
            wo_b=self.wo_b,
        )
        self.kv_cache = oracle.kv_cache
        del freqs_cis
        return out


class RefBlock(nn.Module):
    """SoT ``Block`` (model.py:652) mHC mixing math, draft-relevant subset.

    Copies hc_pre / hc_post / hc_head verbatim. The attention submodule is the
    DSpark attention; the FFN is the single-expert SwiGLU stand-in.
    """

    def __init__(self, args: RefModelArgs) -> None:
        super().__init__()
        self.norm_eps = args.norm_eps
        self.attn = RefDSparkAttention(args)
        self.ffn = RefMoE(args.dim, args.moe_inter_dim)
        self.attn_norm = RefRMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RefRMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Verbatim from model.py:680."""
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes.view(-1, mixes.shape[-1]),
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        pre = pre.view(*shape[:2], self.hc_mult)
        post = post.view(*shape[:2], self.hc_mult)
        comb = comb.view(*shape[:2], self.hc_mult, self.hc_mult)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Verbatim from model.py:690."""
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
        """Verbatim from model.py:709 (the collapse, PRE-norm)."""
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
        """Verbatim from model.py:695, threading the DSpark attention's ``main_x``."""
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
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class DSparkMarkovHead(nn.Module):
    """Verbatim from model.py:795."""

    def __init__(self, vocab_size: int, dspark_markov_rank: int) -> None:
        super().__init__()
        self.markov_w1 = RefEmbedding(vocab_size, dspark_markov_rank)
        self.markov_w2 = RefHead(vocab_size, dspark_markov_rank)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.markov_w1(token_ids)
        logits = self.markov_w2(embed, full_logits=True)
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    """Verbatim from model.py:807 (proj stored fp32; squeeze last dim)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = RefLinear(input_dim, 1, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)


class DSparkBlock(RefBlock):
    """Verbatim from model.py:818 (DSpark stage: main_proj, markov + confidence heads)."""

    def __init__(self, args: RefModelArgs, stage_id: int) -> None:
        super().__init__(args)
        self.dim = args.dim
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.temperature = args.temperature
        self.enable_confidence_head = args.enable_confidence_head
        hc_dim = self.hc_mult * args.dim
        if stage_id == 0:
            assert len(args.dspark_target_layer_ids) > 0, "DSpark needs target layers"
            self.main_proj = RefLinear(
                args.dim * len(args.dspark_target_layer_ids), args.dim
            )
            self.main_norm = RefRMSNorm(args.dim, args.norm_eps)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RefRMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(
                args.vocab_size, args.dspark_markov_rank
            )
            self.confidence_head = DSparkConfidenceHead(
                args.dim + args.dspark_markov_rank
            )
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.embed: Optional[RefEmbedding] = None
        self.head: Optional[RefHead] = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
    ) -> torch.Tensor:
        """Verbatim from model.py:845."""
        if start_pos > 0:
            return super().forward(x, start_pos, input_ids, main_x)
        return self.attn(x, start_pos, main_x)

    def forward_embed(
        self, main_hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verbatim from model.py:851."""
        assert self.embed is not None
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_input_ids = input_ids.new_full(
            [input_ids.size(0), self.block_size], self.noise_token_id
        )
        draft_input_ids[:, 0] = input_ids
        x = self.embed(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def forward_head(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Verbatim from model.py:860 (hc_head -> norm -> head; serial markov loop)."""
        assert self.head is not None
        x = self.hc_head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        logits = self.head(self.norm(x), full_logits=True)
        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, i])
            logits[:, i].add_(logits_bias)
            markov_embeds.append(markov_embed)
            output_ids[:, i + 1] = sample(logits[:, i], self.temperature)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = None
        if self.enable_confidence_head:
            confidence = self.confidence_head(x, markov_embed)
        return output_ids, logits, confidence


class RefTransformer(nn.Module):
    """SoT ``Transformer`` (model.py:877) draft path: forward_spec + the head taps.

    Builds the embed, the (non-DSpark) main blocks (only their KV ring is exercised
    by the parity tests' prefill) and the DSpark MTP stages. The full-model dense
    ``forward`` (moe routing / compress / indexer) is not vendored; the tests drive
    ``forward_spec`` and feed ``main_hidden`` directly.
    """

    def __init__(self, args: RefModelArgs) -> None:
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.temperature = args.temperature
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = args.hc_mult
        self.embed = RefEmbedding(args.vocab_size, args.dim)
        self.norm = RefRMSNorm(args.dim, self.norm_eps)
        self.head = RefHead(args.vocab_size, args.dim)
        self.target_layer_ids = args.dspark_target_layer_ids
        self.mtp = nn.ModuleList()
        for stage_id in range(args.n_mtp_layers):
            self.mtp.append(DSparkBlock(args, stage_id))
            self.mtp[-1].embed = self.embed
            self.mtp[-1].head = self.head

    @torch.inference_mode()
    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        start_pos: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Verbatim from model.py:928.

        Prefill (start_pos == 0) writes the target-hidden window into each stage's KV
        ring and returns None; decode (start_pos > 0) runs the block forward and the
        head finish, returning (output_ids, logits, confidence).
        """
        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
        for layer in self.mtp:
            h = layer(h, start_pos, input_ids, main_x)
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(h, input_ids)

    def project_target_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """``main_norm(main_proj(.))`` (model.py:853), the Ch6 projection tap."""
        stage0 = self.mtp[0]
        return stage0.main_norm(stage0.main_proj(main_hidden))

    def collapse_hc_head(self, h: torch.Tensor) -> torch.Tensor:
        """``hc_head(h)`` (model.py:862), the post-hc_head PRE-norm tap."""
        last = self.mtp[-1]
        return last.hc_head(h, last.hc_head_fn, last.hc_head_scale, last.hc_head_base)

    def base_logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """``head(norm(hc_head(h)))`` (model.py:862-863), full-vocab base logits."""
        last = self.mtp[-1]
        x = self.collapse_hc_head(h)
        return self.head(last.norm(x), full_logits=True)


__all__ = [
    "RefModelArgs",
    "RefTransformer",
    "DSparkBlock",
    "DSparkMarkovHead",
    "DSparkConfidenceHead",
    "RefDSparkAttention",
    "hc_split_sinkhorn",
    "sample",
]
