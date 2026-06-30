# Vendored DSpark V4 source-of-truth (SoT) attention oracle.
#
# Upstream source:
#   lab/code/scratch/2026-06-29-spark/DeepSeek-V4-Flash-DSpark-inference/model.py
#   lab/code/scratch/2026-06-29-spark/DeepSeek-V4-Flash-DSpark-inference/kernel.py
#
# This is the EXTERNAL standard-answer for the T1 block-forward parity guardrail.
# It is intentionally NOT the in-repo `_sparse_attn` scaffold: the production path
# and the in-repo scaffold share the same index semantics, so comparing them would
# be blind to a shared bug. This oracle reproduces the SoT math independently.
#
# Copied verbatim from the SoT except:
# - `apply_rotary_emb`, `precompute_freqs_cis`, `get_dspark_topk_idxs` are copied
#   verbatim (pure torch, no kernel dependency).
# - `sparse_attn` is re-expressed in pure torch (the SoT `kernel.py` version is a
#   tilelang JIT kernel unavailable in the parity environment). The pure-torch
#   form reproduces the same index-gather + online-softmax-with-attn_sink math:
#     o[h] = softmax([q.k_j for valid j] ++ attn_sink[h]) @ ([v_j] ++ 0)
#   i.e. the attn_sink contributes an extra logit (key = sink, value = 0). Indices
#   equal to -1 are masked out (the kernel fills those slots with 0 and sets their
#   pre-softmax score to -inf), so the value contribution and the softmax weight of
#   a masked slot are both zero. Verified against the SoT kernel's online-softmax
#   accumulation (kernel.py sparse_attn_kernel_): the running-max rescale and the
#   final `+ exp(attn_sink - max)` denominator term are algebraically identical to
#   appending the sink as one extra (key, value=0) pair.
# - `DSparkAttention.forward` is copied verbatim from model.py (~752). It still
#   reads/writes `self.kv_cache` (the SoT dense ring) so the oracle owns its own KV
#   state, exactly as the SoT does. The production path under test owns a paged
#   pool instead; the parity test feeds BOTH the SAME inputs (target hidden / anchor
#   / KV / positions / start_pos) and compares the attention output + base_logits.

import math
from functools import lru_cache

import torch


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """Verbatim copy of SoT model.py:206 (YaRN-scaled rotary frequencies)."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Verbatim copy of SoT model.py:238 (in-place rotary embedding)."""
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


@lru_cache(1)
def get_dspark_topk_idxs(
    window_size: int, bsz: int, block_size: int, start_pos: int
) -> torch.Tensor:
    """Verbatim copy of SoT model.py:744.

    NON-CAUSAL full-block index layout: every one of the ``block_size`` draft
    query rows shares the SAME ``[whole window ++ whole block]`` index set. There
    is NO causal triangle (compare SoT ``get_window_topk_idxs`` @ 261 /
    ``get_compress_topk_idxs`` @ 280, which DO mask causally). This shared-row,
    triangle-free layout is exactly what the production non-causal builder must
    reproduce; a causal regression makes parity diverge (T1 negative test).
    """
    assert start_pos > 0
    matrix = torch.cat(
        [
            torch.arange(min(window_size, start_pos + 1)),
            window_size + torch.arange(block_size),
        ]
    )
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Pure-torch re-expression of SoT kernel.py:355 ``sparse_attn``.

    Args mirror the SoT kernel:
      q:          [b, m, h, d]   query rows
      kv:         [b, n, d]      gathered KV table (shared K and V latent)
      attn_sink:  [h]            per-head learnable sink logit
      topk_idxs:  [b, m, topk]   per (batch, query-row) gathered KV positions; -1 = invalid
      softmax_scale: float       pre-softmax scale

    Returns o: [b, m, h, d].

    Reproduces the SoT online-softmax-with-sink: for each (b, m, h),
      logits_j = scale * <q, kv[idx_j]>     (idx_j != -1)
      weights  = softmax([logits_j ... , attn_sink[h]])   over valid j AND the sink
      o        = sum_j weights_j * kv[idx_j]               (sink value is 0)
    A masked slot (idx == -1) gets a -inf logit (zero softmax weight) and a zero
    value, matching the kernel's `idxs == -1 -> kv 0, score -inf` handling.
    """
    b, m, h, d = q.size()
    topk = topk_idxs.size(-1)
    device = q.device

    out = torch.empty_like(q, dtype=torch.float32)
    q_f = q.float()
    kv_f = kv.float()
    sink = attn_sink.float()

    for bi in range(b):
        for mi in range(m):
            idxs = topk_idxs[bi, mi].long()
            valid = idxs != -1
            safe_idxs = torch.where(valid, idxs, torch.zeros_like(idxs))
            gathered_kv = kv_f[bi, safe_idxs]
            scores = torch.einsum("hd,td->ht", q_f[bi, mi], gathered_kv) * softmax_scale
            neg_inf = torch.full_like(scores, float("-inf"))
            scores = torch.where(valid.unsqueeze(0).expand_as(scores), scores, neg_inf)
            sink_logit = sink.view(h, 1)
            all_logits = torch.cat([scores, sink_logit], dim=1)
            weights = torch.softmax(all_logits, dim=1)
            kv_weights = weights[:, :topk]
            out[bi, mi] = torch.einsum("ht,td->hd", kv_weights, gathered_kv)

    return out.to(q.dtype)


class SoTDSparkAttentionOracle:
    """Standalone SoT ``DSparkAttention.forward`` (model.py:752), weight-injected.

    Reproduces the SoT decode-block attention math VERBATIM (the start_pos > 0
    branch of model.py DSparkAttention.forward), but takes the linear weights and
    KV ring from the caller so the parity test can share the SAME weights with the
    production path. The prefill branch (``write_window``) writes the target-hidden
    window into the ring exactly as the SoT does.

    The intent is end-to-end equivalence of the attention output ``o`` (post wo_a /
    wo_b) and, when wired through the head, the hc-collapsed base_logits.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        rope_head_dim: int,
        n_heads: int,
        n_groups: int,
        o_lora_rank: int,
        window_size: int,
        eps: float,
        softmax_scale: float,
        freqs_cis: torch.Tensor,
        attn_sink: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.n_local_heads = n_heads
        self.n_local_groups = n_groups
        self.o_lora_rank = o_lora_rank
        self.window_size = window_size
        self.eps = eps
        self.softmax_scale = softmax_scale
        self.freqs_cis = freqs_cis
        self.attn_sink = attn_sink
        self.kv_cache: torch.Tensor | None = None
        self._device = device
        self._dtype = dtype

    def _kv_norm(self, x: torch.Tensor, kv_norm_weight: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (kv_norm_weight.float() * x).to(dtype)

    def write_window(self, main_kv: torch.Tensor, start_pos: int, bsz: int) -> None:
        """SoT prefill window write (model.py:763 start_pos == 0 branch)."""
        win = self.window_size
        seqlen = main_kv.size(1)
        if self.kv_cache is None:
            self.kv_cache = torch.zeros(
                bsz, win, self.head_dim, device=self._device, dtype=self._dtype
            )
        if seqlen <= win:
            self.kv_cache[:bsz, :seqlen] = main_kv
        else:
            cutoff = seqlen % win
            (
                self.kv_cache[:bsz, cutoff:win],
                self.kv_cache[:bsz, :cutoff],
            ) = main_kv[
                :, -win:
            ].split([win - cutoff, cutoff], dim=1)

    def forward_block(
        self,
        *,
        q_normed: torch.Tensor,
        kv_normed: torch.Tensor,
        main_kv_step: torch.Tensor,
        start_pos: int,
        seqlen: int,
        wo_a_weight: torch.Tensor,
        wo_b: torch.nn.Module,
    ) -> torch.Tensor:
        """SoT decode-block attention (model.py:771-792 start_pos > 0 branch).

        Args are the already-projected/normed tensors so weights are shared with
        production: ``q_normed`` is ``q_norm(wq_a(x))`` reshaped+rmsnorm'd per head
        (pre-rope), ``kv_normed`` is ``kv_norm(wkv(x))`` (pre-rope), ``main_kv_step``
        is the single committed target-hidden KV (post norm+rope) to store at the
        ring slot. wo_a is applied as the SoT grouped einsum; wo_b is the module.
        """
        bsz, block_size = q_normed.shape[0], q_normed.shape[1]
        win = self.window_size
        rd = self.rope_head_dim
        freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]

        q = q_normed.clone()
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = kv_normed.clone()
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos)
        self.kv_cache[:bsz, start_pos % win] = main_kv_step.squeeze(1)
        kv = torch.cat([self.kv_cache[:bsz], kv], dim=1)
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = wo_a_weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x, _ = wo_b(o.flatten(2))
        return x


__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "get_dspark_topk_idxs",
    "sparse_attn",
    "SoTDSparkAttentionOracle",
]
