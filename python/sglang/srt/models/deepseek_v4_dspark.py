# Adapted from the DeepSeek-V4-Flash DSpark reference (DSparkBlock / DSparkAttention /
# DSparkMarkovHead / Transformer.forward_spec) but implemented with SGLang primitives.
# The V4 DSpark draft is a block draft that reads and writes the production paged
# DeepSeekV4TokenToKVPool sliding-window MLA ring (the same pool the target uses),
# driven by the production sparse FlashMLA kernel. Its attention is compress_ratio == 0
# (sliding window only, no compressor / indexer) and NON-CAUSAL over the full draft
# block: every one of the gamma block queries attends the whole injected target-hidden
# window plus the whole draft block (reference get_dspark_topk_idxs, model.py:744). The
# draft carries a target-hidden projection (main_proj/main_norm), a serial Markov head,
# and reuses the target model's token embedding / lm_head for the base logits. When the
# draft config enables it, the draft also carries an opt-in confidence head that consumes
# the post-hc_head PRE-norm draft hidden (reference model.py:862/873: x=hc_head(x),
# confidence=confidence_head(x, markov_embed); the norm only feeds the logits).

from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.layers.attention.deepseek_v4_backend import (
    PAGE_INDEX_ALIGNED_SIZE,
    SWA_WINDOW,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v4 import (
    DEEPSEEK_V4_STACKED_PARAMS_MAPPING,
    DeepseekV4DecoderLayer,
    hc_head_torch,
    make_hc_head_params,
)
from sglang.srt.models.dspark import DSparkConfidenceHead
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_utils import parse_dspark_draft_config
from sglang.srt.utils import add_prefix
from sglang.srt.utils.common import ceil_align

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


def build_dspark_swa_page_indices(
    *,
    window_swa_locs: torch.Tensor,
    block_swa_locs: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the NON-CAUSAL full-block paged SWA index layout for the draft block.

    Paged port of the reference ``get_dspark_topk_idxs`` (model.py:744):

        matrix = cat([arange(min(window, start_pos+1)), window + arange(block_size)])
                 .view(1, 1, -1).expand(bsz, block_size, -1)

    Every one of the ``block_size`` draft queries in a request attends the SAME set of
    SWA slots: the whole sliding window of injected target-hidden KV plus the whole draft
    block (NON-CAUSAL, no triangular mask). The window part is the per-request committed
    target-hidden slots; the block part is the slots this draft forward just wrote for the
    gamma draft tokens. Both are already in SWA space (the caller translates from full
    space via ``translate_loc_from_full_to_swa``).

    Args:
        window_swa_locs: ``[bs, SWA_WINDOW]`` int32 SWA slots of the committed window,
            most-recent-last, with positions before the request's start padded ``-1``.
        block_swa_locs: ``[bs, block_size]`` int32 SWA slots written by this forward for
            the gamma draft tokens (shared by every query row of the request).
        context_lens: ``[bs]`` int the number of valid committed window tokens per request
            (= min(SWA_WINDOW, prefix_len)); rows beyond this in ``window_swa_locs`` are
            padding and excluded from the attended length.
        block_size: gamma, the number of draft-block query rows / draft tokens.

    Returns:
        ``swa_page_indices`` ``[bs * block_size, K]`` int32 (K padded to a multiple of
        ``PAGE_INDEX_ALIGNED_SIZE``; padding value ``-1``) and ``swa_topk_lengths``
        ``[bs * block_size]`` int32 (= context_lens + block_size, identical across the
        ``block_size`` rows of a request). The kernel attends only the first
        ``swa_topk_lengths[q]`` entries, so the ``-1`` padding (both intra-window and
        alignment) is never read.
    """
    if window_swa_locs.ndim != 2 or window_swa_locs.shape[1] != SWA_WINDOW:
        raise ValueError(
            "window_swa_locs must be [bs, SWA_WINDOW]; "
            f"got shape={tuple(window_swa_locs.shape)} (SWA_WINDOW={SWA_WINDOW})."
        )
    if block_swa_locs.ndim != 2 or block_swa_locs.shape[1] != block_size:
        raise ValueError(
            "block_swa_locs must be [bs, block_size]; "
            f"got shape={tuple(block_swa_locs.shape)} (block_size={block_size})."
        )
    bs = window_swa_locs.shape[0]
    device = window_swa_locs.device

    window_swa_locs = window_swa_locs.to(torch.int32)
    block_swa_locs = block_swa_locs.to(torch.int32)
    context_lens = context_lens.to(device=device, dtype=torch.int32)

    # The widest row holds the full window (SWA_WINDOW) + the whole block, aligned up.
    target_width = ceil_align(SWA_WINDOW + block_size, PAGE_INDEX_ALIGNED_SIZE)

    # The valid attended entries are the first context_lens window slots immediately
    # followed by the block slots, with no causal mask. Compact the window so the valid
    # window slots are contiguous and directly precede the block slots, matching
    # ``topk_lengths = context_lens + block_size`` (the kernel reads a prefix run).
    swa_page_indices = _compact_window_then_block(
        window_swa_locs=window_swa_locs,
        block_swa_locs=block_swa_locs,
        context_lens=context_lens,
        target_width=target_width,
        block_size=block_size,
    )

    # Replicate the request's single shared row to all block_size query rows (non-causal:
    # every query sees the same window + whole block).
    swa_page_indices = (
        swa_page_indices.view(bs, 1, target_width)
        .expand(bs, block_size, target_width)
        .reshape(bs * block_size, target_width)
        .contiguous()
    )
    swa_topk_lengths = (
        (context_lens + block_size)
        .view(bs, 1)
        .expand(bs, block_size)
        .reshape(bs * block_size)
        .contiguous()
        .to(torch.int32)
    )
    return swa_page_indices, swa_topk_lengths


def _compact_window_then_block(
    *,
    window_swa_locs: torch.Tensor,
    block_swa_locs: torch.Tensor,
    context_lens: torch.Tensor,
    target_width: int,
    block_size: int,
) -> torch.Tensor:
    """Left-pack each request's valid window slots, then its block slots, then -1.

    ``window_swa_locs`` keeps invalid (pre-start) slots as ``-1`` interleaved at the
    front (the reference fills the most-recent-last window with padding at the start when
    ``start_pos + 1 < window``). The kernel reads a length-prefix run, so the valid window
    slots must be contiguous and immediately followed by the block slots. This gathers the
    last ``context_lens`` window slots, appends the block slots, and ``-1``-pads the rest.
    """
    bs = window_swa_locs.shape[0]
    device = window_swa_locs.device
    out = torch.full((bs, target_width), -1, dtype=torch.int32, device=device)

    col = torch.arange(SWA_WINDOW, device=device, dtype=torch.int32).view(1, -1)
    # The last context_lens entries of the SWA_WINDOW columns are the valid window slots
    # (most-recent-last). Map them to packed columns [0, context_lens).
    valid_window = col >= (SWA_WINDOW - context_lens.view(-1, 1))
    packed_window_col = col - (SWA_WINDOW - context_lens.view(-1, 1))
    rows = torch.arange(bs, device=device).view(-1, 1).expand(-1, SWA_WINDOW)
    out[rows[valid_window], packed_window_col[valid_window]] = window_swa_locs[
        valid_window
    ]

    block_col = context_lens.view(-1, 1) + torch.arange(
        block_size, device=device, dtype=torch.int32
    ).view(1, -1)
    block_rows = torch.arange(bs, device=device).view(-1, 1).expand(-1, block_size)
    out[block_rows, block_col] = block_swa_locs
    return out


class DSparkAttention(nn.Module):
    """Sliding-window sparse MLA for the V4 DSpark draft (compress_ratio == 0).

    Reuses the weight / projection structure of the production ``MQALayer``
    (``deepseek_v4.py``) at compress_ratio == 0 -- same wq_a/wkv/q_norm/wq_b/kv_norm/
    wo_a/wo_b/attn_sink shapes, same attn-TP sharding -- but drives the production paged
    ``DeepSeekV4TokenToKVPool`` sliding-window ring and the production sparse FlashMLA
    kernel with a NON-CAUSAL full-block index layout (every draft-block query attends the
    whole injected target-hidden window plus the whole draft block, reference
    ``DSparkAttention.forward`` model.py:752). It is intentionally NOT the causal SWA
    metadata that ``MQALayer.forward`` / the backend's default forward build.
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        assert self.head_dim == config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // self.attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // self.attn_tp_size
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

        assert config.head_dim == self.head_dim
        assert config.num_key_value_heads == 1

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self._attn_sink_local: Optional[torch.Tensor] = (
            self.attn_sink if self.attn_tp_size == 1 else None
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
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            params_dtype=torch.bfloat16,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis
        from sglang.srt.utils.hf_transformers_utils import get_rope_config

        rope_theta, rope_scaling = get_rope_config(config)
        rope_scaling = rope_scaling or {}
        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=0,
            base=rope_theta,
            factor=rope_scaling.get("factor", 1.0),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        # RadixAttention is the layer handle the pool / FlashMLA paths key off (layer_id);
        # it is not invoked directly (the DSpark forward calls the sparse kernel itself).
        self.attn = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:
        kv, _ = self.wkv(x)
        return kv

    def _local_attn_sink(self) -> torch.Tensor:
        if self.attn_tp_size == 1:
            return self.attn_sink
        if self._attn_sink_local is None:
            rank = self.attn_tp_rank
            self._attn_sink_local = self.attn_sink[
                rank * self.n_local_heads : (rank + 1) * self.n_local_heads
            ].contiguous()
        return self._attn_sink_local

    def store_block_kv(
        self,
        *,
        x: torch.Tensor,
        positions: torch.Tensor,
        block_swa_locs: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        """Project the draft block hidden into KV and write it into the SWA ring.

        Mirrors ``MQALayer._compute_kv_to_cache``: wkv proj -> fused kv_norm + rope +
        packed FlashMLA store into the production SWA buffer at the block's SWA slots.
        """
        kv = self.kv_proj_only(x)
        pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=block_swa_locs.to(torch.int32),
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )

    def compute_q(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Project the draft block hidden to per-head queries with rmsnorm + rope.

        Returns ``[num_queries, n_local_heads, head_dim]`` (flat over bs * block_size).
        """
        rd = self.rope_head_dim
        q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(
            q.dtype
        )
        freqs_cis = self.freqs_cis[positions]
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        return q

    def attend(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        swa_page_indices: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> torch.Tensor:
        """Run the sparse FlashMLA kernel over the non-causal full-block SWA index.

        Reads the production SWA key buffer for this layer, attends each query to its
        ``swa_topk_lengths`` slots given by ``swa_page_indices`` (no causal mask in the
        kernel), then applies the inverse rope and the wo_a/wo_b output projection.
        """
        import sgl_kernel.flash_mla as flash_mla

        rd = self.rope_head_dim
        swa_k_cache = pool.get_swa_key_buffer_radix(self.layer_id)
        swa_window_size = pool.swa_window_size
        k_cache_total_dim = pool.swa_kv_pool.kv_cache_total_dim
        swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
            swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
        )

        attn_q = q.unsqueeze(1)
        page_indices = swa_page_indices.unsqueeze(1)
        assert (
            page_indices.shape[-1] % PAGE_INDEX_ALIGNED_SIZE == 0
        ), f"{page_indices.shape=} last dim not aligned to {PAGE_INDEX_ALIGNED_SIZE}"

        o = flash_mla.flash_mla_with_kvcache(
            q=attn_q,
            k_cache=swa_k_cache,
            head_dim_v=self.head_dim,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=flash_mla.get_mla_metadata()[0],
            softmax_scale=self.softmax_scale,
            is_fp8_kvcache=True,
            indices=page_indices,
            topk_length=swa_topk_lengths,
            attn_sink=self._local_attn_sink(),
            extra_k_cache=None,
            extra_indices_in_kvcache=None,
            extra_topk_length=None,
        )[0]
        o = o.squeeze(1)

        freqs_cis = self.freqs_cis[positions]
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        o = o.view(o.shape[0], self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bgd,grd->bgr", o.float(), wo_a.float()).to(q.dtype)
        out, _ = self.wo_b(o.reshape(o.shape[0], -1))
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        pool = _resolve_dspark_pool()
        index = build_dspark_attn_index(
            forward_batch=forward_batch,
            window_size=self.window_size,
            pool=pool,
        )
        self.store_block_kv(
            x=hidden_states,
            positions=positions,
            block_swa_locs=index.block_swa_locs,
            pool=pool,
        )
        q = self.compute_q(hidden_states, positions)
        return self.attend(
            q=q,
            positions=positions,
            swa_page_indices=index.swa_page_indices,
            swa_topk_lengths=index.swa_topk_lengths,
            pool=pool,
        )


def _resolve_dspark_pool() -> DeepSeekV4TokenToKVPool:
    pool = get_token_to_kv_pool()
    assert isinstance(pool, DeepSeekV4TokenToKVPool), (
        "DSpark draft attention requires a DeepSeekV4TokenToKVPool, "
        f"got {type(pool).__name__}."
    )
    return pool


class _DSparkAttnIndex:
    """Resolved per-forward non-causal full-block index + the block's SWA write slots."""

    def __init__(
        self,
        *,
        swa_page_indices: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        block_swa_locs: torch.Tensor,
    ) -> None:
        self.swa_page_indices = swa_page_indices
        self.swa_topk_lengths = swa_topk_lengths
        self.block_swa_locs = block_swa_locs


def build_dspark_attn_index(
    *,
    forward_batch: ForwardBatch,
    window_size: int,
    pool: DeepSeekV4TokenToKVPool,
) -> _DSparkAttnIndex:
    """Resolve the non-causal full-block SWA index from a draft ForwardBatch.

    The draft block ForwardBatch carries (per request): ``req_pool_indices``,
    ``seq_lens`` (committed prefix length), ``out_cache_loc`` (the gamma block slots in
    full space, flat ``[bs * block_size]``). The committed window's full locations are
    ``req_to_token[req, prefix_len - W : prefix_len]`` (most-recent-last, ``-1`` where the
    request is shorter than the window). Both window and block locs are translated to SWA
    space and handed to ``build_dspark_swa_page_indices``.
    """
    spec_info = forward_batch.spec_info
    block_size = int(spec_info.draft_token_num)
    bs = int(forward_batch.batch_size)
    device = forward_batch.out_cache_loc.device

    req_pool_indices = forward_batch.req_pool_indices.to(device=device)
    prefix_lens = forward_batch.seq_lens.to(device=device, dtype=torch.int64)
    context_lens = torch.clamp(prefix_lens, max=window_size).to(torch.int32)

    req_to_token = get_req_to_token_pool().req_to_token
    offsets = (
        prefix_lens.view(-1, 1)
        - window_size
        + torch.arange(window_size, device=device, dtype=torch.int64).view(1, -1)
    )
    invalid = offsets < 0
    offsets = offsets.clamp(min=0)
    window_full_locs = req_to_token[req_pool_indices.view(-1, 1), offsets]
    window_full_locs = window_full_locs.masked_fill(invalid, 0)
    window_swa_locs = pool.translate_loc_from_full_to_swa(window_full_locs).to(
        torch.int32
    )
    window_swa_locs = window_swa_locs.masked_fill(invalid, -1)

    block_full_locs = forward_batch.out_cache_loc.view(bs, block_size)
    block_swa_locs = pool.translate_loc_from_full_to_swa(block_full_locs).to(
        torch.int32
    )

    swa_page_indices, swa_topk_lengths = build_dspark_swa_page_indices(
        window_swa_locs=window_swa_locs,
        block_swa_locs=block_swa_locs,
        context_lens=context_lens,
        block_size=block_size,
    )
    return _DSparkAttnIndex(
        swa_page_indices=swa_page_indices,
        swa_topk_lengths=swa_topk_lengths,
        block_swa_locs=block_full_locs.reshape(-1),
    )


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

    Subclasses ``DeepseekV4DecoderLayer`` to reuse its MoE, mHC mixing params, norms, and
    the ``hc_pre``/``hc_post`` math; only the attention submodule (``DSparkAttention`` via
    ``_build_self_attn``) and the forward contract differ. The forward is the standard
    SGLang ``(positions, hidden_states, forward_batch)`` contract on token-flattened
    ``[N, hc, d]`` tensors -- the worker builds the draft ForwardBatch and the attention
    reads / writes the production paged pool.
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
        """Run the base ``hc_pre`` on the token-flattened draft tensor ``[N, hc, d]``."""
        y, post, comb, _ = self.hc_pre(x, hc_fn, hc_scale, hc_base)
        return y, post, comb

    def _hc_post_block(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Run the base ``hc_post`` on the token-flattened draft tensor ``[N, hc, d]``."""
        return self.hc_post(x, residual, post, comb)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        x, post, comb = self._hc_pre_block(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        x = self.self_attn(positions, x, forward_batch)
        x = self._hc_post_block(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre_block(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self._run_ffn(x)
        x = self._hc_post_block(x, residual, post, comb)
        return x

    def _run_ffn(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        y = self.mlp(x)
        return y.view(shape)


class DeepseekV4ForCausalLMDSpark(nn.Module):
    """V4 DSpark draft model: block draft over the production paged sliding-window MLA KV.

    Owns the target-hidden projection (main_proj/main_norm), the DSpark MTP stages, the
    Markov head, an opt-in confidence head, and the head finish (hc_head + the target's
    shared lm_head). The token embedding and lm_head are supplied by the target model and
    attached via ``attach_shared_modules`` (the worker wires them).

    The standard ``forward(input_ids, positions, forward_batch, input_embeds)`` runs the
    DSpark backbone on the production paged pool (every stage reads / writes the real SWA
    ring through ``DSparkAttention``) and returns the per-token draft backbone hidden. The
    base-logit collapse (hc_head) + serial Markov head finish are driven by the worker via
    ``compute_base_logits`` / the shared sampler so the dense and dsv4 worker paths agree.

    When the confidence head is enabled the draft computes the confidence at the correct
    tap (post-hc_head, PRE-norm draft hidden) and stashes it on ``self._last_confidence``;
    the worker relay reads it via ``last_confidence`` (the dense relay's post-norm
    ``draft_hidden`` would be the wrong tap for V4, reference model.py:862/873).
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
        if dspark_config.target_layer_ids is not None:
            self.num_stages = len(dspark_config.target_layer_ids)
        else:
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
        """Confidence stashed by the most recent ``forward_head`` (worker relay).

        Returns the post-STS confidence ``[bs, gamma]`` in ``(0, 1)`` computed from the
        post-hc_head PRE-norm tap, or ``None`` when the confidence head is disabled.
        """
        return self._last_confidence

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        """Attach the target model's shared embedding and lm_head (worker wiring)."""
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def project_target_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hc-mean features -> draft hidden (main_proj)."""
        stage0 = self.stages[0]
        projected, _ = stage0.main_proj(main_hidden)
        return stage0.main_norm(projected)

    def kv_proj_only(self, layer_idx: int, ctx_hidden: torch.Tensor) -> torch.Tensor:
        """Project a (already main_proj'd) draft hidden into one stage's KV latent."""
        return self.stages[layer_idx].self_attn.kv_proj_only(ctx_hidden)

    def forward_embed(self, input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Build the draft block input embeddings: [anchor, noise, ..., noise] per row.

        ``input_ids`` is the flat ``[bs * gamma]`` draft block ids the worker built (anchor
        at column 0, noise elsewhere). Returns the hc-expanded embedding ``[N, hc, d]``.
        """
        if self.embed_tokens is None:
            raise ValueError(
                "DeepseekV4ForCausalLMDSpark requires the target embed_tokens "
                "(call attach_shared_modules first)."
            )
        x = self.embed_tokens(input_ids)
        x = x.unsqueeze(1).repeat(1, self.hc_mult, 1)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        """Standard SGLang draft forward: embed -> DSpark stages -> backbone hidden.

        The worker builds the draft block ForwardBatch (TARGET_VERIFY mode, the gamma
        block slots in ``out_cache_loc``, per-row positions, ``spec_info.draft_token_num``
        = gamma) and passes the noise-block ``input_embeds`` (anchor at column 0). Returns
        a ``LogitsProcessorOutput`` carrying the post-stage hc-expanded draft hidden
        ``[N, hc, d]``; the head finish (hc_head + Markov) is driven by the worker.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        del get_embedding, pp_proxy_tensors
        if input_embeds is None:
            input_embeds = self.forward_embed(
                input_ids, batch_size=int(forward_batch.batch_size)
            )
        x = input_embeds
        for stage in self.stages:
            x = stage(positions, x, forward_batch)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)

    def collapse_hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse the draft mHC tensor through the last stage's hc_head (PRE-norm).

        Reference model.py:862 ``x = hc_head(x)``. The returned tensor is the post-hc_head
        PRE-norm hidden that feeds both the confidence head and (after ``norm``) the LM
        head. ``x`` is ``[N, hc, d]`` (token-flattened).
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

    def compute_base_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Base logits from the draft backbone hidden: hc_head -> norm -> lm_head gather.

        Collapses the mHC draft hidden through the last stage's hc_head (PRE-norm), then
        applies ``norm`` and the target's local-vocab lm_head matmul, all-gathers to the
        full vocab (no-op at tp=1), and crops the TP vocab padding. This is the dsv4
        analog of the dense worker's ``_compute_base_logits`` (which has no hc_head
        collapse). The worker calls this and feeds the result to the shared Markov loop.
        """
        x_post_hc = self.collapse_hc_head(x)
        self._x_post_hc = x_post_hc
        return self._logits_from_x_post_hc(x_post_hc)

    def _logits_from_x_post_hc(self, x_post_hc: torch.Tensor) -> torch.Tensor:
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

    def compute_confidence(
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
        ``x_post_hc`` / ``sampled_tokens`` are ``[bs, gamma, ...]`` (worker-reshaped).
        """
        confidence_head = self.confidence_head
        if confidence_head is None:
            self._last_confidence = None
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
        self._last_confidence = confidence
        return confidence

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load DSpark draft weights from the V4 ``mtp.{i}.*`` checkpoint namespace.

        Remaps the reference ``mtp.{stage}.*`` names to the draft module tree
        (``stages.{stage}.*``), loads the confidence head when enabled (drops it
        otherwise), drops the shared embed/lm_head (supplied by the target), and routes
        MoE/attention weights through the V4 name conventions. Never goes through the
        NextN loader. A confidence head that is enabled but absent from the checkpoint is
        identity-initialized with a warning (mirrors the dense head).
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
        mapped_rest = mapped_rest.replace(".gate.tid2eid", ".topk.tid2eid")
        mapped_rest = mapped_rest.replace(".gate.bias", ".gate.e_score_correction_bias")
        mapped_rest = mapped_rest.replace(".scale", ".weight_scale_inv")
        return f"stages.{stage_id}.{mapped_rest}"


EntryClass = [DeepseekV4ForCausalLMDSpark]
