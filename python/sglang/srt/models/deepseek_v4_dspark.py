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
from typing import Iterable, List, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.jit_kernel.dsv4 import fused_q_norm_rope, fused_rope_inplace
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v4 import (
    DEEPSEEK_V4_STACKED_PARAMS_MAPPING,
    DeepseekV4DecoderLayer,
    MqaAttentionBase,
    _dequant_fp8_wo_a,
    hc_head_torch,
    make_hc_head_params,
)
from sglang.srt.models.dspark import (
    DSparkConfidenceHead,
    StepSampler,
    gather_and_crop_vocab,
    run_markov_block,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_utils import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.utils import add_prefix, is_blackwell_supported
from sglang.srt.utils.async_probe import maybe_detect_in_closed_range

logger = logging.getLogger(__name__)

# FlashMLA's fp8 sparse decode kernel only specializes h_q for {64, 128}; the draft pads
# its per-rank query heads up to this when tp shards them below 64.
_PAD_NUM_HEADS = 64


class DSparkV4DraftOutput(msgspec.Struct, frozen=True):
    """Structured output of ``DeepseekV4ForCausalLMDSpark.forward`` (worker contract).

    The dsv4 ``forward`` produces ONLY the raw backbone hidden; base-logit production
    (the hc_head collapse the dense path lacks) is now the model's ``compute_base_logits``
    hook driven by the worker post-forward, so this struct no longer carries base logits or
    the confidence tap. This struct is returned as the model-runner ``logits_output``, so
    the worker reads ``draft_out.logits_output.hidden_states`` and then calls
    ``compute_base_logits`` on it.

    NOTE: ``ModelRunner._forward_raw`` passes this struct through unchanged only because the
    dsv4 draft always runs through the ``EagerRunner`` (TARGET_VERIFY + ``input_embeds``
    both make ``PrefillCudaGraphRunner.can_run_graph`` return False). A graph runner's
    ``execute`` only accepts ``LogitsProcessorOutput`` / ``EmbeddingPoolerOutput`` /
    ``PPProxyTensors``; this struct must never enter one.

    Fields:
        draft_hidden: ``[bs * gamma, hc, d]`` post-stage hc-expanded backbone hidden (the
            raw, un-collapsed tensor the stages produced; ``compute_base_logits`` consumes
            this directly).
    """

    draft_hidden: torch.Tensor

    @property
    def hidden_states(self) -> torch.Tensor:
        # Back-compat accessor: the model-runner / generic logits-output consumers read
        # ``.hidden_states``; the DSpark draft backbone hidden is ``draft_hidden``.
        return self.draft_hidden

    @property
    def next_token_logits(self) -> None:
        # The dsv4 draft does not surface logits on this struct; the worker calls
        # ``compute_base_logits`` on ``hidden_states`` and never reads this slot.
        return None


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place rotary embedding, mirroring the DSpark reference (model.py:238).

    Used by the eager (``SGLANG_DSPARK_FAST_KERNEL`` off) attention path; the fast path
    drives the fused CUDA rope kernels instead.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(x.size(0), 1, x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


class DSparkAttention(MqaAttentionBase):
    """Sliding-window sparse MLA for the V4 DSpark draft (compress_ratio == 0).

    Extends ``MqaAttentionBase`` to reuse the weight / projection structure of the
    production ``MQALayer`` (``deepseek_v4.py``) at compress_ratio == 0 -- same
    wq_a/wkv/q_norm/wq_b/kv_norm/wo_a/wo_b/attn_sink shapes, same attn-TP sharding --
    but drives the production paged ``DeepSeekV4TokenToKVPool`` sliding-window ring and
    the production sparse FlashMLA kernel with a NON-CAUSAL full-block index layout
    (every draft-block query attends the whole injected target-hidden window plus the
    whole draft block, reference ``DSparkAttention.forward`` model.py:752). It is
    intentionally NOT the causal SWA metadata that ``MQALayer.forward`` / the backend's
    default forward build.
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            attn_tp_rank=get_parallel().attn_tp_rank,
            attn_tp_size=get_parallel().attn_tp_size,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_a_fp8=False,
            wo_a_keeps_quant_config=False,
            wo_b_reduce_results=True,
            rope_original_seq_len=0,
        )
        assert (
            self.compress_ratio == 0
        ), "DSpark draft attention requires compress_ratio == 0."
        self.window_size = int(
            getattr(config, "sliding_window", None) or config.window_size
        )

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

        self._use_fast_kernel = envs.SGLANG_DSPARK_FAST_KERNEL.get()
        # Alt streams for the capture-mode KV/Q overlap in forward (mirrors
        # MQALayer._forward_prepare_multi_stream's stream_kv). Creation is already
        # gated by the model-level env checks, so None here means overlap off.
        self.alt_streams = alt_streams
        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64
        self._attn_sink_padded: Optional[torch.Tensor] = None

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

    def _padded_attn_sink(self) -> torch.Tensor:
        """Local attn_sink padded to _PAD_NUM_HEADS, built once post weight load.

        Zero padding (not garbage) is required here, unlike the q padding: the sink is
        per-head data the kernel reads for every head it runs. A per-forward rebuild
        would replay a fill + copy per stage in the draft decode graph (mirrors the
        target's build-once ``_attn_sink_local``).
        """
        attn_sink = self._local_attn_sink()
        if self.n_local_heads >= _PAD_NUM_HEADS:
            return attn_sink
        if self._attn_sink_padded is None:
            sink_padded = attn_sink.new_zeros(_PAD_NUM_HEADS)
            sink_padded[: self.n_local_heads] = attn_sink
            self._attn_sink_padded = sink_padded
        return self._attn_sink_padded

    def _store_block_kv(
        self,
        *,
        kv: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        """Write the draft block raw latent KV into the SWA ring (fused norm + rope).

        Mirrors ``MQALayer._compute_kv_to_cache``: the raw ``wkv`` latent is normed +
        rope'd + packed and stored at the block's SWA slots
        (``attn_backend.get_swa_out_cache_loc`` translates ``out_cache_loc`` -> SWA),
        so the backend's ``forward`` runs with ``save_kv_cache=False``. The non-causal
        full-block index that the kernel consumes is built by the backend metadata
        (``get_dspark_swa_page_indices``), which references the same translated slots.
        """
        pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )

    def _compute_q(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project the draft block hidden to per-head queries with rmsnorm + rope.

        Returns ``[num_queries, n_local_heads, head_dim]`` (flat over bs * block_size).
        Fast path (``SGLANG_DSPARK_FAST_KERNEL`` on) drives the production fused
        rmsnorm-self + RoPE kernel (``fused_q_norm_rope``, the same one
        ``MQALayer._compute_q_b`` uses; same layout from ``MqaAttentionBase``). Slow path
        is the eager reference float rsqrt + complex ``apply_rotary_emb``. When ``q_out``
        is given (the caller's padded-q slice, possibly strided -- the fused kernel
        writes strided like ``_compute_q_b``), the result lands there directly, saving
        the padded-copy launch.
        """
        q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self._use_fast_kernel:
            if q_out is None:
                q_out = torch.empty_like(q)
            fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
            return q_out
        else:
            q = q * torch.rsqrt(
                q.float().square().mean(-1, keepdim=True) + self.eps
            ).to(q.dtype)
            apply_rotary_emb(q[..., -self.rope_head_dim :], self.freqs_cis[positions])
            if q_out is not None:
                q_out.copy_(q)
                return q_out
            return q

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        pool = _resolve_dspark_pool()
        attn_backend = get_attn_backend()
        rd = self.rope_head_dim

        # KV-store chain (wkv -> fused norm/rope/pool-write) and Q chain (wq_a ->
        # q_norm -> wq_b -> fused q norm rope) both depend only on hidden_states; the
        # join point is the backend forward (needs q + the KV already in the pool).
        # Capture-mode only, like MQALayer._forward_prepare_multi_stream: the fork/join
        # is recorded into the draft cuda graph as event deps (zero replay CPU cost),
        # while eager runs keep the serial order.
        enable_multi_stream = (
            self.alt_streams is not None
            and get_is_capture_mode()
            and hidden_states.shape[0] <= self._multi_stream_bs_limit
        )

        # Pad the per-rank query heads (and attn_sink) up to the kernel's supported h_q
        # like MQALayer and slice the output heads back afterward. new_empty, not
        # new_zeros: only [:, :n_local_heads] is written and each head's attention is
        # independent, so the garbage padded heads only yield garbage output heads that
        # are sliced away (mirrors the target's non-gfx942 q_padded). _compute_q writes
        # straight into the padded slice, dropping the per-forward memset + copy.
        q_padded: Optional[torch.Tensor] = None
        q_out: Optional[torch.Tensor] = None
        if self.n_local_heads < _PAD_NUM_HEADS:
            q_padded = hidden_states.new_empty(
                hidden_states.shape[0], _PAD_NUM_HEADS, self.head_dim
            )
            q_out = q_padded[:, : self.n_local_heads, :]

        if enable_multi_stream:
            current_stream = torch.cuda.current_stream()
            stream_kv = self.alt_streams[0]
            stream_kv.wait_stream(current_stream)
            with torch.cuda.stream(stream_kv):
                kv = self.kv_proj_only(hidden_states)
                self._store_block_kv(
                    kv=kv,
                    positions=positions,
                    forward_batch=forward_batch,
                    attn_backend=attn_backend,
                    pool=pool,
                )
            q = self._compute_q(hidden_states, positions, q_out=q_out)
            current_stream.wait_stream(stream_kv)
        else:
            kv = self.kv_proj_only(hidden_states)
            self._store_block_kv(
                kv=kv,
                positions=positions,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
                pool=pool,
            )
            q = self._compute_q(hidden_states, positions, q_out=q_out)

        if q_padded is not None:
            q = q_padded
        attn_sink = self._padded_attn_sink()

        # Drive the production sparse backend: it reads the NON-CAUSAL full-block SWA
        # metadata built in make_core_attn_metadata (is_dspark_draft branch) and runs
        # flash_mla over the real SWA key buffer. The KV store is already done above, so
        # save_kv_cache=False (mirrors MQALayer's non-fused path). attn_sink is this
        # rank's local slice.
        o = attn_backend.forward(
            q=q,
            k=kv,
            v=kv,
            layer=self.attn,
            forward_batch=forward_batch,
            compress_ratio=0,
            attn_sink=attn_sink,
            save_kv_cache=False,
        )
        if o.shape[1] != self.n_local_heads:
            o = o[:, : self.n_local_heads, :]

        if self._use_fast_kernel:
            fused_rope_inplace(
                o[..., -rd:], None, self.freqs_cis, positions=positions, inverse=True
            )
        else:
            apply_rotary_emb(o[..., -rd:], self.freqs_cis[positions], inverse=True)

        o = o.view(o.shape[0], self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if self._use_fast_kernel:
            # bf16 wo_a einsum, mirroring the production MQALayer else-path
            # (deepseek_v4.py:1223) and the reference (model.py:790, also plain bf16 --
            # no .float()). wo_a is bf16 (wo_a_fp8=False), so tensor-core fp32
            # accumulation makes this numerically equivalent to the old fp32 cast.
            o = torch.einsum("bgd,grd->bgr", o, wo_a)
        else:
            o = torch.einsum("bgd,grd->bgr", o.float(), wo_a.float()).to(q.dtype)
        out, _ = self.wo_b(o.reshape(o.shape[0], -1))
        return out


def _resolve_dspark_pool() -> DeepSeekV4TokenToKVPool:
    pool = get_token_to_kv_pool()
    assert isinstance(pool, DeepSeekV4TokenToKVPool), (
        "DSpark draft attention requires a DeepSeekV4TokenToKVPool, "
        f"got {type(pool).__name__}."
    )
    return pool


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
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )


def build_dspark_v4_confidence_head(
    *, config: DeepSeekV4Config, markov_rank: int
) -> Optional[DSparkConfidenceHead]:
    """Build the V4 DSpark confidence head (enabled outside static ragged-verify).

    Mirrors the dense ``build_confidence_head`` interface (the head emits a RAW
    accept-rate logit; ``with_markov`` concatenates the per-step markov_embed). The V4
    hidden size comes from ``config.hidden_size`` and the rank from the parsed draft
    config. ``static`` ragged-verify uses a uniform block and never consults the head, so
    the head is skipped there; otherwise it is ALWAYS built (the ``enable_confidence_head``
    flag is not read): a DSpark draft checkpoint is expected to carry trained confidence
    weights, so a missing config field is only warned about, and a checkpoint that
    genuinely lacks the weights is surfaced by the weight-load assert. The ``proj`` matches
    the checkpoint's DeepSpec ``AcceptRatePredictor`` layout (a single weight, no bias).
    """
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    if not hasattr(config, "enable_confidence_head"):
        logger.warning(
            "DSpark draft config has no enable_confidence_head field; treating the "
            "confidence head as enabled."
        )
    with_markov_cfg = getattr(config, "confidence_head_with_markov", None)
    with_markov = (
        (markov_rank > 0) if with_markov_cfg is None else bool(with_markov_cfg)
    )
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark V4 confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=int(config.hidden_size),
        markov_rank=int(markov_rank),
        with_markov=with_markov,
        bias=False,
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
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        # is_nextn disables the MoE hash topk (config.num_hash_layers gates it on the
        # target's first layers by layer_id). The draft uses draft-local layer ids
        # (0..num_stages-1), which would otherwise be misread as those hash layers; the
        # draft gate is the normal noaux_tc gate (its checkpoint carries gate.bias), so
        # force the non-hash path like NextN. It only affects this MoE construction.
        # alt_streams flows to the base __init__ so the MoE gets its dual-stream
        # alt_stream (shared experts overlap gate+routed under capture) and to
        # _build_self_attn for the draft attention's KV/Q overlap.
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            is_nextn=True,
            alt_streams=alt_streams,
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
        del compress_ratio_override
        return DSparkAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
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

        self.start_layer = 0
        self.end_layer = self.num_stages
        # One shared alt stream is enough: within a stage the attention KV/Q fork joins
        # before the MoE starts, so the two overlap sites never run concurrently
        # (the target's MQALayer and MoE share alt_streams[0] the same way).
        use_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and envs.SGLANG_DSPARK_ENABLE_MULTI_STREAM.get()
            and torch.cuda.is_available()
        )
        self.alt_streams: Optional[List[torch.cuda.Stream]] = (
            [torch.cuda.Stream()] if use_multi_stream else None
        )
        self.stages = nn.ModuleList(
            [
                DSparkV4Stage(
                    config=config,
                    layer_id=stage_id,
                    stage_id=stage_id,
                    num_stages=self.num_stages,
                    num_target_layers=self.num_target_features,
                    quant_config=quant_config,
                    prefix=add_prefix(f"stages.{stage_id}", prefix),
                    alt_streams=self.alt_streams,
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
        self._use_fp32_lm_head = envs.SGLANG_DSPARK_FP32_LM_HEAD.get()
        self._last_confidence: Optional[torch.Tensor] = None
        self._x_post_hc: Optional[torch.Tensor] = None

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

    def write_target_hidden_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        """Inject the target hidden as MLA latent KV into every stage's SWA ring slot.

        The dsv4 draft KV is a single MLA latent (kv_lora_rank + qk_rope_head_dim), not the
        MHA k/v pair the dense path writes. For each stage: project the (main_proj'd) target
        hidden through ``wkv``, then ``set_swa_key_buffer_radix_fused_norm_rope`` (the writer
        consumes the RAW latent and applies kv_norm + rope + fp8 pack internally, so do NOT
        pre-norm/rope here) at the already-translated SWA slots ``swa_loc`` with the absolute
        per-row ``positions``. The worker owns full->SWA translation (after allocation) and
        the per-row commit positions; this method owns the projection + pool API. There is
        no MLA ``set_kv_buffer_prefix_valid`` equivalent, so commit-length masking is done by
        the caller marking non-committed slots with ``swa_loc = -1`` (fixed-shape, no gather);
        the fused-norm-rope writer kernel skips ``out_loc < 0``.
        """
        main_x = self.project_target_hidden(main_hidden)
        swa_loc = swa_loc.to(torch.int32)
        for stage in self.stages:
            attn = stage.self_attn
            kv = attn.kv_proj_only(main_x)
            pool.set_swa_key_buffer_radix_fused_norm_rope(
                layer_id=attn.layer_id,
                swa_loc=swa_loc,
                kv=kv,
                kv_weight=attn.kv_norm.weight.data,
                eps=attn.eps,
                freqs_cis=attn.freqs_cis,
                positions=positions,
            )

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build the draft block input embeddings, hc-expanded.

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
    ) -> LogitsProcessorOutput:
        """Standard SGLang draft forward: embed -> DSpark stages -> raw backbone hidden.

        The worker builds the draft block ForwardBatch (TARGET_VERIFY mode, the gamma block
        slots in ``out_cache_loc``, per-row positions, ``spec_info.draft_token_num`` =
        gamma). The worker passes only ``input_ids`` (the dsv4 model hc-expands the
        embedding itself via ``forward_embed``); ``input_embeds`` is accepted for parity
        callers. Runs the DSpark stages on the production paged SWA pool and returns a
        ``DSparkV4DraftOutput`` carrying ONLY the raw, un-collapsed backbone hidden. Base
        logits (hc_head collapse -> norm -> lm_head -> TP all-gather -> org-vocab crop) are
        produced separately by ``compute_base_logits``, which the worker calls on the raw
        hidden post-forward; the head finish (serial Markov sampling) is also worker-driven.
        """
        del get_embedding, pp_proxy_tensors
        if input_embeds is None:
            input_embeds = self.forward_embed(input_ids)
        x = input_embeds
        for stage in self.stages:
            x = stage(positions, x, forward_batch)

        # Return the raw [bs*gamma, hc, d] backbone hidden as a plain LogitsProcessorOutput
        # (next_token_logits stays None; the worker calls compute_base_logits on
        # hidden_states post-forward). LogitsProcessorOutput is the only struct the cuda
        # graph runner's execute accepts, so this is what lets the draft be captured.
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
        analog of the dense ``DSparkDraftMixin.compute_base_logits`` (which has no hc_head
        collapse); the matmul dtype is bf16 by default like the dense path, or the
        reference-parity fp32 ``F.linear`` when ``SGLANG_DSPARK_FP32_LM_HEAD`` is set.
        This is the SOLE base-logit producer: ``forward`` no longer computes them, and the
        post-hc_head PRE-norm tap is stashed on ``self._x_post_hc`` HERE for
        ``compute_confidence`` (the worker calls this before ``compute_confidence``, so the
        stash is fresh). The worker calls this on the raw forward hidden and feeds the
        result to the shared Markov loop.
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
        weight = self.lm_head.weight
        if self._use_fp32_lm_head:
            # Reference-parity path (model.py:735): fp32 matmul, per-step weight upcast.
            local_logits = F.linear(x.float(), weight.float())
        else:
            # sglang default / dense DSpark draft: keep the weight dtype (bf16) matmul.
            local_logits = torch.matmul(x.to(weight.dtype), weight.T)
        return gather_and_crop_vocab(local_logits, self.lm_head)

    def compute_confidence(
        self,
        *,
        anchor_tokens: torch.Tensor,
        sampled_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Confidence on the post-hc_head PRE-norm tap (R9 seam), called by the worker.

        Returns ``None`` when the confidence head is disabled (keeping the a+b path
        unaffected). Otherwise it reads the post-hc_head PRE-norm tap stashed by the most
        recent ``compute_base_logits`` (the same tap the LM head's ``norm`` consumes, but
        BEFORE the norm, reference model.py:873) and, for with_markov heads, the per-step
        markov_embed stack built from the prev-token sequence ``[anchor, s_0, ...,
        s_{gamma-2}]`` (the off-by-one shared with the worker). ``confidence_head.apply_sts``
        applies the per-position STS temperature (identity when no table is loaded) then
        sigmoid, mapping the raw logit to ``(0, 1)``; losslessness does not depend on the
        value. ``anchor_tokens`` is ``[bs]``; ``sampled_tokens`` is ``[bs, gamma]``.
        Returns ``[bs, gamma]``.
        """
        confidence_head = self.confidence_head
        if confidence_head is None:
            self._last_confidence = None
            return None
        if self._x_post_hc is None:
            raise RuntimeError(
                "compute_confidence requires compute_base_logits to run first "
                "(the post-hc_head tap is stashed there)."
            )
        bs = int(anchor_tokens.shape[0])
        x_post_hc = self._x_post_hc.view(bs, self.gamma, -1)
        if confidence_head.with_markov:
            prev_seq = torch.cat(
                [anchor_tokens.view(-1, 1), sampled_tokens[:, : self.gamma - 1]], dim=1
            )
            markov_embed_stack = self.markov_head.get_prev_embeddings(prev_seq)
        else:
            markov_embed_stack = None
        confidence_raw = confidence_head(x_post_hc, markov_embed_stack)
        confidence = confidence_head.apply_sts(confidence_raw)
        # Async, gated probe (SGLANG_ENABLE_ASYNC_ASSERT) instead of ``assert
        # bool(...all())``: the latter forces an is_nonzero -> item ->
        # _local_scalar_dense -> cudaStreamSynchronize on every decode step,
        # a hard d2h sync in the hot verify path.
        maybe_detect_in_closed_range(
            confidence, 0.0, 1.0, "DSpark confidence must lie in [0, 1]."
        )
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

        weights = list(weights)
        if any(name.endswith(".wo_a.scale") for name, _ in weights):
            # The HF dsv4 checkpoint stores wo_a as an fp8 weight + a separate
            # ``.wo_a.scale``, but the DSpark draft's o-projection runs a plain bf16
            # einsum (no fp8 gemm, no weight_scale_inv applied), so the raw fp8 weight
            # must be dequantized to bf16 at load -- mirroring the target's bf16 wo_a
            # path. Without this the einsum consumes unscaled fp8 bytes and the whole
            # attention output (hence every draft proposal) is garbage.
            weights = list(_dequant_fp8_wo_a(weights))

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

        self._assert_confidence_head_loaded(
            params_dict=params_dict, loaded_params=loaded_params
        )

    def _assert_confidence_head_loaded(
        self, *, params_dict: dict, loaded_params: set
    ) -> None:
        if self.confidence_head is None:
            return
        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_params
        if missing:
            raise ValueError(
                f"DSpark V4 confidence head is enabled but the checkpoint is missing "
                f"{sorted(missing)}. Provide a checkpoint with trained confidence weights, "
                f"or disable the confidence head (enable_confidence_head=False)."
            )

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
