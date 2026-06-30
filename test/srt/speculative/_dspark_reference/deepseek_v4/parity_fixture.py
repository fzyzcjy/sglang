# Shared dsv4 DSpark parity fixtures: the tiny config, the SoT RefModelArgs builder, and
# the SoT->SGLang weight-sync helper. Imported by BOTH the granularity-A component parity
# test and the granularity-B block-forward harness so the MLA-LoRA weight mapping has a
# single source of truth. NOT a test module (no register_*; lives under test/srt/).

from __future__ import annotations

import torch


def make_tiny_dsv4_config(
    *,
    enable_confidence_head: bool = False,
    num_heads: int = 4,
    gamma: int = 3,
    hidden_size: int = 32,
    window_size: int = 16,
    vocab_size: int = 128,
):
    """Build a tiny ``DeepSeekV4Config`` carrying the dsv4 DSpark draft fields.

    Mirrors the dense ``_TinyQwen3Config``: a real production config with dims shrunk to
    tiny values and the ``dspark_*`` prefixed draft fields attached (the
    ``parse_dspark_draft_config`` prefixed convention). ``head_dim`` is set explicitly
    (qk_rope_head_dim + qk_nope_head_dim) so the DSparkAttention head-dim assert holds,
    and ``num_key_value_heads == 1`` (single MLA latent), as the draft attention requires.
    """
    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config

    qk_rope_head_dim = 8
    qk_nope_head_dim = 8
    head_dim = qk_rope_head_dim + qk_nope_head_dim
    config = DeepSeekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=head_dim,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        window_size=window_size,
        intermediate_size=64,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        vocab_size=vocab_size,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000,
        rope_scaling={},
        compress_ratios=[],
        quantization_config=None,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )
    config.dspark_block_size = gamma
    config.dspark_markov_rank = 8
    config.dspark_noise_token_id = 1
    config.dspark_target_layer_ids = [0]
    config.temperature = 0.0
    config.enable_confidence_head = enable_confidence_head
    config.confidence_head_with_markov = enable_confidence_head
    return config


def make_ref_args_from_config(config):
    """Build the vendored SoT ``RefModelArgs`` matching the tiny SGLang config dims.

    The reference oracle and the production draft must share identical dims so the weight
    sync is a 1:1 copy and the only remaining difference is the math impl.
    """
    from test.srt.speculative._dspark_reference.deepseek_v4.modeling import RefModelArgs

    return RefModelArgs(
        vocab_size=int(config.vocab_size),
        dim=int(config.hidden_size),
        n_layers=int(config.num_hidden_layers),
        n_mtp_layers=len(config.dspark_target_layer_ids),
        n_heads=int(config.num_attention_heads),
        head_dim=int(config.head_dim),
        rope_head_dim=int(config.qk_rope_head_dim),
        q_lora_rank=int(config.q_lora_rank),
        o_lora_rank=int(config.o_lora_rank),
        o_groups=int(config.o_groups),
        window_size=int(config.window_size),
        norm_eps=float(config.rms_norm_eps),
        rope_theta=float(config.rope_theta),
        rope_factor=1.0,
        original_seq_len=0,
        max_seq_len=int(config.max_position_embeddings),
        temperature=0.0,
        hc_mult=int(config.hc_mult),
        hc_sinkhorn_iters=int(config.hc_sinkhorn_iters),
        hc_eps=float(config.hc_eps),
        moe_inter_dim=int(config.moe_intermediate_size),
        dspark_block_size=int(config.dspark_block_size),
        dspark_noise_token_id=int(config.dspark_noise_token_id),
        dspark_markov_rank=int(config.dspark_markov_rank),
        dspark_target_layer_ids=tuple(config.dspark_target_layer_ids),
        enable_confidence_head=bool(config.enable_confidence_head),
    )


def _copy_attn_lora_weights(*, ref_attn, sgl_attn) -> None:
    """Copy the SoT MLA-LoRA attention weights into the SGLang DSparkAttention (tp=1).

    The dsv4 draft attention is MLA-LoRA, NOT the qwen-style fused q/k/v_proj: the weights
    are ``wq_a`` (down) -> ``q_norm`` -> ``wq_b`` (up, per-head) for Q, ``wkv`` (latent
    down) -> ``kv_norm`` for the single MLA KV latent, and a grouped low-rank output
    ``wo_a`` (bf16) -> ``wo_b``. At tp=1 the SGLang ColumnParallel/RowParallel projections
    hold the full (unsharded) weight, so each is a direct ``.weight`` copy. ``attn_sink``
    is shape ``n_heads`` (not ``n_local_heads``); at tp=1 the whole vector copies across.
    There is no fused ``wqkv_a`` on the draft path (separate ReplicatedLinear wq_a / wkv),
    so no split is needed.
    """
    with torch.no_grad():
        sgl_attn.wq_a.weight.copy_(ref_attn.wq_a.weight.to(sgl_attn.wq_a.weight.dtype))
        sgl_attn.q_norm.weight.copy_(ref_attn.q_norm.weight)
        sgl_attn.wq_b.weight.copy_(ref_attn.wq_b.weight.to(sgl_attn.wq_b.weight.dtype))
        sgl_attn.wkv.weight.copy_(ref_attn.wkv.weight.to(sgl_attn.wkv.weight.dtype))
        sgl_attn.kv_norm.weight.copy_(ref_attn.kv_norm.weight)
        sgl_attn.wo_a.weight.copy_(ref_attn.wo_a.weight.to(sgl_attn.wo_a.weight.dtype))
        sgl_attn.wo_b.weight.copy_(ref_attn.wo_b.weight.to(sgl_attn.wo_b.weight.dtype))
        sgl_attn.attn_sink.copy_(ref_attn.attn_sink)


def _copy_stage_hc_mixing(*, ref_block, sgl_stage) -> None:
    """Copy the SoT Block mHC mixing params into the SGLang stage (verbatim names).

    The SGLang stage inherits ``hc_attn_fn``/``hc_ffn_fn``/``hc_attn_base``/``hc_ffn_base``/
    ``hc_attn_scale``/``hc_ffn_scale`` from DeepseekV4DecoderLayer and the per-block
    ``input_layernorm``/``post_attention_layernorm`` (= SoT ``attn_norm``/``ffn_norm``).
    All are 1:1.
    """
    with torch.no_grad():
        sgl_stage.hc_attn_fn.copy_(ref_block.hc_attn_fn)
        sgl_stage.hc_ffn_fn.copy_(ref_block.hc_ffn_fn)
        sgl_stage.hc_attn_base.copy_(ref_block.hc_attn_base)
        sgl_stage.hc_ffn_base.copy_(ref_block.hc_ffn_base)
        sgl_stage.hc_attn_scale.copy_(ref_block.hc_attn_scale)
        sgl_stage.hc_ffn_scale.copy_(ref_block.hc_ffn_scale)
        sgl_stage.input_layernorm.weight.copy_(ref_block.attn_norm.weight)
        sgl_stage.post_attention_layernorm.weight.copy_(ref_block.ffn_norm.weight)


def _copy_stage_ffn(*, ref_block, sgl_stage) -> None:
    """Copy the SoT single-expert SwiGLU FFN into the SGLang stage MoE (shared expert).

    The granularity-B block forward runs the stage FFN; the tiny config's SoT FFN is one
    SwiGLU expert (w1/w2/w3). The SGLang stage's MoE shared expert carries the same
    gate_up_proj/down_proj weights. Only used by the block-forward harness (the component
    tests do not run the FFN); copies into the shared-expert weights when present.
    """
    mlp = getattr(sgl_stage, "mlp", None)
    shared = getattr(mlp, "shared_experts", None) if mlp is not None else None
    if shared is None:
        return
    gate_up = getattr(shared, "gate_up_proj", None)
    down = getattr(shared, "down_proj", None)
    if gate_up is None or down is None:
        return
    with torch.no_grad():
        inter = ref_block.ffn.w1.weight.shape[0]
        gate_up.weight[:inter].copy_(ref_block.ffn.w1.weight.to(gate_up.weight.dtype))
        gate_up.weight[inter : 2 * inter].copy_(
            ref_block.ffn.w3.weight.to(gate_up.weight.dtype)
        )
        down.weight.copy_(ref_block.ffn.w2.weight.to(down.weight.dtype))


def sync_sot_to_sgl_dsv4(*, ref, sgl, config, sync_ffn: bool = False) -> None:
    """Sync the vendored SoT ``RefTransformer`` weights into the SGLang dsv4 draft.

    The dsv4 analog of the dense ``_sync_ref_to_sgl_kv_projections``, mapping the full
    MLA-LoRA + mHC + DSpark-head weight surface:

      * per stage: the MLA-LoRA attention weights (``_copy_attn_lora_weights``) and the
        mHC mixing params + per-block norms (``_copy_stage_hc_mixing``);
      * stage 0: the target-hidden projection ``main_proj`` (ReplicatedLinear) +
        ``main_norm``;
      * last stage: the hc_head params (``hc_head_fn``/``hc_head_base``/``hc_head_scale``)
        and the final ``norm``;
      * the Markov head ``markov_w1`` (embedding, padding-robust) + ``markov_w2`` (fp32
        linear);
      * the confidence head ``proj`` (when enabled);
      * optionally (block-forward harness only) the stage FFN.

    The shared embed / lm_head are attached separately by the caller (the vocab-parallel
    modules the worker wires). The RoPE table is config-derived (original_seq_len forced 0
    on both sides), so it is NOT copied; it is already identical. Both sides run tp=1.
    """
    for stage_id, sgl_stage in enumerate(sgl.stages):
        ref_block = ref.mtp[stage_id]
        _copy_attn_lora_weights(ref_attn=ref_block.attn, sgl_attn=sgl_stage.self_attn)
        _copy_stage_hc_mixing(ref_block=ref_block, sgl_stage=sgl_stage)
        if sync_ffn:
            _copy_stage_ffn(ref_block=ref_block, sgl_stage=sgl_stage)

    stage0_ref = ref.mtp[0]
    sgl_stage0 = sgl.stages[0]
    last_ref = ref.mtp[-1]
    sgl_last = sgl.stages[-1]
    org_vocab = int(config.vocab_size)
    with torch.no_grad():
        sgl_stage0.main_proj.weight.copy_(
            stage0_ref.main_proj.weight.to(sgl_stage0.main_proj.weight.dtype)
        )
        sgl_stage0.main_norm.weight.copy_(stage0_ref.main_norm.weight)

        sgl_last.hc_head_fn.copy_(last_ref.hc_head_fn)
        sgl_last.hc_head_base.copy_(last_ref.hc_head_base)
        sgl_last.hc_head_scale.copy_(last_ref.hc_head_scale)
        sgl_last.norm.weight.copy_(last_ref.norm.weight)

        # markov_w1 is a VocabParallelEmbedding (may carry padding rows); copy only the
        # real-vocab rows. markov_w2 is a plain fp32 Linear -> direct copy.
        sgl.markov_head.markov_w1.weight[:org_vocab].copy_(
            last_ref.markov_head.markov_w1.weight
        )
        sgl.markov_head.markov_w2.weight.copy_(last_ref.markov_head.markov_w2.weight)

        if sgl.confidence_head is not None:
            sgl.confidence_head.proj.weight.copy_(
                last_ref.confidence_head.proj.weight.to(
                    sgl.confidence_head.proj.weight.dtype
                )
            )
            if sgl.confidence_head.proj.bias is not None:
                sgl.confidence_head.proj.bias.zero_()

        # Allocator hygiene: the SoT model and a few SGLang modules allocate weights with
        # torch.empty (the checkpoint-load contract); any element not overwritten above
        # keeps uninitialized GPU memory, which can be NaN/inf depending on allocator
        # history and makes the parity harness order-dependent (a prior test's freed
        # buffers leak in). Map non-finite garbage to 0 on BOTH sides so equal positions
        # stay equal; finite synced weights and the config-derived RoPE table are untouched.
        for module in (ref, sgl):
            for tensor in list(module.parameters()) + list(module.buffers()):
                if tensor.is_floating_point():
                    tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


def attach_shared_modules_from_ref(*, sgl, ref, config, device) -> None:
    """Build the shared embed_tokens / lm_head with the SoT weights and attach them.

    The dsv4 draft reuses the target model's token embedding (for the noise-block embed)
    and lm_head (for the base logits). The worker attaches them via
    ``attach_shared_modules``; here we build minimal vocab-parallel modules carrying the
    SoT ``embed``/``head`` weights so the parity inputs match. The vocab-parallel weights
    may carry padding rows beyond org_vocab_size; only the real-vocab rows are copied.
    """
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    embed = VocabParallelEmbedding(
        config.vocab_size, config.hidden_size, enable_tp=False
    ).to(device)
    lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, enable_tp=False).to(
        device
    )
    org_vocab = int(config.vocab_size)
    with torch.no_grad():
        # The vendored SoT embed/head weights are torch.empty (the SoT model expects a
        # checkpoint load); the parity harness uses random weights, so initialize them
        # deterministically here. Left as-is they inherit uninitialized GPU memory, which
        # can be NaN depending on allocator history -- a prior test's freed buffers leak
        # in and the harness becomes order-dependent.
        generator = torch.Generator(device=device).manual_seed(0)
        ref.embed.weight.normal_(generator=generator)
        ref.head.weight.normal_(generator=generator)
        embed.weight.zero_()
        lm_head.weight.zero_()
        embed.weight[:org_vocab].copy_(ref.embed.weight.to(embed.weight.dtype))
        lm_head.weight[:org_vocab].copy_(ref.head.weight.to(lm_head.weight.dtype))
    sgl.attach_shared_modules(embed_tokens=embed, lm_head=lm_head)


def force_native_ops(model: torch.nn.Module) -> None:
    """Route every MultiPlatformOp to its native (platform-agnostic) fp32-safe forward."""
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    for module in model.modules():
        if isinstance(module, MultiPlatformOp):
            module._forward_method = module.forward_native


__all__ = [
    "make_tiny_dsv4_config",
    "make_ref_args_from_config",
    "sync_sot_to_sgl_dsv4",
    "attach_shared_modules_from_ref",
    "force_native_ops",
]
