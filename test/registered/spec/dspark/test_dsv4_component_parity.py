import functools
import os
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()

# Granularity-A components are pure linear (no through-sparse-attention fp8), so the
# dense parity tolerance (test_dspark_model_parity.py) applies tight.
_ATOL = 1e-4
_RTOL = 1e-4

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _ensure_repo_test_package() -> None:
    """Put repo root on sys.path and evict the stdlib ``test`` package shadow."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    for name in [m for m in list(sys.modules) if m == "test" or m.startswith("test.")]:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", "") or ""
        if not file.startswith(_REPO_ROOT + os.sep):
            del sys.modules[name]


def _requires_cuda(test_method):
    """Decorator: skip when CUDA is unavailable (the dsv4 backbone is GPU-only)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; dsv4 sparse backend is GPU-only.")
        return test_method(self, *args, **kwargs)

    return wrapper


def _setup_sglang_runtime() -> None:
    """Set global server args and a single-rank model-parallel group (tp=1).

    The dsv4 DSpark backbone builds VocabParallelEmbedding / TP-sharded projections
    that require the model-parallel group, and reads get_global_server_args; both must
    exist before the SGLang model is constructed. Mirrors the dense parity test.
    """
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29653")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="nccl",
        )


def _force_native_ops(model: torch.nn.Module) -> None:
    """Route every MultiPlatformOp to its native (platform-agnostic) forward.

    The SGLang norm/activation CUDA kernels do not dispatch fp32, and this parity
    oracle keeps the float32 path; the native forward is fp32-safe. Mirrors the dense
    parity test's ``_force_native_ops``.
    """
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    for module in model.modules():
        if isinstance(module, MultiPlatformOp):
            module._forward_method = module.forward_native


def make_tiny_dsv4_config(*, enable_confidence_head: bool = False):
    """Build a tiny ``DeepSeekV4Config`` carrying the dsv4 DSpark draft fields.

    Mirrors the dense ``_TinyQwen3Config``: a real production config subclass with dims
    shrunk to tiny values and the ``dspark_*`` prefixed draft fields attached (the
    ``parse_dspark_draft_config`` prefixed convention). head_dim is set explicitly
    (qk_rope_head_dim + qk_nope_head_dim) so the DSparkAttention head-dim assert holds,
    and num_key_value_heads == 1 (single MLA latent), as the draft attention requires.
    """
    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config

    qk_rope_head_dim = 8
    qk_nope_head_dim = 8
    head_dim = qk_rope_head_dim + qk_nope_head_dim
    hidden_size = 32
    num_attention_heads = 4
    config = DeepSeekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=head_dim,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        window_size=16,
        intermediate_size=64,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        vocab_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000,
        rope_scaling={},
        compress_ratios=[],
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )
    config.dspark_block_size = 3
    config.dspark_markov_rank = 8
    config.dspark_noise_token_id = 1
    config.dspark_target_layer_ids = [0]
    config.temperature = 0.0
    config.enable_confidence_head = enable_confidence_head
    config.confidence_head_with_markov = enable_confidence_head
    return config


def make_ref_args_from_config(config):
    """Build the vendored SoT ``RefModelArgs`` matching the tiny SGLang config dims.

    The reference oracle and the production draft must share identical dims so the
    weight sync is a 1:1 copy and the only remaining difference is the math impl.
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

    The dsv4 draft attention is MLA-LoRA, NOT the qwen-style fused q/k/v_proj: the
    weights are ``wq_a`` (down) -> ``q_norm`` -> ``wq_b`` (up, per-head) for Q, ``wkv``
    (latent down) -> ``kv_norm`` for the single MLA KV latent, and a grouped low-rank
    output ``wo_a`` (bf16) -> ``wo_b``. At tp=1 the SGLang ColumnParallel/RowParallel
    projections hold the full (unsharded) weight, so each is a direct ``.weight`` copy.
    ``attn_sink`` is shape ``n_heads`` (not ``n_local_heads``); at tp=1 the whole vector
    copies across. There is no fused ``wqkv_a`` on the draft path (separate
    ReplicatedLinear wq_a / wkv), so no split is needed.
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

    The SGLang stage inherits ``hc_attn_fn``/``hc_ffn_fn``/``hc_attn_base``/
    ``hc_ffn_base``/``hc_attn_scale``/``hc_ffn_scale`` from DeepseekV4DecoderLayer and
    the per-block ``input_layernorm``/``post_attention_layernorm`` (= SoT
    ``attn_norm``/``ffn_norm``). All are 1:1.
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


def sync_sot_to_sgl_dsv4(*, ref, sgl, config) -> None:
    """Sync the vendored SoT ``RefTransformer`` weights into the SGLang dsv4 draft.

    This is the dsv4 analog of the dense ``_sync_ref_to_sgl_kv_projections``, but the
    dsv4 draft has the full MLA-LoRA + mHC + DSpark-head weight surface to mirror:

      * per stage: the MLA-LoRA attention weights (``_copy_attn_lora_weights``) and the
        mHC mixing params + per-block norms (``_copy_stage_hc_mixing``);
      * stage 0: the target-hidden projection ``main_proj`` (ReplicatedLinear) +
        ``main_norm``;
      * last stage: the hc_head params (``hc_head_fn``/``hc_head_base``/
        ``hc_head_scale``) and the final ``norm``;
      * the Markov head ``markov_w1`` (embedding) + ``markov_w2`` (fp32 linear);
      * the confidence head ``proj`` (when enabled);
      * the shared embed / lm_head: the SoT ``embed``/``head`` weights are mirrored onto
        the modules the worker attaches to the SGLang draft (``attach_shared_modules``).

    The RoPE table is config-derived (original_seq_len forced 0 on both sides), so it is
    NOT copied; it is already identical. Both sides run tp=1 + native ops.
    """
    for stage_id, sgl_stage in enumerate(sgl.stages):
        ref_block = ref.mtp[stage_id]
        _copy_attn_lora_weights(ref_attn=ref_block.attn, sgl_attn=sgl_stage.self_attn)
        _copy_stage_hc_mixing(ref_block=ref_block, sgl_stage=sgl_stage)

    stage0_ref = ref.mtp[0]
    sgl_stage0 = sgl.stages[0]
    with torch.no_grad():
        sgl_stage0.main_proj.weight.copy_(
            stage0_ref.main_proj.weight.to(sgl_stage0.main_proj.weight.dtype)
        )
        sgl_stage0.main_norm.weight.copy_(stage0_ref.main_norm.weight)

        last_ref = ref.mtp[-1]
        sgl_last = sgl.stages[-1]
        sgl_last.hc_head_fn.copy_(last_ref.hc_head_fn)
        sgl_last.hc_head_base.copy_(last_ref.hc_head_base)
        sgl_last.hc_head_scale.copy_(last_ref.hc_head_scale)
        sgl_last.norm.weight.copy_(last_ref.norm.weight)

        org_vocab = int(config.vocab_size)
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


def _attach_shared_modules_from_ref(*, sgl, ref, config, device) -> None:
    """Build the shared embed_tokens / lm_head with the SoT weights and attach them.

    The dsv4 draft reuses the target model's token embedding (for the noise-block embed)
    and lm_head (for the base logits). The worker attaches them via
    ``attach_shared_modules``; here we build minimal modules carrying the SoT
    ``embed``/``head`` weights so the parity inputs match.
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
        # The vocab-parallel weights may carry padding rows beyond org_vocab_size; copy
        # only the real-vocab rows so the parity inputs match (the rest is unused).
        embed.weight[:org_vocab].copy_(ref.embed.weight.to(embed.weight.dtype))
        lm_head.weight[:org_vocab].copy_(ref.head.weight.to(lm_head.weight.dtype))
    sgl.attach_shared_modules(embed_tokens=embed, lm_head=lm_head)


class TestDsv4ComponentParity(CustomTestCase):
    """Granularity-A (category 1): per-component dsv4 draft parity vs the SoT oracle.

    Builds the SGLang ``DeepseekV4ForCausalLMDSpark`` and the vendored
    ``RefTransformer`` with identical random weights (``sync_sot_to_sgl_dsv4``), then
    feeds each pure-linear component the SAME input and asserts the outputs match
    tight (1e-4, greedy exact). These avoid the through-sparse-attention fp8 path
    (granularity B), so they isolate the MLA-LoRA weight mapping and the head math:

      * kv-projection (isolates the MLA-LoRA weight mapping FIRST);
      * target-hidden projection (main_proj + main_norm);
      * hc_head collapse (the dsv4-specific PRE-norm collapse the dense path lacks);
      * markov head (serial bias-then-sample on the collapsed base logits);
      * confidence head (post-hc_head PRE-norm tap).
    """

    enable_confidence_head: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        cls._skip_reason = None
        if not _CUDA_AVAILABLE:
            cls._skip_reason = "CUDA not available; dsv4 backbone is GPU-only."
            return
        try:
            _ensure_repo_test_package()
            from test.srt.speculative._dspark_reference.deepseek_v4.modeling import (
                RefTransformer,
            )

            from sglang.srt.models.deepseek_v4_dspark import (
                DeepseekV4ForCausalLMDSpark,
            )
            from sglang.srt.runtime_context import get_parallel
        except ImportError as exc:
            cls._skip_reason = f"Import error: {exc}"
            return

        _setup_sglang_runtime()
        device = torch.device("cuda")
        cls.device = device
        config = make_tiny_dsv4_config(
            enable_confidence_head=cls.enable_confidence_head
        )
        cls.config = config

        torch.manual_seed(60)
        ref_args = make_ref_args_from_config(config)
        ref = RefTransformer(ref_args).to(device).eval()
        with get_parallel().override(tp_size=1, tp_rank=0):
            sgl = DeepseekV4ForCausalLMDSpark(config=config).to(device).eval()
        _force_native_ops(sgl)

        sync_sot_to_sgl_dsv4(ref=ref, sgl=sgl, config=config)
        _attach_shared_modules_from_ref(sgl=sgl, ref=ref, config=config, device=device)

        cls.ref = ref
        cls.sgl = sgl

    def _skip_if_unready(self) -> None:
        if self._skip_reason:
            self.skipTest(self._skip_reason)

    @_requires_cuda
    def test_kv_projection_parity(self) -> None:
        """SGLang kv_proj_only + kv_norm matches the SoT kv_norm(wkv(.)) (weight map)."""
        self._skip_if_unready()
        config = self.config
        ctx_len = 5
        torch.manual_seed(61)
        ctx_hidden = torch.randn(ctx_len, config.hidden_size, device=self.device)

        for stage_id, sgl_stage in enumerate(self.sgl.stages):
            attn = sgl_stage.self_attn
            ref_attn = self.ref.mtp[stage_id].attn
            with torch.no_grad():
                sgl_kv = attn.kv_norm(attn.kv_proj_only(ctx_hidden))
                ref_kv = ref_attn.kv_norm(ref_attn.wkv(ctx_hidden))
            torch.testing.assert_close(
                sgl_kv.float(), ref_kv.float(), atol=_ATOL, rtol=_RTOL
            )

    @_requires_cuda
    def test_target_hidden_projection_parity(self) -> None:
        """SGLang project_target_hidden matches the SoT main_norm(main_proj(.))."""
        self._skip_if_unready()
        config = self.config
        ctx_len = 5
        n_features = len(config.dspark_target_layer_ids)
        torch.manual_seed(62)
        main_hidden = torch.randn(
            ctx_len, n_features * config.hidden_size, device=self.device
        )

        with torch.no_grad():
            sgl_ctx = self.sgl.project_target_hidden(main_hidden)
            ref_ctx = self.ref.project_target_hidden(main_hidden)
        self.assertEqual(sgl_ctx.shape, (ctx_len, config.hidden_size))
        torch.testing.assert_close(
            sgl_ctx.float(), ref_ctx.float(), atol=_ATOL, rtol=_RTOL
        )

    @_requires_cuda
    def test_hc_head_collapse_parity(self) -> None:
        """SGLang collapse_hc_head matches the SoT hc_head(.) (the PRE-norm collapse)."""
        self._skip_if_unready()
        config = self.config
        n = 6
        torch.manual_seed(63)
        x = torch.randn(n, config.hc_mult, config.hidden_size, device=self.device)

        with torch.no_grad():
            sgl_post_hc = self.sgl.collapse_hc_head(x)
            ref_post_hc = self.ref.collapse_hc_head(x.unsqueeze(0)).squeeze(0)
        self.assertEqual(sgl_post_hc.shape, (n, config.hidden_size))
        torch.testing.assert_close(
            sgl_post_hc.float(), ref_post_hc.float(), atol=_ATOL, rtol=_RTOL
        )

    @_requires_cuda
    def test_base_logits_from_hidden_parity(self) -> None:
        """SGLang hc_head -> norm -> lm_head matches the SoT full-vocab base logits."""
        self._skip_if_unready()
        config = self.config
        n = 6
        torch.manual_seed(64)
        x = torch.randn(n, config.hc_mult, config.hidden_size, device=self.device)

        with torch.no_grad():
            sgl_logits = self.sgl.compute_base_logits(x)
            ref_logits = self.ref.base_logits_from_hidden(x.unsqueeze(0)).squeeze(0)
        self.assertEqual(sgl_logits.shape, (n, config.vocab_size))
        torch.testing.assert_close(
            sgl_logits.float(), ref_logits.float(), atol=_ATOL, rtol=_RTOL
        )
        torch.testing.assert_close(sgl_logits.argmax(dim=-1), ref_logits.argmax(dim=-1))

    @_requires_cuda
    def test_markov_head_serial_correction_parity(self) -> None:
        """SGLang Markov sample_block matches the SoT serial bias-then-sample loop."""
        self._skip_if_unready()
        config = self.config
        bs = 2
        gamma = config.dspark_block_size
        vocab = config.vocab_size
        torch.manual_seed(65)
        base_logits = torch.randn(bs, gamma, vocab, device=self.device)
        first_prev = torch.randint(0, vocab, (bs,), device=self.device)

        def greedy_sampler(step_logits, step_idx):
            return step_logits.argmax(dim=-1)

        with torch.no_grad():
            sgl_sampled, sgl_corrected = self.sgl.markov_head.sample_block(
                base_logits,
                first_prev_tokens=first_prev,
                hidden_states=None,
                sampler=greedy_sampler,
            )
            ref_head = self.ref.mtp[-1].markov_head
            ref_corrected = []
            prev = first_prev.long()
            for step_idx in range(gamma):
                bias, _ = ref_head(prev)
                step_logits = base_logits[:, step_idx, :] + bias
                ref_corrected.append(step_logits.unsqueeze(1))
                prev = step_logits.argmax(dim=-1)
            ref_corrected = torch.cat(ref_corrected, dim=1)

        torch.testing.assert_close(
            sgl_corrected.float(), ref_corrected.float(), atol=_ATOL, rtol=_RTOL
        )
        torch.testing.assert_close(sgl_sampled, ref_corrected.argmax(dim=-1))


class TestDsv4ComponentParityWithConfidence(TestDsv4ComponentParity):
    """Granularity-A parity with the confidence head enabled (post-hc_head PRE-norm tap)."""

    enable_confidence_head = True

    @_requires_cuda
    def test_confidence_head_parity(self) -> None:
        """SGLang compute_confidence matches the SoT confidence on the same tap.

        The dsv4 confidence tap is the post-hc_head PRE-norm draft hidden (reference
        model.py:873: confidence_head(x, markov_embed) where x = hc_head(.) BEFORE the
        norm), with the per-step markov_embed built from the prev-token sequence
        [anchor, s_0, ..., s_{gamma-2}]. The dense worker's post-norm tap would be the
        wrong tap for V4, so this pins the correct one.
        """
        self._skip_if_unready()
        config = self.config
        bs = 2
        gamma = config.dspark_block_size
        vocab = config.vocab_size
        d = config.hidden_size
        torch.manual_seed(66)

        # Stash the post-hc_head PRE-norm tap on the model exactly as a real forward
        # would (compute_base_logits stashes self._x_post_hc), then drive the worker
        # confidence hook with explicit anchor + sampled tokens.
        x_post_hc = torch.randn(bs * gamma, d, device=self.device)
        self.sgl._x_post_hc = x_post_hc
        self.sgl.gamma = gamma
        anchor = torch.randint(0, vocab, (bs,), device=self.device)
        sampled = torch.randint(0, vocab, (bs, gamma), device=self.device)

        with torch.no_grad():
            sgl_conf = self.sgl.compute_confidence(
                anchor_tokens=anchor, sampled_tokens=sampled
            )
            # SoT confidence: cat([x_post_hc, markov_embed], -1) -> proj -> sigmoid.
            ref_head = self.ref.mtp[-1]
            prev_seq = torch.cat([anchor.view(-1, 1), sampled[:, : gamma - 1]], dim=1)
            ref_markov_embed = ref_head.markov_head.markov_w1(prev_seq)
            ref_x = x_post_hc.view(bs, gamma, d)
            ref_conf_raw = ref_head.confidence_head(ref_x, ref_markov_embed)
            ref_conf = torch.sigmoid(ref_conf_raw.float())

        self.assertIsNotNone(sgl_conf)
        self.assertEqual(sgl_conf.shape, (bs, gamma))
        torch.testing.assert_close(
            sgl_conf.float(), ref_conf.float(), atol=_ATOL, rtol=_RTOL
        )


if __name__ == "__main__":
    unittest.main()
