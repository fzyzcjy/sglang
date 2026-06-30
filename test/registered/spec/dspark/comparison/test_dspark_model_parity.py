import functools
import os
import sys
import unittest

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()
_ATOL_HIDDEN = 1e-4
_RTOL_HIDDEN = 1e-4
_ATOL_LOGITS = 1e-4
_RTOL_LOGITS = 1e-4

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _ensure_repo_test_package() -> None:
    """Put repo root on sys.path and evict the stdlib ``test`` package.

    CI runs each file as ``python3 <path>`` (repo root absent from sys.path) and
    Python ships a stdlib ``test`` package that shadows the repo's ``test``
    namespace; both must be corrected before importing the vendored reference.
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    for name in [m for m in list(sys.modules) if m == "test" or m.startswith("test.")]:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", "") or ""
        if not file.startswith(_REPO_ROOT + os.sep):
            del sys.modules[name]


def _requires_cuda(test_method):
    """Decorator: skip test if CUDA is not available."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available")
        return test_method(self, *args, **kwargs)

    return wrapper


def _setup_sglang_runtime() -> None:
    """Set global server args and a single-rank model-parallel group.

    The SGLang DSpark backbone reads get_global_server_args() (RotaryEmbedding)
    and builds VocabParallelEmbedding / TP-sharded projections that require the
    model-parallel group; both must exist before the SGLang model is constructed.
    """
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29651")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
        )

    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="nccl",
        )


def _force_native_ops(model: torch.nn.Module) -> None:
    """Route every MultiPlatformOp to its native forward.

    The SGLang norm/activation CUDA kernels do not dispatch float32, and this
    fp32 parity oracle keeps both backbones in float32 to match the HF reference
    exactly; the native path is platform-agnostic float32.
    """
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    for module in model.modules():
        if isinstance(module, MultiPlatformOp):
            module._forward_method = module.forward_native


class _TinyQwen3Config(Qwen3Config):
    """Minimal Qwen3-like config for DSpark model parity tests.

    Subclasses the real HF config so the DeepSpec reference backbone (an HF
    PreTrainedModel) accepts it, while overriding the dims to tiny values and
    attaching the DSpark-specific draft fields.
    """

    model_type = "qwen3"

    def __init__(self, markov_head_type: str = "vanilla", markov_rank: int = 8) -> None:
        super().__init__(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64 // 4,
            intermediate_size=128,
            vocab_size=256,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=512,
            rope_theta=10000.0,
            sliding_window=None,
            hidden_act="silu",
        )
        self.layer_types = ["full_attention"] * self.num_hidden_layers

        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type

        self.target_layer_ids = [0]
        self.block_size = 3
        self.mask_token_id = 1
        self.num_anchors = 2
        self.enable_confidence_head = False

        self._attn_implementation = "eager"


class _TinyGemma4Config(Gemma4TextConfig):
    """Minimal Gemma4-like config for DSpark model parity tests.

    Subclasses the real HF config so the DeepSpec reference backbone (an HF
    PreTrainedModel) accepts it, while overriding the dims to tiny values and
    attaching the DSpark-specific draft fields.
    """

    model_type = "gemma4"

    def __init__(
        self,
        markov_head_type: str = "vanilla",
        markov_rank: int = 8,
        attention_k_eq_v: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            global_head_dim=16,
            num_global_key_value_heads=4,
            intermediate_size=128,
            vocab_size=256,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=512,
            rope_theta=10000.0,
            attention_k_eq_v=attention_k_eq_v,
            final_logit_softcapping=None,
            enable_moe_block=False,
            hidden_size_per_layer_input=0,
            head_dim=16,
            hidden_activation="gelu_pytorch_tanh",
        )
        self.query_pre_attn_scalar = 1.0

        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type

        self.target_layer_ids = [0]
        self.block_size = 3
        self.mask_token_id = 1
        self.num_anchors = 2
        self.enable_confidence_head = False

        self._attn_implementation = "eager"


def _sync_weights_by_name(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy matching named parameters from src into dst."""
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    with torch.no_grad():
        for name, param in src_params.items():
            if name in dst_params:
                dst_params[name].copy_(param)


def _sync_ref_to_sgl_kv_projections(
    *,
    ref_layer: torch.nn.Module,
    sgl_attn: torch.nn.Module,
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    has_v_proj: bool,
) -> None:
    """Copy reference per-layer attention weights into the SGLang fused qkv_proj.

    The reference uses separate ``q_proj``/``k_proj``/``v_proj`` while the SGLang
    draft attention fuses them into a single ``qkv_proj`` (q rows, then k, then
    v). When the reference shares K and V (``attention_k_eq_v``) it has no
    ``v_proj``; the SGLang fused weight still carries a v block, so we mirror the
    reference K weight into it to keep ``kv_proj_only`` consistent.

    Both backbones apply RMSNorm as ``norm(x) * w``: the HF gemma4 norm in this
    transformers version drops the legacy ``(1 + w)`` convention and the SGLang
    gemma norm uses ``scale_shift=0``, so the norm weights copy across directly.
    """
    ref_attn = ref_layer.self_attn
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    with torch.no_grad():
        fused = sgl_attn.qkv_proj.weight
        fused[:q_size].copy_(ref_attn.q_proj.weight)
        fused[q_size : q_size + kv_size].copy_(ref_attn.k_proj.weight)
        v_weight = ref_attn.v_proj.weight if has_v_proj else ref_attn.k_proj.weight
        fused[q_size + kv_size : q_size + 2 * kv_size].copy_(v_weight)
        sgl_attn.q_norm.weight.copy_(ref_attn.q_norm.weight)
        sgl_attn.k_norm.weight.copy_(ref_attn.k_norm.weight)


def _reference_ctx_kv(
    *,
    ref_attn: torch.nn.Module,
    ctx_hidden: torch.Tensor,
    num_key_value_heads: int,
    head_dim: int,
    has_v_proj: bool,
    apply_v_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ctx K/V (projection + per-head K/V norm, no rope).

    Mirrors the worker injection path (``kv_proj_only`` -> ``apply_k_norm`` ->
    ``apply_v_norm``) on the independent reference modules so the two can be
    compared head-for-head.
    """
    ctx_len = ctx_hidden.shape[0]
    k = ref_attn.k_proj(ctx_hidden).view(ctx_len, num_key_value_heads, head_dim)
    if has_v_proj:
        v = ref_attn.v_proj(ctx_hidden).view(ctx_len, num_key_value_heads, head_dim)
    else:
        v = k.clone()
    k = ref_attn.k_norm(k)
    if apply_v_norm:
        v = ref_attn.v_norm(v)
    return k, v


class TestQwen3DSparkModelParity(CustomTestCase):
    """Worker-faithful parity: SGLang Qwen3DSparkModel vs DeepSpec reference.

    Builds both the SGLang DSpark draft backbone and the independent reference
    backbone with identical random weights, then verifies the KV-injection
    equivalence (the #1 risk per plan §8/§11): per-layer ctx K/V projections,
    the fc/hidden_norm context projection, and draft logits/greedy tokens.
    GPU-only (the SGLang draft attention needs CUDA for rotary embedding).
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _CUDA_AVAILABLE:
            return
        try:
            _ensure_repo_test_package()
            from test.srt.speculative._dspark_reference.qwen3.modeling import (
                Qwen3DSparkModel as RefQwen3DSparkModel,
            )

            from sglang.srt.models.dspark import Qwen3DSparkModel as SglQwen3DSparkModel
            from sglang.srt.runtime_context import get_parallel
        except ImportError as exc:
            cls._import_error = str(exc)
            return
        cls._import_error = None

        _setup_sglang_runtime()
        cfg = _TinyQwen3Config(markov_head_type="vanilla", markov_rank=8)
        device = torch.device("cuda")
        cls.device = device
        cls.cfg = cfg

        torch.manual_seed(10)
        ref_model = RefQwen3DSparkModel(cfg).to(device).eval()
        with get_parallel().override(tp_size=1, tp_rank=0):
            sgl_model = SglQwen3DSparkModel(config=cfg).to(device).eval()
        _force_native_ops(sgl_model)

        for layer_id, sgl_layer in enumerate(sgl_model.layers):
            _sync_ref_to_sgl_kv_projections(
                ref_layer=ref_model.layers[layer_id],
                sgl_attn=sgl_layer.self_attn,
                head_dim=cfg.head_dim,
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                has_v_proj=True,
            )
        with torch.no_grad():
            sgl_model.fc.weight.copy_(ref_model.fc.weight)
            sgl_model.hidden_norm.weight.copy_(ref_model.hidden_norm.weight)

        cls.ref_model = ref_model
        cls.sgl_model = sgl_model
        cls.lm_head_weight = ref_model.lm_head.weight.detach().clone()

    @_requires_cuda
    def test_ctx_kv_injection_parity(self):
        """SGLang ctx K/V (kv_proj_only+norms) matches the reference projections."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        cfg = self.cfg
        ctx_len = 5
        torch.manual_seed(21)
        ctx_hidden = torch.randn(ctx_len, cfg.hidden_size, device=self.device)

        for layer_id, sgl_layer in enumerate(self.sgl_model.layers):
            attn = sgl_layer.self_attn
            with torch.no_grad():
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k).view(ctx_len, attn.num_kv_heads, attn.head_dim)
                v = attn.apply_v_norm(v).view(ctx_len, attn.num_kv_heads, attn.head_dim)
                ref_k, ref_v = _reference_ctx_kv(
                    ref_attn=self.ref_model.layers[layer_id].self_attn,
                    ctx_hidden=ctx_hidden,
                    num_key_value_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    has_v_proj=True,
                    apply_v_norm=False,
                )
            torch.testing.assert_close(k, ref_k, atol=_ATOL_HIDDEN, rtol=_RTOL_HIDDEN)
            torch.testing.assert_close(v, ref_v, atol=_ATOL_HIDDEN, rtol=_RTOL_HIDDEN)

    @_requires_cuda
    def test_fc_hidden_norm_projection_parity(self):
        """SGLang project_target_hidden matches reference hidden_norm(fc(.))."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        cfg = self.cfg
        ctx_len = 5
        n_features = len(cfg.target_layer_ids)
        torch.manual_seed(22)
        target_hidden = torch.randn(
            ctx_len, n_features * cfg.hidden_size, device=self.device
        )

        with torch.no_grad():
            sgl_ctx = self.sgl_model.project_target_hidden(target_hidden)
            ref_ctx = self.ref_model.hidden_norm(self.ref_model.fc(target_hidden))
        self.assertEqual(sgl_ctx.shape, (ctx_len, cfg.hidden_size))
        torch.testing.assert_close(
            sgl_ctx, ref_ctx, atol=_ATOL_HIDDEN, rtol=_RTOL_HIDDEN
        )

    @_requires_cuda
    def test_draft_logits_greedy_token_parity(self):
        """Draft logits (lm_head matmul) and greedy tokens match the reference."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        cfg = self.cfg
        bs = 4
        torch.manual_seed(23)
        draft_hidden = torch.randn(bs, cfg.hidden_size, device=self.device)
        weight = self.lm_head_weight.to(self.device)

        with torch.no_grad():
            sgl_logits = torch.matmul(draft_hidden.to(weight.dtype), weight.T)
            ref_logits = self.ref_model.compute_logits(draft_hidden)
        torch.testing.assert_close(
            sgl_logits, ref_logits, atol=_ATOL_LOGITS, rtol=_RTOL_LOGITS
        )
        torch.testing.assert_close(sgl_logits.argmax(dim=-1), ref_logits.argmax(dim=-1))

    @_requires_cuda
    def test_markov_head_parity_vanilla(self):
        """SGLang markov head bias matches reference for Qwen3 vanilla config."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        _ensure_repo_test_package()
        from test.srt.speculative._dspark_reference.markov_head import (
            VanillaMarkov as RefVanillaMarkov,
        )

        from sglang.srt.models.dspark import build_markov_head

        cfg = self.cfg
        torch.manual_seed(20)
        sgl_head = build_markov_head(cfg).to(self.device)
        ref_head = RefVanillaMarkov(
            vocab_size=cfg.vocab_size, markov_rank=cfg.markov_rank
        ).to(self.device)
        sgl_head.eval()
        ref_head.eval()
        _sync_weights_by_name(sgl_head, ref_head)

        bs, gamma = 1, cfg.block_size
        base_logits = torch.randn(bs, gamma, cfg.vocab_size, device=self.device)
        token_ids = torch.randint(0, cfg.vocab_size, (bs,), device=self.device)

        with torch.no_grad():
            sgl_out = sgl_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=None
            )
            ref_out = ref_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=None
            )
        torch.testing.assert_close(
            sgl_out, ref_out, atol=_ATOL_LOGITS, rtol=_RTOL_LOGITS
        )

    @_requires_cuda
    def test_draft_probs_row_order_no_anchor_row(self):
        """draft_probs row 0 is q(s_0); there is no anchor row (plan §3 [P0-B])."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        from sglang.srt.models.dspark import build_markov_head

        cfg = self.cfg
        gamma = cfg.block_size
        bs = 1
        vocab = cfg.vocab_size
        torch.manual_seed(30)

        head = build_markov_head(cfg).to(self.device)
        head.eval()

        base_logits = torch.randn(bs, gamma, vocab, device=self.device)
        first_prev = torch.randint(0, vocab, (bs,), device=self.device)

        def greedy_sampler(logits, step_idx):
            return torch.argmax(logits, dim=-1)

        with torch.no_grad():
            sampled, corrected_logits = head.sample_block(
                base_logits,
                first_prev_tokens=first_prev,
                hidden_states=None,
                sampler=greedy_sampler,
            )

        self.assertEqual(sampled.shape, (bs, gamma))
        self.assertEqual(corrected_logits.shape, (bs, gamma, vocab))

        for k in range(gamma):
            expected_token = sampled[0, k].item()
            row_probs = torch.softmax(corrected_logits[0, k], dim=-1)
            best_token = row_probs.argmax().item()
            self.assertEqual(
                best_token,
                expected_token,
                f"Row {k}: argmax of q(s_{k}) should equal sampled s_{k}.",
            )


class TestGemma4DSparkModelParity(CustomTestCase):
    """Worker-faithful parity for Gemma4DSparkModel vs DeepSpec reference.

    Like the Qwen3 case, builds both backbones with identical random weights and
    checks the ctx K/V injection projections, fc/hidden_norm projection, and
    draft logits. Runs once with the standard K/V path and once with
    ``attention_k_eq_v`` so the shared-K/V load branch is covered. GPU-only.
    """

    attention_k_eq_v: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        if not _CUDA_AVAILABLE:
            return
        try:
            _ensure_repo_test_package()
            from test.srt.speculative._dspark_reference.gemma4.modeling import (
                Gemma4DSparkModel as RefGemma4,
            )

            from sglang.srt.models.dspark_gemma import Gemma4DSparkModel as SglGemma4
            from sglang.srt.runtime_context import get_parallel
        except ImportError as exc:
            cls._import_error = str(exc)
            return
        cls._import_error = None

        _setup_sglang_runtime()
        cfg = _TinyGemma4Config(
            markov_head_type="vanilla",
            markov_rank=8,
            attention_k_eq_v=cls.attention_k_eq_v,
        )
        device = torch.device("cuda")
        cls.device = device
        cls.cfg = cfg

        torch.manual_seed(40)
        ref_model = RefGemma4(cfg).to(device).eval()
        with get_parallel().override(tp_size=1, tp_rank=0):
            sgl_model = SglGemma4(config=cfg).to(device).eval()
        _force_native_ops(sgl_model)

        head_dim = cfg.global_head_dim
        num_kv_heads = (
            cfg.num_global_key_value_heads
            if cls.attention_k_eq_v
            else cfg.num_key_value_heads
        )
        for layer_id, sgl_layer in enumerate(sgl_model.layers):
            _sync_ref_to_sgl_kv_projections(
                ref_layer=ref_model.layers[layer_id],
                sgl_attn=sgl_layer.self_attn,
                head_dim=head_dim,
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=num_kv_heads,
                has_v_proj=not cls.attention_k_eq_v,
            )
        with torch.no_grad():
            sgl_model.fc.weight.copy_(ref_model.fc.weight)
            sgl_model.hidden_norm.weight.copy_(ref_model.hidden_norm.weight)

        cls.ref_model = ref_model
        cls.sgl_model = sgl_model
        cls.num_kv_heads = num_kv_heads
        cls.head_dim = head_dim

    @_requires_cuda
    def test_ctx_kv_injection_parity(self):
        """SGLang ctx K/V (kv_proj_only+norms) matches the reference projections."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        cfg = self.cfg
        ctx_len = 5
        torch.manual_seed(41)
        ctx_hidden = torch.randn(ctx_len, cfg.hidden_size, device=self.device)

        for layer_id, sgl_layer in enumerate(self.sgl_model.layers):
            attn = sgl_layer.self_attn
            with torch.no_grad():
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k).view(ctx_len, attn.num_kv_heads, attn.head_dim)
                v = attn.apply_v_norm(v).view(ctx_len, attn.num_kv_heads, attn.head_dim)
                ref_k, ref_v = _reference_ctx_kv(
                    ref_attn=self.ref_model.layers[layer_id].self_attn,
                    ctx_hidden=ctx_hidden,
                    num_key_value_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    has_v_proj=not self.attention_k_eq_v,
                    apply_v_norm=True,
                )
            torch.testing.assert_close(k, ref_k, atol=_ATOL_HIDDEN, rtol=_RTOL_HIDDEN)
            torch.testing.assert_close(v, ref_v, atol=_ATOL_HIDDEN, rtol=_RTOL_HIDDEN)

    @_requires_cuda
    def test_fc_hidden_norm_projection_parity(self):
        """SGLang project_target_hidden matches reference hidden_norm(fc(.))."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        cfg = self.cfg
        ctx_len = 5
        n_features = len(cfg.target_layer_ids)
        torch.manual_seed(42)
        target_hidden = torch.randn(
            ctx_len, n_features * cfg.hidden_size, device=self.device
        )

        with torch.no_grad():
            sgl_ctx = self.sgl_model.project_target_hidden(target_hidden)
            ref_ctx = self.ref_model.hidden_norm(self.ref_model.fc(target_hidden))
        self.assertEqual(sgl_ctx.shape, (ctx_len, cfg.hidden_size))
        torch.testing.assert_close(
            sgl_ctx, ref_ctx, atol=_ATOL_HIDDEN, rtol=_RTOL_HIDDEN
        )

    @_requires_cuda
    def test_markov_head_parity_vanilla(self):
        """SGLang markov head bias matches reference for Gemma4 vanilla config."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        _ensure_repo_test_package()
        from test.srt.speculative._dspark_reference.markov_head import (
            VanillaMarkov as RefVanillaMarkov,
        )

        from sglang.srt.models.dspark import build_markov_head

        cfg = self.cfg
        torch.manual_seed(50)
        sgl_head = build_markov_head(cfg).to(self.device)
        ref_head = RefVanillaMarkov(
            vocab_size=cfg.vocab_size, markov_rank=cfg.markov_rank
        ).to(self.device)
        sgl_head.eval()
        ref_head.eval()
        _sync_weights_by_name(sgl_head, ref_head)

        bs, gamma = 1, cfg.block_size
        base_logits = torch.randn(bs, gamma, cfg.vocab_size, device=self.device)
        token_ids = torch.randint(0, cfg.vocab_size, (bs,), device=self.device)

        with torch.no_grad():
            sgl_out = sgl_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=None
            )
            ref_out = ref_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=None
            )
        torch.testing.assert_close(
            sgl_out, ref_out, atol=_ATOL_LOGITS, rtol=_RTOL_LOGITS
        )

    @_requires_cuda
    def test_draft_probs_row_order_no_anchor_row(self):
        """draft_probs has gamma rows (no anchor row) for Gemma4 gated config."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        from sglang.srt.models.dspark import build_markov_head

        cfg = _TinyGemma4Config(markov_head_type="gated", markov_rank=8)
        gamma = cfg.block_size
        bs = 1
        vocab = cfg.vocab_size
        torch.manual_seed(70)

        head = build_markov_head(cfg).to(self.device)
        head.eval()

        base_logits = torch.randn(bs, gamma, vocab, device=self.device)
        first_prev = torch.randint(0, vocab, (bs,), device=self.device)
        block_hidden = torch.randn(bs, gamma, cfg.hidden_size, device=self.device)

        def greedy_sampler(logits, step_idx):
            return torch.argmax(logits, dim=-1)

        with torch.no_grad():
            sampled, corrected_logits = head.sample_block(
                base_logits,
                first_prev_tokens=first_prev,
                hidden_states=block_hidden,
                sampler=greedy_sampler,
            )

        self.assertEqual(sampled.shape, (bs, gamma))
        self.assertEqual(corrected_logits.shape, (bs, gamma, vocab))


class TestGemma4DSparkModelParityKEqV(TestGemma4DSparkModelParity):
    """Gemma4 DSpark parity with attention_k_eq_v=True (shared-K/V load branch)."""

    attention_k_eq_v = True


if __name__ == "__main__":
    unittest.main()
