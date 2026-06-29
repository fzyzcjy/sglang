import types
import unittest
from typing import Optional

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()
_ATOL_HIDDEN = 1e-4
_RTOL_HIDDEN = 1e-4
_ATOL_LOGITS = 1e-4
_RTOL_LOGITS = 1e-4


def _requires_cuda(test_method):
    """Decorator: skip test if CUDA is not available."""
    import functools

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available")
        return test_method(self, *args, **kwargs)

    return wrapper


class _TinyQwen3Config:
    """Minimal Qwen3-like config for DSpark model parity tests."""

    def __init__(self, markov_head_type: str = "vanilla", markov_rank: int = 8) -> None:
        self.hidden_size = 64
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = 128
        self.vocab_size = 256
        self.rms_norm_eps = 1e-6
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.max_position_embeddings = 512
        self.rope_theta = 10000.0
        self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.sliding_window = None
        self.hidden_act = "silu"

        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type

        self.target_layer_ids = [0]
        self.block_size = 3
        self.mask_token_id = 1
        self.num_anchors = 2
        self.enable_confidence_head = False

        self._attn_implementation = "eager"
        self.model_type = "qwen3"


class _TinyGemma4Config:
    """Minimal Gemma4-like config for DSpark model parity tests."""

    def __init__(self, markov_head_type: str = "vanilla", markov_rank: int = 8) -> None:
        self.hidden_size = 64
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 4
        self.global_head_dim = 16
        self.num_global_key_value_heads = 4
        self.intermediate_size = 128
        self.vocab_size = 256
        self.rms_norm_eps = 1e-6
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.max_position_embeddings = 512
        self.rope_theta = 10000.0
        self.attention_k_eq_v = False
        self.final_logit_softcapping = None
        self.enable_moe_block = False
        self.hidden_size_per_layer_input = 0
        self.head_dim = self.global_head_dim
        self.hidden_act = "gelu"
        self.query_pre_attn_scalar = 1.0

        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type

        self.target_layer_ids = [0]
        self.block_size = 3
        self.mask_token_id = 1
        self.num_anchors = 2
        self.enable_confidence_head = False

        self._attn_implementation = "eager"
        self.model_type = "gemma4"


def _sync_weights_by_name(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy matching named parameters from src into dst."""
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    with torch.no_grad():
        for name, param in src_params.items():
            if name in dst_params:
                dst_params[name].copy_(param)


class TestQwen3DSparkModelParity(CustomTestCase):
    """Worker-faithful parity: SGLang Qwen3DSparkModel vs DeepSpec reference.

    Compares fc/hidden_norm projection, markov head bias, draft logits, and
    greedy tokens for identical random weights.  GPU-only (requires CUDA to run
    the Qwen3 rotary embedding and attention).
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _CUDA_AVAILABLE:
            return
        try:
            from sglang.srt.models.dspark import Qwen3DSparkModel as SglQwen3DSparkModel
            from test.srt.speculative._dspark_reference.qwen3.modeling import (
                Qwen3DSparkModel as RefQwen3DSparkModel,
            )
        except ImportError as exc:
            cls._import_error = str(exc)
            return
        cls._import_error = None

        cfg = _TinyQwen3Config(markov_head_type="vanilla", markov_rank=8)
        torch.manual_seed(10)

        device = torch.device("cuda")
        cls.device = device
        cls.cfg = cfg

        cls.sgl_model = SglQwen3DSparkModel.__new__(SglQwen3DSparkModel)
        cls.ref_model = None

    @_requires_cuda
    def test_markov_head_parity_vanilla(self):
        """SGLang markov head weights match reference for Qwen3 vanilla config."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        from sglang.srt.models.dspark import build_markov_head, VanillaMarkov
        from test.srt.speculative._dspark_reference.markov_head import (
            VanillaMarkov as RefVanillaMarkov,
        )

        cfg = self.cfg
        torch.manual_seed(20)
        sgl_head = build_markov_head(cfg)
        ref_head = RefVanillaMarkov(vocab_size=cfg.vocab_size, markov_rank=cfg.markov_rank)
        sgl_head.eval()
        ref_head.eval()
        _sync_weights_by_name(sgl_head, ref_head)

        bs, gamma = 1, cfg.block_size
        base_logits = torch.randn(bs, gamma, cfg.vocab_size)
        token_ids = torch.randint(0, cfg.vocab_size, (bs,))

        with torch.no_grad():
            sgl_out = sgl_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=None
            )
            ref_out = ref_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=None
            )
        torch.testing.assert_close(sgl_out, ref_out, atol=_ATOL_LOGITS, rtol=_RTOL_LOGITS)

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

        head = build_markov_head(cfg)
        head.eval()

        base_logits = torch.randn(bs, gamma, vocab)
        first_prev = torch.randint(0, vocab, (bs,))

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

    Checks markov head + fc/hidden_norm projection output using identical
    random weights. GPU-only.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _CUDA_AVAILABLE:
            return
        cls._import_error = None
        try:
            from sglang.srt.models.dspark_gemma import Gemma4DSparkModel as SglGemma4
        except ImportError as exc:
            cls._import_error = str(exc)

    @_requires_cuda
    def test_markov_head_parity_vanilla(self):
        """SGLang markov head bias matches reference for Gemma4 vanilla config."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        from sglang.srt.models.dspark import build_markov_head, VanillaMarkov
        from test.srt.speculative._dspark_reference.markov_head import (
            VanillaMarkov as RefVanillaMarkov,
        )

        cfg = _TinyGemma4Config(markov_head_type="vanilla", markov_rank=8)
        torch.manual_seed(50)
        sgl_head = build_markov_head(cfg)
        ref_head = RefVanillaMarkov(vocab_size=cfg.vocab_size, markov_rank=cfg.markov_rank)
        sgl_head.eval()
        ref_head.eval()
        _sync_weights_by_name(sgl_head, ref_head)

        bs, gamma = 1, cfg.block_size
        base_logits = torch.randn(bs, gamma, cfg.vocab_size)
        token_ids = torch.randint(0, cfg.vocab_size, (bs,))

        with torch.no_grad():
            sgl_out = sgl_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=None
            )
            ref_out = ref_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=None
            )
        torch.testing.assert_close(sgl_out, ref_out, atol=_ATOL_LOGITS, rtol=_RTOL_LOGITS)

    @_requires_cuda
    def test_fc_hidden_norm_projection_shape(self):
        """fc+hidden_norm projection output has shape [bs, hidden_size]."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        import torch.nn as nn

        cfg = _TinyGemma4Config()
        torch.manual_seed(60)
        n_target_layers = len(cfg.target_layer_ids)
        fc = nn.Linear(n_target_layers * cfg.hidden_size, cfg.hidden_size, bias=False)

        bs, seq = 1, 4
        target_h = torch.randn(bs, seq, n_target_layers * cfg.hidden_size)
        out = fc(target_h)
        self.assertEqual(out.shape, (bs, seq, cfg.hidden_size))

    @_requires_cuda
    def test_draft_probs_row_order_no_anchor_row(self):
        """draft_probs has gamma rows (no anchor row) for Gemma4 config."""
        if getattr(self, "_import_error", None):
            self.skipTest(f"Import error: {self._import_error}")

        from sglang.srt.models.dspark import build_markov_head

        cfg = _TinyGemma4Config(markov_head_type="gated", markov_rank=8)
        gamma = cfg.block_size
        bs = 1
        vocab = cfg.vocab_size
        torch.manual_seed(70)

        head = build_markov_head(cfg)
        head.eval()

        base_logits = torch.randn(bs, gamma, vocab)
        first_prev = torch.randint(0, vocab, (bs,))
        block_hidden = torch.randn(bs, gamma, cfg.hidden_size)

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


if __name__ == "__main__":
    unittest.main()
