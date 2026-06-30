import importlib
import os
import sys
import types
import unittest

import torch

from sglang.srt.models.dspark import (
    DSparkConfidenceHead,
    DSparkDraftMixin,
    build_confidence_head,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_ATOL = 1e-5
_RTOL = 1e-5

_REF_PKG = "test.manual._dspark_reference"
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
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


def _load_ref_accept_rate_predictor():
    """Import the DeepSpec reference AcceptRatePredictor without the deepspec package."""
    _ensure_repo_test_package()
    return importlib.import_module(f"{_REF_PKG}.common").AcceptRatePredictor


def _load_ref_markov_head():
    """Import the DeepSpec reference markov_head without the deepspec package."""
    _ensure_repo_test_package()
    return importlib.import_module(f"{_REF_PKG}.markov_head")


def _sync_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy all named parameters from src into dst by name."""
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    with torch.no_grad():
        for name, param in src_params.items():
            if name not in dst_params:
                raise KeyError(f"Parameter {name!r} missing in dst model.")
            dst_params[name].copy_(param)


class _HeadConfig:
    """Minimal config-like object for build_confidence_head / markov heads."""

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        markov_head_type: str = "vanilla",
        vocab_size: int = 128,
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
    ) -> None:
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.vocab_size = vocab_size
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov


class TestConfidenceHeadParity(CustomTestCase):
    hidden_size = 64
    markov_rank = 16
    batch_size = 2
    gamma = 4

    @classmethod
    def setUpClass(cls) -> None:
        ref_cls = _load_ref_accept_rate_predictor()
        torch.manual_seed(0)
        cls.sgl_head = DSparkConfidenceHead(
            hidden_size=cls.hidden_size,
            markov_rank=cls.markov_rank,
            with_markov=True,
        )
        cls.ref_head = ref_cls(input_dim=cls.hidden_size + cls.markov_rank)
        cls.sgl_head.eval()
        cls.ref_head.eval()
        _sync_weights(cls.sgl_head, cls.ref_head)

    def _make_inputs(self):
        torch.manual_seed(42)
        hidden = torch.randn(self.batch_size, self.gamma, self.hidden_size)
        markov_embed = torch.randn(self.batch_size, self.gamma, self.markov_rank)
        return hidden, markov_embed

    def test_with_markov_raw_logit_matches_reference(self):
        """DSparkConfidenceHead RAW logit must match AcceptRatePredictor(cat([h, m]))."""
        hidden, markov_embed = self._make_inputs()
        with torch.no_grad():
            sgl_out = self.sgl_head(hidden, markov_embed)
            ref_out = self.ref_head(torch.cat([hidden, markov_embed], dim=-1))
        self.assertEqual(sgl_out.shape, (self.batch_size, self.gamma))
        torch.testing.assert_close(sgl_out, ref_out, atol=_ATOL, rtol=_RTOL)

    def test_raw_logit_is_not_sigmoid(self):
        """The head emits a RAW logit (range unbounded), not a (0,1) probability."""
        hidden, markov_embed = self._make_inputs()
        original_bias = self.sgl_head.proj.bias.detach().clone()
        with torch.no_grad():
            self.sgl_head.proj.bias.fill_(50.0)
            out = self.sgl_head(hidden, markov_embed)
            self.sgl_head.proj.bias.copy_(original_bias)
        self.assertGreater(float(out.max()), 1.0)


class TestConfidenceHeadWithoutMarkov(CustomTestCase):
    hidden_size = 32
    markov_rank = 8
    batch_size = 2
    gamma = 3

    def test_without_markov_matches_hidden_only_reference(self):
        """with_markov=False feeds only the hidden state to the projection."""
        ref_cls = _load_ref_accept_rate_predictor()
        torch.manual_seed(1)
        sgl_head = DSparkConfidenceHead(
            hidden_size=self.hidden_size,
            markov_rank=self.markov_rank,
            with_markov=False,
        )
        ref_head = ref_cls(input_dim=self.hidden_size)
        sgl_head.eval()
        ref_head.eval()
        _sync_weights(sgl_head, ref_head)
        hidden = torch.randn(self.batch_size, self.gamma, self.hidden_size)
        with torch.no_grad():
            sgl_out = sgl_head(hidden, None)
            ref_out = ref_head(hidden)
        torch.testing.assert_close(sgl_out, ref_out, atol=_ATOL, rtol=_RTOL)

    def test_with_markov_requires_markov_embed(self):
        """with_markov=True raises when no markov_embed_stack is given."""
        head = DSparkConfidenceHead(
            hidden_size=self.hidden_size,
            markov_rank=self.markov_rank,
            with_markov=True,
        )
        hidden = torch.randn(self.batch_size, self.gamma, self.hidden_size)
        with self.assertRaises(ValueError):
            head(hidden, None)


class TestMarkovEmbedOffByOne(CustomTestCase):
    vocab_size = 128
    markov_rank = 16
    batch_size = 2
    gamma = 4

    def test_markov_embed_stack_uses_prev_token_offset(self):
        """markov_embed[:, i] = markov_w1(prev_seq[:, i]); prev_seq=[anchor, s_0..s_{g-2}]."""
        ref = _load_ref_markov_head()
        torch.manual_seed(7)
        head = ref.VanillaMarkov(
            vocab_size=self.vocab_size, markov_rank=self.markov_rank
        )
        head.eval()
        anchor = torch.randint(0, self.vocab_size, (self.batch_size,))
        draft_tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.gamma))

        prev_seq = torch.cat(
            [anchor.view(-1, 1), draft_tokens[:, : self.gamma - 1]], dim=1
        )
        with torch.no_grad():
            stack = head.get_prev_embeddings(prev_seq)

        self.assertEqual(stack.shape, (self.batch_size, self.gamma, self.markov_rank))
        with torch.no_grad():
            expected_step0 = head.get_prev_embeddings(anchor)
            expected_step1 = head.get_prev_embeddings(draft_tokens[:, 0])
            expected_last = head.get_prev_embeddings(draft_tokens[:, self.gamma - 2])
        torch.testing.assert_close(stack[:, 0], expected_step0, atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(stack[:, 1], expected_step1, atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(
            stack[:, self.gamma - 1], expected_last, atol=_ATOL, rtol=_RTOL
        )


class TestBuildConfidenceHead(CustomTestCase):
    def test_disabled_returns_none(self):
        """build_confidence_head returns None when enable_confidence_head is False."""
        cfg = _HeadConfig(hidden_size=64, markov_rank=16, enable_confidence_head=False)
        self.assertIsNone(build_confidence_head(cfg))

    def test_enabled_with_markov_input_dim(self):
        """Enabled with_markov head has proj in_features == hidden_size + markov_rank."""
        cfg = _HeadConfig(
            hidden_size=64,
            markov_rank=16,
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
        head = build_confidence_head(cfg)
        self.assertIsInstance(head, DSparkConfidenceHead)
        self.assertTrue(head.with_markov)
        self.assertEqual(head.proj.in_features, 64 + 16)

    def test_enabled_without_markov_input_dim(self):
        """Enabled head without markov has proj in_features == hidden_size."""
        cfg = _HeadConfig(
            hidden_size=64,
            markov_rank=16,
            enable_confidence_head=True,
            confidence_head_with_markov=False,
        )
        head = build_confidence_head(cfg)
        self.assertEqual(head.proj.in_features, 64)

    def test_with_markov_requires_positive_rank(self):
        """confidence_head_with_markov with markov_rank == 0 raises."""
        cfg = _HeadConfig(
            hidden_size=64,
            markov_rank=0,
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
        with self.assertRaises(ValueError):
            build_confidence_head(cfg)


class TestMissingConfidenceWeightsRaises(CustomTestCase):
    hidden_size = 32
    markov_rank = 8

    def test_missing_confidence_weights_raises(self) -> None:
        """Dense draft with enabled confidence head raises ValueError when weights are absent."""
        head = DSparkConfidenceHead(
            hidden_size=self.hidden_size,
            markov_rank=self.markov_rank,
            with_markov=True,
        )
        stub = types.SimpleNamespace(confidence_head=head)
        params_dict = {f"confidence_head.{k}": v for k, v in head.named_parameters()}
        with self.assertRaises(ValueError):
            DSparkDraftMixin._load_confidence_weights(
                stub,
                confidence_weights=[],
                params_dict=params_dict,
            )


if __name__ == "__main__":
    unittest.main()
