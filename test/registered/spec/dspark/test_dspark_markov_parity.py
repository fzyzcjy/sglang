import os
import sys
import unittest

import torch

from sglang.srt.models.dspark import (
    GatedMarkovHead,
    RNNHead,
    VanillaMarkov,
    build_markov_head,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_ATOL = 1e-5
_RTOL = 1e-5

_REF_PKG = "test.srt.speculative._dspark_reference"
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


def _load_ref_markov_head():
    """Import the DeepSpec reference markov_head without the deepspec package."""
    import importlib

    _ensure_repo_test_package()
    ref = importlib.import_module(f"{_REF_PKG}.markov_head")
    return ref


def _greedy_sampler(logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def _sync_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy all named parameters from src into dst by name."""
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    with torch.no_grad():
        for name, param in src_params.items():
            if name in dst_params:
                dst_params[name].copy_(param)
            else:
                raise KeyError(f"Parameter {name!r} missing in dst model.")


class _HeadConfig:
    """Minimal config-like object for build_markov_head."""

    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        markov_head_type: str,
    ) -> None:
        self.vocab_size = vocab_size
        self.markov_rank = markov_rank
        self.hidden_size = hidden_size
        self.markov_head_type = markov_head_type


class TestVanillaMarkovParity(CustomTestCase):
    vocab_size = 128
    markov_rank = 16
    batch_size = 2
    proposal_len = 4

    @classmethod
    def setUpClass(cls) -> None:
        ref = _load_ref_markov_head()
        torch.manual_seed(0)
        cfg = _HeadConfig(
            vocab_size=cls.vocab_size,
            markov_rank=cls.markov_rank,
            hidden_size=64,
            markov_head_type="vanilla",
        )
        cls.sgl_head = build_markov_head(cfg)
        cls.ref_head = ref.VanillaMarkov(
            vocab_size=cls.vocab_size,
            markov_rank=cls.markov_rank,
        )
        cls.sgl_head.eval()
        cls.ref_head.eval()
        _sync_weights(cls.sgl_head, cls.ref_head)

    def _make_inputs(self):
        torch.manual_seed(42)
        token_ids = torch.randint(0, self.vocab_size, (self.batch_size,))
        base_logits = torch.randn(self.batch_size, self.proposal_len, self.vocab_size)
        first_prev = torch.randint(0, self.vocab_size, (self.batch_size,))
        return token_ids, base_logits, first_prev

    def test_compute_step_bias_exact_match(self):
        """VanillaMarkov.compute_step_bias must match reference exactly."""
        token_ids, _, _ = self._make_inputs()
        with torch.no_grad():
            sgl_bias = self.sgl_head.compute_step_bias(token_ids, hidden_states=None)
            ref_bias = self.ref_head.compute_step_bias(token_ids, hidden_states=None)
        torch.testing.assert_close(sgl_bias, ref_bias, atol=_ATOL, rtol=_RTOL)

    def test_apply_block_logits_exact_match(self):
        """VanillaMarkov.apply_block_logits must match reference exactly."""
        _, base_logits, _ = self._make_inputs()
        token_ids_block = torch.randint(
            0, self.vocab_size, (self.batch_size, self.proposal_len)
        )
        with torch.no_grad():
            sgl_out = self.sgl_head.apply_block_logits(
                base_logits, token_ids=token_ids_block, hidden_states=None
            )
            ref_out = self.ref_head.apply_block_logits(
                base_logits, token_ids=token_ids_block, hidden_states=None
            )
        torch.testing.assert_close(sgl_out, ref_out, atol=_ATOL, rtol=_RTOL)

    def test_sample_block_greedy_tokens_exact_match(self):
        """VanillaMarkov.sample_block greedy tokens must match reference."""
        token_ids, base_logits, first_prev = self._make_inputs()

        def ref_sampler_compat(logits, step_idx):
            return _greedy_sampler(logits, step_idx)

        with torch.no_grad():
            sgl_tokens, sgl_corr = self.sgl_head.sample_block(
                base_logits,
                first_prev_tokens=first_prev,
                hidden_states=None,
                sampler=_greedy_sampler,
            )
            ref_tokens, ref_corr = self.ref_head.sample_block_tokens(
                base_logits,
                first_prev_token_ids=first_prev,
                hidden_states=None,
                temperature=0.0,
            )
        self.assertTrue(
            torch.equal(sgl_tokens, ref_tokens),
            f"Greedy tokens mismatch: sgl={sgl_tokens} ref={ref_tokens}",
        )
        torch.testing.assert_close(sgl_corr, ref_corr, atol=_ATOL, rtol=_RTOL)

    def test_empty_proposal_len(self):
        """sample_block with proposal_len=0 returns empty tensors."""
        first_prev = torch.zeros(self.batch_size, dtype=torch.long)
        empty_logits = torch.randn(self.batch_size, 0, self.vocab_size)
        with torch.no_grad():
            tokens, corr = self.sgl_head.sample_block(
                empty_logits,
                first_prev_tokens=first_prev,
                hidden_states=None,
                sampler=_greedy_sampler,
            )
        self.assertEqual(tokens.shape, (self.batch_size, 0))

    def test_get_prev_embeddings_shape(self):
        """get_prev_embeddings returns [bs, markov_rank]."""
        token_ids = torch.zeros(self.batch_size, dtype=torch.long)
        emb = self.sgl_head.get_prev_embeddings(token_ids)
        self.assertEqual(emb.shape, (self.batch_size, self.markov_rank))

    def test_project_bias_shape(self):
        """project_bias output shape is [bs, vocab_size]."""
        latent = torch.randn(self.batch_size, self.markov_rank)
        bias = self.sgl_head.project_bias(latent)
        self.assertEqual(bias.shape, (self.batch_size, self.vocab_size))


class TestGatedMarkovParity(CustomTestCase):
    vocab_size = 128
    markov_rank = 16
    hidden_size = 32
    batch_size = 2
    proposal_len = 4

    @classmethod
    def setUpClass(cls) -> None:
        ref = _load_ref_markov_head()
        torch.manual_seed(1)
        cfg = _HeadConfig(
            vocab_size=cls.vocab_size,
            markov_rank=cls.markov_rank,
            hidden_size=cls.hidden_size,
            markov_head_type="gated",
        )
        cls.sgl_head = build_markov_head(cfg)
        cls.ref_head = ref.GatedMarkovHead(
            vocab_size=cls.vocab_size,
            markov_rank=cls.markov_rank,
            hidden_size=cls.hidden_size,
        )
        cls.sgl_head.eval()
        cls.ref_head.eval()
        _sync_weights(cls.sgl_head, cls.ref_head)

    def _make_inputs(self):
        torch.manual_seed(43)
        token_ids = torch.randint(0, self.vocab_size, (self.batch_size,))
        hidden = torch.randn(self.batch_size, self.hidden_size)
        base_logits = torch.randn(self.batch_size, self.proposal_len, self.vocab_size)
        first_prev = torch.randint(0, self.vocab_size, (self.batch_size,))
        return token_ids, hidden, base_logits, first_prev

    def test_compute_step_bias_exact_match(self):
        """GatedMarkovHead.compute_step_bias must match reference exactly."""
        token_ids, hidden, _, _ = self._make_inputs()
        with torch.no_grad():
            sgl_bias = self.sgl_head.compute_step_bias(token_ids, hidden_states=hidden)
            ref_bias = self.ref_head.compute_step_bias(token_ids, hidden_states=hidden)
        torch.testing.assert_close(sgl_bias, ref_bias, atol=_ATOL, rtol=_RTOL)

    def test_apply_block_logits_exact_match(self):
        """GatedMarkovHead.apply_block_logits must match reference exactly."""
        _, _, base_logits, _ = self._make_inputs()
        token_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.proposal_len)
        )
        hidden = torch.randn(self.batch_size, self.proposal_len, self.hidden_size)
        with torch.no_grad():
            sgl_out = self.sgl_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=hidden
            )
            ref_out = self.ref_head.apply_block_logits(
                base_logits, token_ids=token_ids, hidden_states=hidden
            )
        torch.testing.assert_close(sgl_out, ref_out, atol=_ATOL, rtol=_RTOL)

    def test_sample_block_greedy_tokens_exact_match(self):
        """GatedMarkovHead greedy sample_block tokens must match reference."""
        token_ids, _, base_logits, first_prev = self._make_inputs()
        block_hidden = torch.randn(self.batch_size, self.proposal_len, self.hidden_size)

        with torch.no_grad():
            sgl_tokens, sgl_corr = self.sgl_head.sample_block(
                base_logits,
                first_prev_tokens=first_prev,
                hidden_states=block_hidden,
                sampler=_greedy_sampler,
            )
            ref_tokens, ref_corr = self.ref_head.sample_block_tokens(
                base_logits,
                first_prev_token_ids=first_prev,
                hidden_states=block_hidden,
                temperature=0.0,
            )
        self.assertTrue(
            torch.equal(sgl_tokens, ref_tokens),
            f"Greedy tokens mismatch: sgl={sgl_tokens} ref={ref_tokens}",
        )
        torch.testing.assert_close(sgl_corr, ref_corr, atol=_ATOL, rtol=_RTOL)

    def test_requires_hidden_states(self):
        """GatedMarkovHead.compute_step_bias raises ValueError when hidden_states is None."""
        token_ids = torch.zeros(self.batch_size, dtype=torch.long)
        with self.assertRaises((ValueError, AssertionError)):
            self.sgl_head.compute_step_bias(token_ids, hidden_states=None)


class TestRNNHeadParity(CustomTestCase):
    vocab_size = 128
    markov_rank = 16
    hidden_size = 32
    batch_size = 2
    proposal_len = 4

    @classmethod
    def setUpClass(cls) -> None:
        ref = _load_ref_markov_head()
        torch.manual_seed(2)
        cfg = _HeadConfig(
            vocab_size=cls.vocab_size,
            markov_rank=cls.markov_rank,
            hidden_size=cls.hidden_size,
            markov_head_type="rnn",
        )
        cls.sgl_head = build_markov_head(cfg)
        cls.ref_head = ref.RNNHead(
            vocab_size=cls.vocab_size,
            markov_rank=cls.markov_rank,
            hidden_size=cls.hidden_size,
        )
        cls.sgl_head.eval()
        cls.ref_head.eval()
        _sync_weights(cls.sgl_head, cls.ref_head)

    def _make_inputs(self):
        torch.manual_seed(44)
        token_ids = torch.randint(0, self.vocab_size, (self.batch_size,))
        hidden = torch.randn(self.batch_size, self.hidden_size)
        base_logits = torch.randn(self.batch_size, self.proposal_len, self.vocab_size)
        first_prev = torch.randint(0, self.vocab_size, (self.batch_size,))
        block_hidden = torch.randn(self.batch_size, self.proposal_len, self.hidden_size)
        token_ids_block = torch.randint(
            0, self.vocab_size, (self.batch_size, self.proposal_len)
        )
        return token_ids, hidden, base_logits, first_prev, block_hidden, token_ids_block

    def test_compute_step_bias_exact_match(self):
        """RNNHead.compute_step_bias (zero state) must match reference exactly."""
        token_ids, hidden, _, _, _, _ = self._make_inputs()
        with torch.no_grad():
            sgl_bias = self.sgl_head.compute_step_bias(token_ids, hidden_states=hidden)
            ref_bias = self.ref_head.compute_step_bias(token_ids, hidden_states=hidden)
        torch.testing.assert_close(sgl_bias, ref_bias, atol=_ATOL, rtol=_RTOL)

    def test_apply_block_logits_exact_match(self):
        """RNNHead.apply_block_logits (unrolled RNN) must match reference exactly."""
        _, _, base_logits, _, block_hidden, token_ids_block = self._make_inputs()
        with torch.no_grad():
            sgl_out = self.sgl_head.apply_block_logits(
                base_logits,
                token_ids=token_ids_block,
                hidden_states=block_hidden,
            )
            ref_out = self.ref_head.apply_block_logits(
                base_logits,
                token_ids=token_ids_block,
                hidden_states=block_hidden,
            )
        torch.testing.assert_close(sgl_out, ref_out, atol=_ATOL, rtol=_RTOL)

    def test_sample_block_greedy_tokens_exact_match(self):
        """RNNHead greedy sample_block tokens must match reference serial sampling."""
        _, _, base_logits, first_prev, block_hidden, _ = self._make_inputs()

        with torch.no_grad():
            sgl_tokens, sgl_corr = self.sgl_head.sample_block(
                base_logits,
                first_prev_tokens=first_prev,
                hidden_states=block_hidden,
                sampler=_greedy_sampler,
            )
            ref_tokens, ref_corr = self.ref_head.sample_block_tokens(
                base_logits,
                first_prev_token_ids=first_prev,
                hidden_states=block_hidden,
                temperature=0.0,
            )
        self.assertTrue(
            torch.equal(sgl_tokens, ref_tokens),
            f"Greedy tokens mismatch: sgl={sgl_tokens} ref={ref_tokens}",
        )
        torch.testing.assert_close(sgl_corr, ref_corr, atol=_ATOL, rtol=_RTOL)

    def test_rnn_state_carries_across_steps(self):
        """apply_block_logits with 2 different token sequences must differ (state dependency)."""
        _, _, base_logits, _, block_hidden, _ = self._make_inputs()
        token_ids_a = torch.zeros(self.batch_size, self.proposal_len, dtype=torch.long)
        token_ids_b = (
            torch.ones(self.batch_size, self.proposal_len, dtype=torch.long) * 10
        )
        with torch.no_grad():
            out_a = self.sgl_head.apply_block_logits(
                base_logits, token_ids=token_ids_a, hidden_states=block_hidden
            )
            out_b = self.sgl_head.apply_block_logits(
                base_logits, token_ids=token_ids_b, hidden_states=block_hidden
            )
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "RNN outputs should differ for different prev-token sequences.",
        )

    def test_requires_hidden_states(self):
        """RNNHead.compute_step_bias raises when hidden_states is None."""
        token_ids = torch.zeros(self.batch_size, dtype=torch.long)
        with self.assertRaises((ValueError, AssertionError)):
            self.sgl_head.compute_step_bias(token_ids, hidden_states=None)


class TestBuildMarkovHead(CustomTestCase):
    def test_build_vanilla(self):
        """build_markov_head returns VanillaMarkov for vanilla head type."""
        cfg = _HeadConfig(
            vocab_size=100, markov_rank=8, hidden_size=32, markov_head_type="vanilla"
        )
        head = build_markov_head(cfg)
        self.assertIsInstance(head, VanillaMarkov)
        self.assertNotIsInstance(head, GatedMarkovHead)

    def test_build_gated(self):
        """build_markov_head returns GatedMarkovHead for gated head type."""
        cfg = _HeadConfig(
            vocab_size=100, markov_rank=8, hidden_size=32, markov_head_type="gated"
        )
        head = build_markov_head(cfg)
        self.assertIsInstance(head, GatedMarkovHead)

    def test_build_rnn(self):
        """build_markov_head returns RNNHead for rnn head type."""
        cfg = _HeadConfig(
            vocab_size=100, markov_rank=8, hidden_size=32, markov_head_type="rnn"
        )
        head = build_markov_head(cfg)
        self.assertIsInstance(head, RNNHead)

    def test_build_unknown_type_raises(self):
        """build_markov_head raises ValueError for unknown head type."""
        cfg = _HeadConfig(
            vocab_size=100, markov_rank=8, hidden_size=32, markov_head_type="unknown"
        )
        with self.assertRaises(ValueError):
            build_markov_head(cfg)

    def test_build_zero_rank_raises(self):
        """build_markov_head raises ValueError for markov_rank == 0."""
        cfg = _HeadConfig(
            vocab_size=100, markov_rank=0, hidden_size=32, markov_head_type="vanilla"
        )
        with self.assertRaises(ValueError):
            build_markov_head(cfg)


if __name__ == "__main__":
    unittest.main()
