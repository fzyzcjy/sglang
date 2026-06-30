import os
import types
import unittest

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models import dspark as dspark_base
from sglang.srt.models.deepseek_v4 import hc_head_torch, make_hc_head_params
from sglang.srt.models.deepseek_v4_dspark import (
    DeepseekV4ForCausalLMDSpark,
    DSparkV4MarkovHead,
    build_dspark_v4_confidence_head,
)
from sglang.srt.models.dspark import DSparkConfidenceHead
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_HIDDEN = 4096
_HC_MULT = 4
_MARKOV_RANK = 256
_VOCAB = 129280
_GAMMA = 5
_NORM_EPS = 1e-6
_HC_EPS = 1e-3


class _DsparkConfig:
    """Minimal config exposing the fields build_dspark_v4_confidence_head reads."""

    def __init__(
        self,
        *,
        enable_confidence_head: bool,
        confidence_head_with_markov: bool = True,
    ) -> None:
        self.hidden_size = _HIDDEN
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov


def _ensure_dist_initialized() -> None:
    """Set up a single-rank gloo distributed environment plus TP/PP/EP groups.

    DSparkV4MarkovHead.markov_w1 is a VocabParallelEmbedding whose forward calls
    get_tp_group(); even at tp=1 that asserts the model-parallel group exists, so
    the CPU stub must initialize it before any markov-head forward runs.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
        )

    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


def _make_last_stage() -> types.SimpleNamespace:
    """A stand-in for stages[-1]: real norm + real hc_head params (initialized)."""
    norm = RMSNorm(_HIDDEN, eps=_NORM_EPS)
    # CPU test runs on a CUDA box, where RMSNorm dispatches to the CUDA-only
    # sgl_kernel; force the platform-agnostic native path so it runs on CPU.
    norm._forward_method = norm.forward_native
    hc_head_fn, hc_head_base, hc_head_scale = make_hc_head_params(_HC_MULT, _HIDDEN)
    with torch.no_grad():
        hc_head_fn.normal_()
        hc_head_base.normal_()
        hc_head_scale.fill_(1.0)
    return types.SimpleNamespace(
        norm=norm,
        hc_head_fn=hc_head_fn,
        hc_head_scale=hc_head_scale,
        hc_head_base=hc_head_base,
    )


def _make_lm_head() -> types.SimpleNamespace:
    weight = torch.randn(_VOCAB, _HIDDEN)
    return types.SimpleNamespace(weight=weight, org_vocab_size=_VOCAB)


def _make_model_stub(
    *, with_confidence: bool, with_markov: bool = True
) -> DeepseekV4ForCausalLMDSpark:
    """Build a DeepseekV4ForCausalLMDSpark stub without the heavy backbone __init__.

    Attaches only the fields the tap-point / base-logits / confidence methods touch so
    the real (unbound) methods run on CPU.
    """
    _ensure_dist_initialized()
    model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
    torch.nn.Module.__init__(model)
    model.gamma = _GAMMA
    model.block_size = _GAMMA
    model.hc_mult = _HC_MULT
    model.norm_eps = _NORM_EPS
    model.hc_eps = _HC_EPS
    model.stages = [_make_last_stage()]
    model.lm_head = _make_lm_head()
    model.markov_head = DSparkV4MarkovHead(vocab_size=_VOCAB, markov_rank=_MARKOV_RANK)
    model.confidence_head = (
        DSparkConfidenceHead(
            hidden_size=_HIDDEN, markov_rank=_MARKOV_RANK, with_markov=with_markov
        )
        if with_confidence
        else None
    )
    with torch.no_grad():
        for param in model.markov_head.parameters():
            param.normal_()
        if model.confidence_head is not None:
            for param in model.confidence_head.parameters():
                param.normal_()
    model._last_confidence = None
    return model


class TestDsv4ConfidenceTapPoint(CustomTestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        # All-gather is a no-op at tp=1; patch it to identity so the test needs no
        # process group while still exercising the slice / call path.
        self._captured_gather_dim = None

        def _fake_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
            self._captured_gather_dim = dim
            return tensor

        self._orig_gather = dspark_base.tensor_model_parallel_all_gather
        dspark_base.tensor_model_parallel_all_gather = _fake_all_gather

    def tearDown(self) -> None:
        dspark_base.tensor_model_parallel_all_gather = self._orig_gather

    def test_confidence_consumes_x_post_hc_not_norm_not_pre_hc_head(self) -> None:
        """V4 confidence must see post-hc_head PRE-norm x, not norm(x) and not pre_hc_head."""
        model = _make_model_stub(with_confidence=True)
        bsz = 2
        x = torch.randn(bsz, _GAMMA, _HC_MULT, _HIDDEN)
        pre_hc_head = x.flatten(2)
        expected_x_post_hc = hc_head_torch(
            x,
            model.stages[-1].hc_head_fn,
            model.stages[-1].hc_head_scale,
            model.stages[-1].hc_head_base,
            norm_eps=_NORM_EPS,
            hc_eps=_HC_EPS,
        )
        expected_norm = model.stages[-1].norm(expected_x_post_hc)

        captured = {}
        orig_forward = model.confidence_head.forward

        def _spy(hidden_states, markov_embed_stack=None):
            captured["hidden"] = hidden_states
            return orig_forward(hidden_states, markov_embed_stack)

        model.confidence_head.forward = _spy

        anchor_tokens = torch.randint(0, _VOCAB, (bsz,))
        sampled_tokens = torch.randint(0, _VOCAB, (bsz, _GAMMA))
        model.compute_base_logits(x)
        model.compute_confidence(
            anchor_tokens=anchor_tokens, sampled_tokens=sampled_tokens
        )

        fed = captured["hidden"]
        self.assertTrue(torch.allclose(fed, expected_x_post_hc, atol=1e-5))
        self.assertFalse(torch.allclose(fed, expected_norm, atol=1e-4))
        self.assertFalse(
            fed.shape == pre_hc_head.shape
            and torch.allclose(fed, pre_hc_head, atol=1e-4)
        )

    def test_base_logits_tap_matches_norm_of_x_post_hc(self) -> None:
        """LM-head logits apply norm to the same x_post_hc the confidence head sees raw."""
        model = _make_model_stub(with_confidence=False)
        bsz = 2
        x = torch.randn(bsz, _GAMMA, _HC_MULT, _HIDDEN)
        x_post_hc = model.collapse_hc_head(x)
        logits = model._logits_from_x_post_hc(x_post_hc)

        normed = model.stages[-1].norm(x_post_hc)
        expected = torch.matmul(normed.float(), model.lm_head.weight.float().T)
        self.assertEqual(logits.shape, (bsz, _GAMMA, _VOCAB))
        self.assertTrue(torch.allclose(logits, expected, atol=1e-5))
        self.assertEqual(self._captured_gather_dim, -1)

    def test_base_logits_slices_to_org_vocab_size(self) -> None:
        """TP all-gather output is sliced back to org_vocab_size (drops vocab padding)."""
        model = _make_model_stub(with_confidence=False)
        padded_vocab = _VOCAB + 8
        model.lm_head = types.SimpleNamespace(
            weight=torch.randn(padded_vocab, _HIDDEN), org_vocab_size=_VOCAB
        )
        x = torch.randn(1, _GAMMA, _HC_MULT, _HIDDEN)
        logits = model._logits_from_x_post_hc(model.collapse_hc_head(x))
        self.assertEqual(logits.shape[-1], _VOCAB)

    def test_confidence_is_none_when_head_disabled(self) -> None:
        """Disabled confidence head leaves _last_confidence None (a+b unaffected)."""
        model = _make_model_stub(with_confidence=False)
        x = torch.randn(2, _GAMMA, _HC_MULT, _HIDDEN)
        anchor_tokens = torch.randint(0, _VOCAB, (2,))
        sampled_tokens = torch.randint(0, _VOCAB, (2, _GAMMA))
        model.compute_base_logits(x)
        self.assertIsNone(
            model.compute_confidence(
                anchor_tokens=anchor_tokens, sampled_tokens=sampled_tokens
            )
        )
        self.assertIsNone(model._last_confidence)
        self.assertIsNone(model.last_confidence())

    def test_confidence_shape_and_range_when_enabled(self) -> None:
        """Enabled confidence head stashes [bs, gamma] sigmoid values in [0, 1]."""
        model = _make_model_stub(with_confidence=True)
        bsz = 2
        x = torch.randn(bsz, _GAMMA, _HC_MULT, _HIDDEN)
        anchor_tokens = torch.randint(0, _VOCAB, (bsz,))
        sampled_tokens = torch.randint(0, _VOCAB, (bsz, _GAMMA))
        model.compute_base_logits(x)
        model.compute_confidence(
            anchor_tokens=anchor_tokens, sampled_tokens=sampled_tokens
        )
        confidence = model.last_confidence()
        self.assertIsNotNone(confidence)
        self.assertEqual(confidence.shape, (bsz, _GAMMA))
        self.assertTrue(bool(((confidence >= 0) & (confidence <= 1)).all()))

    def test_markov_embed_stack_off_by_one(self) -> None:
        """markov_embed prev seq is [anchor, s_0, ..., s_{gamma-2}] (off-by-one)."""
        model = _make_model_stub(with_confidence=True)
        bsz = 2
        anchor = torch.randint(0, _VOCAB, (bsz,))
        sampled = torch.randint(0, _VOCAB, (bsz, _GAMMA))
        expected_prev = torch.cat([anchor.view(-1, 1), sampled[:, : _GAMMA - 1]], dim=1)
        expected_stack = model.markov_head.get_prev_embeddings(expected_prev)

        captured = {}
        orig_forward = model.confidence_head.forward

        def _spy(hidden_states, markov_embed_stack=None):
            captured["stack"] = markov_embed_stack
            return orig_forward(hidden_states, markov_embed_stack)

        model.confidence_head.forward = _spy
        x_post_hc = torch.randn(bsz, _GAMMA, _HIDDEN)
        model._x_post_hc = x_post_hc.reshape(bsz * _GAMMA, _HIDDEN)
        model.compute_confidence(anchor_tokens=anchor, sampled_tokens=sampled)
        self.assertTrue(torch.allclose(captured["stack"], expected_stack, atol=1e-6))


class TestDsv4ConfidenceBuild(CustomTestCase):
    def test_build_ignores_disable_flag(self) -> None:
        """The head is built even when enable_confidence_head is False (flag not read)."""
        config = _DsparkConfig(enable_confidence_head=False)
        head = build_dspark_v4_confidence_head(config=config, markov_rank=_MARKOV_RANK)
        self.assertIsInstance(head, DSparkConfidenceHead)

    def test_build_warns_when_field_absent(self) -> None:
        """A config lacking enable_confidence_head warns once, then builds the head."""
        config = types.SimpleNamespace(
            hidden_size=_HIDDEN, confidence_head_with_markov=True
        )
        with self.assertLogs(
            "sglang.srt.models.deepseek_v4_dspark", level="WARNING"
        ) as captured:
            head = build_dspark_v4_confidence_head(
                config=config, markov_rank=_MARKOV_RANK
            )
        self.assertIsInstance(head, DSparkConfidenceHead)
        self.assertTrue(
            any("enable_confidence_head" in message for message in captured.output)
        )

    def test_build_returns_head_when_enabled(self) -> None:
        """Enabled config builds a with_markov head with input dim hidden + markov_rank."""
        config = _DsparkConfig(enable_confidence_head=True)
        head = build_dspark_v4_confidence_head(config=config, markov_rank=_MARKOV_RANK)
        self.assertIsInstance(head, DSparkConfidenceHead)
        self.assertTrue(head.with_markov)
        self.assertEqual(head.proj.in_features, _HIDDEN + _MARKOV_RANK)

    def test_build_with_markov_requires_positive_rank(self) -> None:
        """with_markov head with markov_rank <= 0 raises."""
        config = _DsparkConfig(
            enable_confidence_head=True, confidence_head_with_markov=True
        )
        with self.assertRaises(ValueError):
            build_dspark_v4_confidence_head(config=config, markov_rank=0)


class TestDsv4ConfidenceWeightRemap(CustomTestCase):
    def _remap(self, model: DeepseekV4ForCausalLMDSpark, name: str):
        return model._remap_dspark_weight_name(name)

    def test_confidence_weight_loaded_when_head_enabled(self) -> None:
        """confidence_head.* maps into the draft tree when the head is enabled."""
        model = _make_model_stub(with_confidence=True)
        self.assertEqual(
            self._remap(model, "mtp.0.confidence_head.proj.weight"),
            "confidence_head.proj.weight",
        )
        self.assertEqual(
            self._remap(model, "mtp.0.confidence_head.proj.bias"),
            "confidence_head.proj.bias",
        )

    def test_confidence_weight_dropped_when_head_disabled(self) -> None:
        """confidence_head.* is dropped (None) when the head is disabled."""
        model = _make_model_stub(with_confidence=False)
        self.assertIsNone(self._remap(model, "mtp.0.confidence_head.proj.weight"))

    def test_markov_and_lm_head_remap_unchanged(self) -> None:
        """Markov maps to markov_head.*; shared embed/lm_head still dropped."""
        model = _make_model_stub(with_confidence=True)
        self.assertEqual(
            self._remap(model, "mtp.0.markov_head.markov_w2.weight"),
            "markov_head.markov_w2.weight",
        )
        self.assertIsNone(self._remap(model, "lm_head.weight"))
        self.assertIsNone(self._remap(model, "embed_tokens.weight"))

    def test_missing_confidence_weights_raises(self) -> None:
        """Enabled head absent from checkpoint raises ValueError instead of degrading."""
        model = _make_model_stub(with_confidence=True)
        params_dict = dict(model.confidence_head.named_parameters())
        params_dict = {f"confidence_head.{k}": v for k, v in params_dict.items()}
        with self.assertRaises(ValueError):
            model._assert_confidence_head_loaded(
                params_dict=params_dict, loaded_params=set()
            )


if __name__ == "__main__":
    unittest.main()
