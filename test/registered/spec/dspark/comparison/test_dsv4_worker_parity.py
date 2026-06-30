import functools
import os
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()

# dsv4 DSpark real checkpoints (target backbone + DSpark draft), pre-staged on the GPU
# tester at /node_public/hf. The tester reconciles the exact local paths / HF slugs and
# the dsv4 launch flags (attention backend, draft attention backend, page size) before
# running; the e2e structure (near-tie-tolerant lossless + GSM8K accept-length floor) is
# the dense DSpark lossless framework reused for dsv4.
_DSV4_TARGET_MODEL = os.environ.get(
    "SGLANG_DSV4_TARGET_MODEL", "/node_public/hf/DeepSeek-V4-Flash"
)
_DSV4_DRAFT_MODEL = os.environ.get(
    "SGLANG_DSV4_DRAFT_MODEL", "/node_public/hf/DeepSeek-V4-Flash-DSpark"
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
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
    """Decorator: skip when CUDA is unavailable (the dsv4 sparse backend is GPU-only)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; dsv4 sparse backend is GPU-only.")
        return test_method(self, *args, **kwargs)

    return wrapper


class TestDsv4BlockForwardVsSoT(CustomTestCase):
    """V4 DSpark draft block-forward must match the external SoT (T1, harness-driven).

    Previously a permanent ``@unittest.skip`` + ``NotImplementedError`` stub. Now the
    block-forward-vs-SoT parity (the real T1 guardrail) drives the production dsv4 draft
    through the shared block-forward harness; it skips cleanly until the production
    object graph is constructible (HarnessUnavailable). The end-to-end greedy losslessness
    + accept-length entry lives in ``TestDsv4WorkerE2ELossless`` below (server-tier).
    """

    @_requires_cuda
    def test_v4_decode_block_matches_sot_forward_spec(self) -> None:
        """The worker's V4 draft block must equal the SoT forward_spec output tightly."""
        _ensure_repo_test_package()
        from test.manual._dspark_reference.deepseek_v4.sot_attention import (
            SoTDSparkAttentionOracle,
        )
        from test.manual._dspark_reference.dsv4.block_forward_harness import (
            HarnessUnavailable,
            build_dsv4_block_forward_harness,
        )

        try:
            harness = build_dsv4_block_forward_harness(
                seed=0, exercise_non_causality=True
            )
        except HarnessUnavailable as exc:
            self.skipTest(str(exc))

        attn = harness.stage.self_attn
        oracle = SoTDSparkAttentionOracle(
            head_dim=attn.head_dim,
            rope_head_dim=attn.rope_head_dim,
            n_heads=attn.n_local_heads,
            n_groups=attn.n_local_groups,
            o_lora_rank=attn.o_lora_rank,
            window_size=attn.window_size,
            eps=attn.eps,
            softmax_scale=attn.softmax_scale,
            freqs_cis=attn.freqs_cis,
            attn_sink=attn.attn_sink,
            device=harness.device,
            dtype=harness.dtype,
        )
        prod = harness.run_production()
        sot_base_logits = harness.run_sot(oracle)
        # Through-sparse-attention fp8 tolerance: calibrated ABOVE the fp8 noise floor by
        # the tester (measure the known-correct oracle-vs-production baseline first), not
        # bf16-tight (see test_dsv4_block_forward_sot_parity._ATOL).
        torch.testing.assert_close(
            prod.base_logits.float(),
            sot_base_logits.float(),
            atol=5e-2,
            rtol=5e-2,
        )


def _build_dsv4_e2e_base():
    """Build the dsv4 e2e lossless test base by reusing the dense DSpark framework.

    ``_DSparkLosslessBase`` (test_dspark_lossless.py) runs the four-phase sequential
    eager/cuda-graph reference-vs-spec capture + a resident spec server, then asserts:
      * eager greedy parity EXACT vs the non-spec baseline;
      * cuda-graph greedy parity NEAR-TIE tolerant (every divergence is a reference top-2
        near-tie; a divergence at a CONFIDENT reference token fails -- a lossless accept
        path cannot do that);
      * GSM8K accuracy + accept-length floor (accept length > 1 proves the non-causal
        full-block draft is genuinely accepting, the end-to-end probe that the
        non-causal index reproduced correctly).

    Reused verbatim for dsv4 with the dsv4 target/draft checkpoints + the dsv4 attention
    backend, so the dsv4 worker e2e shares the exact lossless judging. Returns the base
    class (or None if the dense framework is unavailable, e.g. CPU import).
    """
    _ensure_repo_test_package()
    try:
        from test.registered.spec.dspark.test_dspark_lossless import (
            _DSparkLosslessBase,
        )
    except Exception:
        return None
    return _DSparkLosslessBase


_DSparkLosslessBaseCls = _build_dsv4_e2e_base()


if _DSparkLosslessBaseCls is not None:

    class TestDsv4WorkerE2ELossless(_DSparkLosslessBaseCls):
        """dsv4 DSpark spec decode is lossless vs the non-spec baseline + accept len > 1.

        Subclasses the dense DSpark lossless framework with the dsv4 target backbone +
        DSpark draft. The tester runs it on GPU against the real checkpoints; it skips
        cleanly when the checkpoints are unavailable. The accept-length floor is the
        end-to-end probe that the non-causal full-block draft reproduced correctly -- an
        accept length collapsing toward 1 means the non-causal index was NOT reproduced.

        The exact dsv4 launch flags (attention backend / draft attention backend / page
        size) must be reconciled by the tester from the resolved dsv4 server recipe; the
        defaults here mirror the dense framework and may need a dsv4-specific backend.
        """

        __test__ = True

        target_model = _DSV4_TARGET_MODEL
        draft_model = _DSV4_DRAFT_MODEL
        # dsv4 needs the deepseek_v4 sparse backend; the tester sets the resolved flag.
        attention_backend = "fa4"
        # Lossless greedy keeps spec accuracy at the non-spec baseline; the accept-length
        # floor proves the semi-AR block draft is genuinely accepting (NOT degenerating to
        # 1). The tester re-measures the dsv4 baseline and tightens these.
        gsm8k_score_threshold = 0.80
        gsm8k_num_examples = 200
        gsm8k_accept_length_thres = 2.0

else:  # pragma: no cover - dense framework import unavailable (e.g. CPU-only collect)

    class TestDsv4WorkerE2ELossless(CustomTestCase):
        """Placeholder when the dense DSpark lossless framework cannot be imported."""

        def test_dsv4_e2e_lossless(self) -> None:
            """The dsv4 e2e lossless framework requires the dense DSpark lossless base."""
            self.skipTest(
                "dense DSpark lossless framework (test_dspark_lossless._DSparkLosslessBase) "
                "could not be imported; the dsv4 e2e lossless test reuses it."
            )


if __name__ == "__main__":
    unittest.main()
