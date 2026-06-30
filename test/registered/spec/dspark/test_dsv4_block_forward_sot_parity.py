import functools
import os
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()

# fp8 noise floor. The production sparse path quantizes the nope latent to fp8
# (DeepSeekV4 pool stores qk_nope FP8 + qk_rope BF16); the SoT oracle here keeps
# the latent in float. The tolerance is calibrated ABOVE that fp8 granularity so
# the parity assertion does not fail on quant noise, while the negative test's
# divergence (a causal index regression drops whole KV positions) is orders of
# magnitude larger than this floor.
_ATOL = 5e-2
_RTOL = 5e-2
# The negative (causal) test must diverge by FAR more than the fp8 floor, else
# the guardrail has no discriminating power.
_MIN_NEGATIVE_DIVERGENCE = 0.5

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
    """Decorator: skip when CUDA is unavailable (the sparse backend is GPU-only)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; dsv4 sparse backend is GPU-only.")
        return test_method(self, *args, **kwargs)

    return wrapper


def _contract_available() -> tuple[bool, str]:
    """Return (available, reason) for the dsv4 production-path block-forward contract.

    T1 drives the PRODUCTION dsv4 draft through the real ``DeepseekV4AttnBackend``
    with the NON-CAUSAL full-block index builder (``get_dspark_swa_page_indices``)
    and the standard ``forward(input_embeds, positions, forward_batch)`` returning a
    ``DSparkV4DraftOutput``. Those symbols are landed by the model+backend impl
    agents; until then this guardrail skips cleanly (it is the un-skipped successor
    to the old ``test_dsv4_worker_parity`` GPU stub).
    """
    try:
        from sglang.srt.layers.attention.deepseek_v4_backend import (  # noqa: F401
            DeepseekV4AttnBackend,
        )
    except ImportError as exc:  # pragma: no cover - import guard
        return False, f"dsv4 backend import failed: {exc}"

    from sglang.srt.layers.attention import deepseek_v4_backend as dsv4_backend
    from sglang.srt.models import deepseek_v4_dspark as dsv4_model

    if not hasattr(dsv4_backend.DeepseekV4AttnBackend, "get_dspark_swa_page_indices"):
        return (
            False,
            "non-causal builder DeepseekV4AttnBackend.get_dspark_swa_page_indices "
            "not yet implemented (model+backend agents own it)",
        )
    if not hasattr(dsv4_model, "DSparkV4DraftOutput"):
        return (
            False,
            "structured draft output DSparkV4DraftOutput not yet implemented "
            "(model agent owns it)",
        )
    model_cls = dsv4_model.DeepseekV4ForCausalLMDSpark
    if not callable(getattr(model_cls, "forward", None)):
        return False, "DeepseekV4ForCausalLMDSpark.forward(forward_batch) missing"
    return True, ""


class TestDsv4BlockForwardSoTParity(CustomTestCase):
    """T1 guardrail: production dsv4 draft block-forward must match the external SoT.

    THE most important guardrail in the dsv4 rewrite. The fatal silent-failure mode
    is "the implementer reused the CAUSAL ``get_swa_page_indices`` for the draft
    block". This test:

      1. drives the PRODUCTION path through the real ``DeepseekV4AttnBackend`` (so
         the actual non-causal metadata builder runs on the GPU - a pure-numerics
         CPU oracle would pass even with causal indices and is therefore useless);
      2. compares against the EXTERNAL SoT oracle (vendored from the DeepSeek-V4
         reference model.py, NOT the in-repo ``_sparse_attn`` scaffold, to avoid a
         shared-bug blindspot);
      3. CONSTRUCTS the inputs so non-causality is load-bearing - distinct KV at
         distinct draft-block positions, so a query row attending a LATER block
         position changes its output. A causal-triangle regression then makes the
         SoT parity diverge;
      4. includes a NEGATIVE test that flips the PRODUCTION builder to causal and
         asserts parity FAILS, proving discriminating power (not vacuously green).

    GPU-tier; skips cleanly until the model+backend agents land the contract.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._skip_reason = None
        if not _CUDA_AVAILABLE:
            cls._skip_reason = "CUDA not available; dsv4 sparse backend is GPU-only."
            return
        available, reason = _contract_available()
        if not available:
            cls._skip_reason = reason
            return
        _ensure_repo_test_package()

    def _skip_if_unready(self) -> None:
        if self._skip_reason:
            self.skipTest(self._skip_reason)

    def _build_oracle(self, harness):
        """Construct the SoT oracle sharing the production draft attention weights.

        ``harness`` is the production block-forward fixture (built by the model
        agent's test helper). It exposes the per-stage attention weights / freqs_cis
        / attn_sink and the projected-and-normed q/kv tensors so the oracle reproduces
        the SoT forward on identical numerics. The helper is intentionally late-bound:
        the model agent supplies ``build_dsv4_block_forward_harness`` next to the
        production model so the production tensors are authoritative.
        """
        _ensure_repo_test_package()
        from test.srt.speculative._dspark_reference.dsv4.sot_dspark_attention import (
            SoTDSparkAttentionOracle,
        )

        stage = harness.stage
        attn = stage.self_attn
        return SoTDSparkAttentionOracle(
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

    @_requires_cuda
    def test_non_causal_block_forward_matches_sot(self) -> None:
        """Production block-forward base_logits match the SoT within fp8 tolerance."""
        self._skip_if_unready()
        _ensure_repo_test_package()
        from test.srt.speculative._dspark_reference.dsv4.block_forward_harness import (
            HarnessUnavailable,
            build_dsv4_block_forward_harness,
        )

        # Construct a case that EXERCISES non-causality: distinct KV content at
        # distinct draft-block positions, so an early query row that (non-causally)
        # attends a later block position has a materially different output than a
        # causal-only row would. exercises_non_causality asserts the fixture really
        # places weight on the upper triangle (else this test is vacuous).
        try:
            harness = build_dsv4_block_forward_harness(
                seed=0, exercise_non_causality=True
            )
        except HarnessUnavailable as exc:
            self.skipTest(str(exc))
        self.assertTrue(
            harness.exercises_non_causality,
            "T1 fixture must place attention weight on later block positions, "
            "otherwise a causal regression would not change the output.",
        )

        prod = harness.run_production()
        oracle = self._build_oracle(harness)
        sot_base_logits = harness.run_sot(oracle)

        torch.testing.assert_close(
            prod.base_logits.float(),
            sot_base_logits.float(),
            atol=_ATOL,
            rtol=_RTOL,
        )

    @_requires_cuda
    def test_causal_index_regression_diverges_from_sot(self) -> None:
        """NEGATIVE: forcing the PRODUCTION builder causal must break SoT parity.

        Proves the guardrail has discriminating power. We patch the production
        non-causal builder to fall back to the causal ``get_swa_page_indices``
        triangle (the exact regression we fear) and assert the production base_logits
        now diverge from the SoT by far more than the fp8 floor. We flip the
        PRODUCTION builder, NOT the SoT oracle, so the SoT stays the fixed standard.
        """
        self._skip_if_unready()
        _ensure_repo_test_package()
        from test.srt.speculative._dspark_reference.dsv4.block_forward_harness import (
            HarnessUnavailable,
            build_dsv4_block_forward_harness,
        )

        try:
            harness = build_dsv4_block_forward_harness(
                seed=0, exercise_non_causality=True
            )
        except HarnessUnavailable as exc:
            self.skipTest(str(exc))
        oracle = self._build_oracle(harness)
        sot_base_logits = harness.run_sot(oracle)

        with harness.force_causal_indices():
            prod_causal = harness.run_production()

        max_abs_diff = (
            (prod_causal.base_logits.float() - sot_base_logits.float()).abs().max()
        )
        self.assertGreater(
            float(max_abs_diff),
            _MIN_NEGATIVE_DIVERGENCE,
            "Causal index regression did NOT diverge from the SoT - the parity "
            "guardrail has no discriminating power (it would pass even if the "
            "production builder were causal).",
        )


if __name__ == "__main__":
    unittest.main()
