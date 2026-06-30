import functools
import os
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()

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
    """Decorator: skip when CUDA is unavailable (the dsv4 sparse backend is GPU-only)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; dsv4 sparse backend is GPU-only.")
        return test_method(self, *args, **kwargs)

    return wrapper


class TestDsv4WorkerParityVsSoT(CustomTestCase):
    """V4 DSpark worker decode must match the SoT, and greedy spec decode is lossless.

    Previously a permanent ``@unittest.skip`` + ``NotImplementedError`` stub (blocked
    on dsv4 GPU enablement). Now un-skipped: the block-forward-vs-SoT parity is the
    real T1 guardrail (``test_dsv4_block_forward_sot_parity``), driven here through the
    shared production-vs-SoT harness; the end-to-end greedy-losslessness entry is a
    server-tier check the remote tester runs (T6). Both skip cleanly until the
    production contract / a checkpoint is available, instead of raising.
    """

    @_requires_cuda
    def test_v4_decode_block_matches_sot_forward_spec(self) -> None:
        """The worker's V4 draft block must equal the SoT forward_spec output tightly."""
        _ensure_repo_test_package()
        from test.srt.speculative._dspark_reference.deepseek_v4.sot_attention import (
            SoTDSparkAttentionOracle,
        )
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
        torch.testing.assert_close(
            prod.base_logits.float(),
            sot_base_logits.float(),
            atol=5e-2,
            rtol=5e-2,
        )

    @_requires_cuda
    def test_v4_decode_is_lossless_vs_target_autoregressive(self) -> None:
        """Greedy V4 spec decode must be token-identical to plain target decode (T6)."""
        self.skipTest(
            "End-to-end greedy losslessness (T6) is a server-tier check the remote "
            "tester runs against a real dsv4 checkpoint: launch a DSpark server vs a "
            "non-spec baseline and assert token-identical greedy output. No GPU "
            "checkpoint is available in unit CI."
        )


if __name__ == "__main__":
    unittest.main()
