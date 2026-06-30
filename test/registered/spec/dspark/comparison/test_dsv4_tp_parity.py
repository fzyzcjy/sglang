import functools
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# TP=2 parity needs >=2 GPUs.
register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()
_NUM_GPUS = torch.cuda.device_count() if _CUDA_AVAILABLE else 0

# n_heads chosen so TP=2 yields n_local_heads NOT in {64, 128}. MQALayer pads q to
# 64 heads for FlashMLA's fp8 sparse decode kernel (which only specializes h_q for
# {64, 128}); agent A's DSparkAttention does NOT replicate that padding. With
# n_heads=64, TP=1 gives n_local_heads=64 (a specialized value, fine) but TP=2 gives
# n_local_heads=32 - exactly the unspecialized count where the missing q-pad would
# make the fp8 sparse kernel dispatch fail or mis-shape. This is the discriminating
# config A flagged: T4 must run it so a TP>1 head regression is actually caught.
_TP_GAP_NUM_HEADS = 64


def _requires_two_gpus(test_method):
    """Decorator: skip unless >=2 CUDA devices (TP=2 needs two ranks)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; dsv4 sparse backend is GPU-only.")
        if _NUM_GPUS < 2:
            self.skipTest(f"TP=2 parity needs 2 GPUs, found {_NUM_GPUS}.")
        return test_method(self, *args, **kwargs)

    return wrapper


def _tp_parity_available() -> tuple[bool, str]:
    """Return (available, reason) for the TP=2-vs-TP=1 block-forward parity path."""
    if not _CUDA_AVAILABLE:
        return False, "CUDA not available; dsv4 sparse backend is GPU-only."
    if _NUM_GPUS < 2:
        return False, f"TP=2 parity needs 2 GPUs, found {_NUM_GPUS}."
    try:
        from test.srt.speculative._dspark_reference.dsv4.block_forward_harness import (  # noqa: F401
            HarnessUnavailable,
            build_dsv4_block_forward_harness,
        )
    except ImportError as exc:  # pragma: no cover - import guard
        return False, f"harness import failed: {exc}"
    return True, ""


class TestDsv4TpParity(CustomTestCase):
    """T4: TP=2 block-forward must match TP=1 (and not crash on the q-pad gap).

    Two regressions this guards:

      1. The old ``n_local_heads`` was set to the FULL ``n_heads`` (not divided by
         tp_size), so any TP>1 run had wrong head shapes. TP=2-vs-TP=1 parity catches
         that directly.
      2. (agent-A-flagged) MQALayer pads q to 64 heads for FlashMLA's fp8 sparse
         decode kernel; DSparkAttention does not. With ``n_heads=64`` the TP=2 shard
         is ``n_local_heads=32`` - NOT a specialized {64,128} count - so a missing
         q-pad would make the kernel dispatch fail or mis-shape on exactly this
         config. T4 runs this config so that risk is exercised, not hidden behind a
         conveniently-specialized head count.

    GPU/2-GPU-tier; skips cleanly until the production block-forward contract lands.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._available, cls._skip_reason = _tp_parity_available()

    def _skip_if_unready(self) -> None:
        if not self._available:
            self.skipTest(self._skip_reason)

    @_requires_two_gpus
    def test_tp2_block_forward_matches_tp1_on_q_pad_gap_config(self) -> None:
        """TP=2 base_logits match TP=1 for an n_local_heads NOT in {64,128} config."""
        self._skip_if_unready()
        from test.srt.speculative._dspark_reference.dsv4.block_forward_harness import (
            HarnessUnavailable,
            build_dsv4_block_forward_harness,
        )

        try:
            tp1 = build_dsv4_block_forward_harness(
                seed=0, num_heads=_TP_GAP_NUM_HEADS, tp_size=1
            )
        except HarnessUnavailable as exc:
            self.skipTest(str(exc))
        except TypeError as exc:
            self.skipTest(
                f"harness does not yet accept the TP/num_heads parameters: {exc}"
            )

        # Guard the discriminating property: at TP=2 the per-rank head count must be
        # the unspecialized 32, NOT a {64,128} value, else this test is vacuous.
        self.assertNotIn(
            _TP_GAP_NUM_HEADS // 2,
            (64, 128),
            "T4 config must put n_local_heads outside {64,128} at TP=2 to exercise "
            "the missing-q-pad risk.",
        )

        tp1_out = tp1.run_production()
        tp2 = build_dsv4_block_forward_harness(
            seed=0, num_heads=_TP_GAP_NUM_HEADS, tp_size=2
        )
        tp2_out = tp2.run_production()

        torch.testing.assert_close(
            tp2_out.base_logits.float(),
            tp1_out.base_logits.float(),
            atol=5e-2,
            rtol=5e-2,
            msg=(
                "TP=2 block-forward diverges from TP=1: either n_local_heads is not "
                "divided by tp_size, or the fp8 sparse kernel mis-handles the "
                "unspecialized per-rank head count (missing q-pad to 64)."
            ),
        )


if __name__ == "__main__":
    unittest.main()
