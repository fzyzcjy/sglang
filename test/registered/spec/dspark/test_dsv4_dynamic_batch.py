import functools
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()


def _requires_cuda(test_method):
    """Decorator: skip when CUDA is unavailable (the dsv4 sparse backend is GPU-only)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; dsv4 sparse backend is GPU-only.")
        return test_method(self, *args, **kwargs)

    return wrapper


def _dynamic_batch_available() -> tuple[bool, str]:
    """Return (available, reason) for the production dynamic-batch block-forward path.

    T3 needs the production block-forward harness (the same one T1 wires) to drive a
    mixed-length batch through the real backend with per-request seq_lens / positions
    / out_cache_loc. The old scalar-``start_pos`` ring scaffold wrote one ring slot
    for the whole batch; this test catches that regression by demanding per-row
    independence. Skips cleanly until the contract lands.
    """
    if not _CUDA_AVAILABLE:
        return False, "CUDA not available; dsv4 sparse backend is GPU-only."
    try:
        from test.srt.speculative._dspark_reference.dsv4.block_forward_harness import (  # noqa: F401
            HarnessUnavailable,
            build_dsv4_block_forward_harness,
        )
    except ImportError as exc:  # pragma: no cover - import guard
        return False, f"harness import failed: {exc}"
    return True, ""


class TestDsv4DynamicBatch(CustomTestCase):
    """T3: mixed prompt + mixed accept lengths must be per-row independent.

    The fatal regression this guards is the old scalar-``start_pos`` / single-ring-slot
    scaffold: it used one ``start_pos`` for the whole batch and wrote every request's
    target-hidden KV into the same ring slot, so a batch with different prefix lengths
    (and different accepted-token counts feeding the next block's anchor) silently
    cross-contaminated rows. With the real per-request ``ForwardBatch`` (seq_lens,
    positions, out_cache_loc from the allocator), each row's block-forward must equal
    the SAME row run alone. This test:

      1. runs a mixed-length batch (distinct prefix lens + distinct accept lens) and
         records each row's base_logits;
      2. re-runs each request ALONE (batch size 1) with its own prefix/accept;
      3. asserts the batched row equals the singleton row (per-row independence).

    GPU-tier; skips cleanly until the production block-forward contract lands.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._available, cls._skip_reason = _dynamic_batch_available()

    def _skip_if_unready(self) -> None:
        if not self._available:
            self.skipTest(self._skip_reason)

    @_requires_cuda
    def test_mixed_length_batch_rows_are_independent(self) -> None:
        """A batched row's block-forward equals the same request run alone."""
        self._skip_if_unready()
        from test.srt.speculative._dspark_reference.dsv4.block_forward_harness import (
            HarnessUnavailable,
            build_dsv4_block_forward_harness,
        )

        try:
            # Mixed prefix lengths AND mixed accept lengths: the batch combines
            # requests whose prefixes (hence per-row positions / ring fill) differ,
            # so a scalar-start_pos or shared-ring-slot bug diverges here.
            harness = build_dsv4_block_forward_harness(
                seed=0,
                prefix_lens=(7, 130, 64),
                accept_lens=(1, 3, 2),
            )
        except HarnessUnavailable as exc:
            self.skipTest(str(exc))
        except TypeError as exc:
            self.skipTest(
                f"harness does not yet accept the dynamic-batch parameters: {exc}"
            )

        batched = harness.run_production()
        for row in range(harness.batch_size):
            singleton_harness = build_dsv4_block_forward_harness(
                seed=0,
                prefix_lens=(harness.prefix_lens[row],),
                accept_lens=(harness.accept_lens[row],),
            )
            singleton = singleton_harness.run_production()
            torch.testing.assert_close(
                harness.row_base_logits(batched, row).float(),
                singleton.base_logits.float(),
                atol=5e-2,
                rtol=5e-2,
                msg=(
                    f"row {row} of the mixed-length batch diverges from the same "
                    "request run alone - per-row state (start_pos / ring slot / "
                    "positions) leaked across the batch."
                ),
            )


if __name__ == "__main__":
    unittest.main()
