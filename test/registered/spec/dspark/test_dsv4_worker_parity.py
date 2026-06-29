import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


@unittest.skip(
    "BLOCKED on dsv4 GPU enablement: the DeepSeek-V4 DSpark draft owns its own "
    "sliding-window MLA KV ring and requires the dsv4 attention backend + real "
    "weights to run forward_spec end-to-end. The worker integration is wired "
    "behind capability dispatch (test_dsv4_worker_smoke covers the dispatch "
    "geometry on CPU); this lossless / SoT-parity entry is a GPU-tier stub. The "
    "reference SoT forward_spec lives outside the repo at "
    "lab/code/scratch/2026-06-29-spark/DeepSeek-V4-Flash-DSpark-inference/model.py "
    "(forward_spec @ line 929)."
)
class TestDsv4WorkerParityVsSoT(CustomTestCase):
    """V4 DSpark worker decode must match the SoT forward_spec (GPU-tier stub)."""

    def test_v4_decode_block_matches_sot_forward_spec(self) -> None:
        """The worker's V4 draft block must equal the SoT forward_spec output tightly."""
        raise NotImplementedError(
            "dsv4 GPU parity harness not wired (blocked on dsv4 backend + weights)."
        )

    def test_v4_decode_is_lossless_vs_target_autoregressive(self) -> None:
        """Greedy V4 spec decode must be token-identical to plain target decode."""
        raise NotImplementedError(
            "dsv4 GPU lossless harness not wired (blocked on dsv4 backend + weights)."
        )


if __name__ == "__main__":
    unittest.main()
