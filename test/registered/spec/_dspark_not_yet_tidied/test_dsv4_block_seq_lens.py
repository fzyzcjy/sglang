import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    build_block_seq_lens_casual,
    compute_dspark_window_gather,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CPU = torch.device("cpu")


class TestBuildBlockSeqLensCasual(CustomTestCase):
    def test_flatten_matches_per_request_steps(self):
        """Each request emits [prefix+1 .. prefix+block_size], flattened row-major."""
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        out = build_block_seq_lens_casual(seq_lens=seq_lens, block_size=4, device=_CPU)
        self.assertEqual(out.tolist(), [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34])

    def test_block_size_one_is_seq_lens_plus_one(self):
        """block_size=1 yields exactly seq_lens + 1."""
        seq_lens = torch.tensor([7, 0, 128], dtype=torch.int32)
        out = build_block_seq_lens_casual(seq_lens=seq_lens, block_size=1, device=_CPU)
        self.assertEqual(out.tolist(), [8, 1, 129])

    def test_shape_dtype_device(self):
        """Output is [bs*block_size] int32 on the requested device."""
        seq_lens = torch.tensor([3, 4], dtype=torch.int32)
        out = build_block_seq_lens_casual(seq_lens=seq_lens, block_size=5, device=_CPU)
        self.assertEqual(tuple(out.shape), (2 * 5,))
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.device, _CPU)

    def test_roundtrip_with_window_gather_prefix(self):
        """+1 in build_block_seq_lens_casual cancels the -1 prefix read in the window gather."""
        block_size = 3
        prefixes = torch.tensor([5, 130], dtype=torch.int32)
        seq_lens_casual = build_block_seq_lens_casual(
            seq_lens=prefixes, block_size=block_size, device=_CPU
        )
        bs = prefixes.numel()
        first_token = seq_lens_casual.view(bs, block_size)[:, 0]
        self.assertTrue(torch.equal(first_token - 1, prefixes))

        gather = compute_dspark_window_gather(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=torch.zeros(bs * block_size, dtype=torch.int32),
            block_size=block_size,
        )
        self.assertTrue(
            torch.equal(gather.context_lens, torch.clamp(prefixes, max=SWA_WINDOW))
        )
        self.assertTrue(
            torch.equal(gather.offsets[:, -1], (prefixes - 1).to(torch.int64))
        )


if __name__ == "__main__":
    unittest.main()
