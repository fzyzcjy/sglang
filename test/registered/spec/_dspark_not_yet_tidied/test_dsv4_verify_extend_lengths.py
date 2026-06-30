import types
import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    compute_ragged_extend_lengths,
    compute_uniform_extend_lengths,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_layout(verify_lens_cpu: list[int]) -> types.SimpleNamespace:
    verify_lens = torch.tensor(verify_lens_cpu, dtype=torch.int32)
    extend_start_loc = torch.tensor(
        [0, *torch.cumsum(verify_lens, dim=0).tolist()[:-1]], dtype=torch.int32
    )
    return types.SimpleNamespace(
        verify_lens=verify_lens,
        verify_lens_cpu=list(verify_lens_cpu),
        total_verify_tokens=int(verify_lens.sum()),
        extend_start_loc=extend_start_loc,
    )


class TestComputeUniformExtendLengths(CustomTestCase):
    def test_target_verify_uniform_extend(self):
        """Uniform extend grows every request by extend_len with no extend_start_loc."""
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        lengths = compute_uniform_extend_lengths(
            seq_lens=seq_lens, seq_lens_cpu=[10, 20, 30], extend_len=6
        )
        self.assertEqual(lengths.seq_lens_extended.tolist(), [16, 26, 36])
        self.assertEqual(lengths.seq_lens_cpu_extended, [16, 26, 36])
        self.assertEqual(lengths.extend_seq_lens_cpu, [6, 6, 6])
        self.assertEqual(lengths.num_tokens, 18)
        self.assertIsNone(lengths.extend_start_loc)

    def test_dspark_draft_block_extends_by_block_size(self):
        """The DSpark draft block reuses the uniform path with extend_len = block_size."""
        seq_lens = torch.tensor([7, 9], dtype=torch.int32)
        lengths = compute_uniform_extend_lengths(
            seq_lens=seq_lens, seq_lens_cpu=[7, 9], extend_len=4
        )
        self.assertEqual(lengths.num_tokens, 4 * 2)
        self.assertEqual(lengths.extend_seq_lens_cpu, [4, 4])
        self.assertEqual(lengths.seq_lens_extended.tolist(), [11, 13])

    def test_seq_lens_cpu_extended_is_plain_ints(self):
        """seq_lens_cpu_extended is plain ints, never a list of 0-d tensors."""
        seq_lens = torch.tensor([5, 6], dtype=torch.int32)
        lengths = compute_uniform_extend_lengths(
            seq_lens=seq_lens, seq_lens_cpu=[5, 6], extend_len=3
        )
        self.assertTrue(all(type(x) is int for x in lengths.seq_lens_cpu_extended))
        self.assertTrue(all(type(x) is int for x in lengths.extend_seq_lens_cpu))


class TestComputeRaggedExtendLengths(CustomTestCase):
    def test_ragged_extend_per_request_verify_lens(self):
        """Ragged extend grows each request by its own verify_lens and exposes start locs."""
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        layout = _make_layout([6, 3, 1])
        lengths = compute_ragged_extend_lengths(
            seq_lens=seq_lens, seq_lens_cpu=[10, 20, 30], ragged_layout=layout
        )
        self.assertEqual(lengths.seq_lens_extended.tolist(), [16, 23, 31])
        self.assertEqual(lengths.seq_lens_cpu_extended, [16, 23, 31])
        self.assertEqual(lengths.extend_seq_lens_cpu, [6, 3, 1])
        self.assertEqual(lengths.num_tokens, 10)
        self.assertEqual(lengths.extend_start_loc.tolist(), [0, 6, 9])

    def test_seq_lens_cpu_extended_is_plain_ints(self):
        """Ragged seq_lens_cpu_extended is plain ints, never a list of 0-d tensors."""
        seq_lens = torch.tensor([4, 4, 4], dtype=torch.int32)
        layout = _make_layout([6, 3, 1])
        lengths = compute_ragged_extend_lengths(
            seq_lens=seq_lens, seq_lens_cpu=[4, 4, 4], ragged_layout=layout
        )
        self.assertTrue(all(type(x) is int for x in lengths.seq_lens_cpu_extended))


if __name__ == "__main__":
    unittest.main()
