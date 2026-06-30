import unittest

import torch

from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GRID = [1, 2, 4, 8, 10, 16, 24, 32, 64]


class TestRaggedVerifyLayoutGeometry(CustomTestCase):
    def test_canonical_geometry_for_mixed_verify_lens(self):
        """Mixed verify_lens produce correct totals, exclusive-cumsum, and qo_indptr."""
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[5, 1, 4],
            device=_DEVICE,
            grid=_GRID,
        )
        self.assertEqual(layout.verify_lens.tolist(), [5, 1, 4])
        self.assertEqual(layout.total_verify_tokens, 10)
        self.assertEqual(layout.graph_num_tokens, 10)
        self.assertEqual(layout.extend_start_loc.tolist(), [0, 5, 6])
        self.assertEqual(layout.qo_indptr_device.tolist(), [0, 5, 6, 10])
        self.assertEqual(layout.bs, 3)

    def test_extend_start_loc_is_exclusive_cumsum(self):
        """extend_start_loc equals the exclusive cumsum of verify_lens (forward_batch parity)."""
        verify_lens = [3, 7, 2, 6]
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens,
            device=_DEVICE,
            grid=_GRID,
        )
        ref = torch.tensor(verify_lens, dtype=torch.int32)
        expected = torch.zeros_like(ref)
        expected[1:] = torch.cumsum(ref[:-1], dim=0)
        self.assertEqual(layout.extend_start_loc.tolist(), expected.tolist())

    def test_qo_indptr_is_zero_prefixed_cumsum(self):
        """qo_indptr_device equals [0, *cumsum(verify_lens)] with int32 dtype."""
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[2, 2, 2],
            device=_DEVICE,
            grid=_GRID,
        )
        self.assertEqual(layout.qo_indptr_device.tolist(), [0, 2, 4, 6])
        self.assertEqual(layout.qo_indptr_device.dtype, torch.int32)
        self.assertEqual(layout.verify_lens.dtype, torch.int32)
        self.assertEqual(layout.extend_start_loc.dtype, torch.int32)

    def test_graph_num_tokens_rounds_up_to_grid(self):
        """graph_num_tokens rounds total up to the next grid tier."""
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[5, 5, 1],
            device=_DEVICE,
            grid=_GRID,
        )
        self.assertEqual(layout.total_verify_tokens, 11)
        self.assertEqual(layout.graph_num_tokens, 16)

    def test_single_request_anchor_only(self):
        """A single anchor-only request (verify_len == 1) is valid."""
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1],
            device=_DEVICE,
            grid=_GRID,
        )
        self.assertEqual(layout.total_verify_tokens, 1)
        self.assertEqual(layout.graph_num_tokens, 1)
        self.assertEqual(layout.qo_indptr_device.tolist(), [0, 1])


class TestRaggedVerifyLayoutUniform(CustomTestCase):
    def test_uniform_matches_static_full_block_shape(self):
        """uniform produces the static full-block geometry (bs*(gamma+1) total, even stride)."""
        bs, num_draft_tokens = 4, 8
        layout = RaggedVerifyLayout.uniform(
            bs=bs,
            num_draft_tokens=num_draft_tokens,
            device=_DEVICE,
            grid=_GRID,
        )
        self.assertEqual(layout.verify_lens_cpu, [8, 8, 8, 8])
        self.assertEqual(layout.total_verify_tokens, 32)
        self.assertEqual(layout.graph_num_tokens, 32)
        expected_qo = list(range(0, (bs + 1) * num_draft_tokens, num_draft_tokens))
        self.assertEqual(layout.qo_indptr_device.tolist(), expected_qo)

    def test_uniform_total_lands_exactly_on_grid_tier(self):
        """A uniform total lands on its own grid tier so no padding occurs."""
        layout = RaggedVerifyLayout.uniform(
            bs=2,
            num_draft_tokens=8,
            device=_DEVICE,
            grid=_GRID,
        )
        self.assertEqual(layout.total_verify_tokens, layout.graph_num_tokens)


class TestRaggedVerifyLayoutValidation(CustomTestCase):
    def test_rejects_empty_batch(self):
        """An empty verify_lens list is rejected."""
        with self.assertRaises(ValueError):
            RaggedVerifyLayout.from_verify_lens(
                verify_lens_cpu=[],
                device=_DEVICE,
                grid=_GRID,
            )

    def test_rejects_verify_len_below_one(self):
        """A verify_len below 1 (anchor not verified) is rejected."""
        with self.assertRaises(ValueError):
            RaggedVerifyLayout.from_verify_lens(
                verify_lens_cpu=[3, 0, 2],
                device=_DEVICE,
                grid=_GRID,
            )

    def test_rejects_total_above_max_grid_tier(self):
        """A total exceeding the largest grid tier is rejected at construction."""
        with self.assertRaises(ValueError):
            RaggedVerifyLayout.from_verify_lens(
                verify_lens_cpu=[64, 64],
                device=_DEVICE,
                grid=_GRID,
            )

    def test_post_init_rejects_inconsistent_total(self):
        """A hand-built layout whose total disagrees with verify_lens_cpu is rejected."""
        with self.assertRaises(ValueError):
            RaggedVerifyLayout(
                verify_lens=torch.tensor([2, 2], dtype=torch.int32),
                verify_lens_cpu=[2, 2],
                total_verify_tokens=5,
                extend_start_loc=torch.tensor([0, 2], dtype=torch.int32),
                qo_indptr_device=torch.tensor([0, 2, 4], dtype=torch.int32),
                graph_num_tokens=8,
            )

    def test_post_init_rejects_total_above_graph_num_tokens(self):
        """A hand-built layout whose total exceeds graph_num_tokens is rejected."""
        with self.assertRaises(ValueError):
            RaggedVerifyLayout(
                verify_lens=torch.tensor([4, 4], dtype=torch.int32),
                verify_lens_cpu=[4, 4],
                total_verify_tokens=8,
                extend_start_loc=torch.tensor([0, 4], dtype=torch.int32),
                qo_indptr_device=torch.tensor([0, 4, 8], dtype=torch.int32),
                graph_num_tokens=4,
            )


if __name__ == "__main__":
    unittest.main()
