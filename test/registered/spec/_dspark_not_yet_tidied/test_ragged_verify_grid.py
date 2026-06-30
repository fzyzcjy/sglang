import unittest

from sglang.srt.speculative.ragged_verify import (
    build_graph_num_tokens_grid,
    round_up_grid,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestRoundUpGrid(CustomTestCase):
    def test_exact_tier_returns_that_tier(self):
        """A total landing exactly on a tier returns that tier (no padding)."""
        self.assertEqual(round_up_grid(total=10, grid=[4, 8, 10, 16]), 10)

    def test_rounds_up_to_next_tier(self):
        """A total between tiers rounds up to the next captured tier."""
        self.assertEqual(round_up_grid(total=9, grid=[4, 8, 10, 16]), 10)

    def test_rounds_up_from_below_first_tier(self):
        """A total below the smallest tier rounds up to the first tier."""
        self.assertEqual(round_up_grid(total=1, grid=[4, 8, 16]), 4)

    def test_total_above_max_tier_raises(self):
        """A total exceeding the largest tier raises (caller must reject earlier)."""
        with self.assertRaises(ValueError):
            round_up_grid(total=17, grid=[4, 8, 16])

    def test_empty_grid_raises(self):
        """An empty grid raises rather than indexing out of bounds."""
        with self.assertRaises(ValueError):
            round_up_grid(total=1, grid=[])


class TestBuildGraphNumTokensGrid(CustomTestCase):
    def test_unions_uniform_tiers_into_token_grid(self):
        """The grid includes every bs*(gamma+1) tier so uniform batches stay bit-identical."""
        grid = build_graph_num_tokens_grid(
            capture_bs=[1, 2, 4],
            num_tokens_per_bs=8,
            token_grid=[10, 20],
        )
        for bs in (1, 2, 4):
            self.assertIn(bs * 8, grid)
        self.assertIn(10, grid)
        self.assertIn(20, grid)

    def test_result_is_sorted_and_deduped(self):
        """Overlapping uniform and token tiers collapse into one sorted ascending grid."""
        grid = build_graph_num_tokens_grid(
            capture_bs=[1, 2],
            num_tokens_per_bs=8,
            token_grid=[8, 16, 16],
        )
        self.assertEqual(grid, [8, 16])

    def test_rejects_non_positive_num_tokens_per_bs(self):
        """num_tokens_per_bs below 1 is rejected."""
        with self.assertRaises(ValueError):
            build_graph_num_tokens_grid(
                capture_bs=[1],
                num_tokens_per_bs=0,
                token_grid=[8],
            )

    def test_uniform_total_always_lands_on_a_tier(self):
        """round_up_grid on a bs*(gamma+1) total never pads (the C7 invariant)."""
        capture_bs = [1, 3, 7]
        num_tokens_per_bs = 8
        grid = build_graph_num_tokens_grid(
            capture_bs=capture_bs,
            num_tokens_per_bs=num_tokens_per_bs,
            token_grid=[5, 11, 50],
        )
        for bs in capture_bs:
            total = bs * num_tokens_per_bs
            self.assertEqual(round_up_grid(total=total, grid=grid), total)


if __name__ == "__main__":
    unittest.main()
