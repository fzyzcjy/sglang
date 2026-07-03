import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_accept import (
    sample_simulated_correct_drafts,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def sample(**overrides) -> torch.Tensor:
    values = dict(
        simulate_acc_len=4.5,
        simulate_acc_method="match-expected",
        gamma=5,
        bs=4096,
        forward_ct=7,
        device=torch.device("cpu"),
    )
    values.update(overrides)
    return sample_simulated_correct_drafts(**values)


class TestSampleSimulatedCorrectDrafts(CustomTestCase):
    def test_match_expected_mean_matches_target(self):
        """match-expected sampling averages to simulate_acc_len - 1 correct drafts."""
        correct_drafts = sample()
        self.assertAlmostEqual(
            float(correct_drafts.float().mean()), 3.5, delta=0.05
        )

    def test_values_stay_within_draft_bounds(self):
        """Sampled correct drafts never leave [0, gamma] for either method."""
        for method in ("match-expected", "multinomial"):
            correct_drafts = sample(simulate_acc_method=method, simulate_acc_len=9.0)
            self.assertGreaterEqual(int(correct_drafts.min()), 0)
            self.assertLessEqual(int(correct_drafts.max()), 5)

    def test_acc_len_below_one_clamps_to_zero_drafts(self):
        """An accept length of 1 (bonus only) yields zero correct drafts."""
        correct_drafts = sample(simulate_acc_len=1.0)
        self.assertEqual(int(correct_drafts.max()), 0)

    def test_same_forward_ct_is_deterministic_across_calls(self):
        """Two ranks sampling at the same forward_ct derive identical lengths."""
        self.assertTrue(torch.equal(sample(forward_ct=42), sample(forward_ct=42)))

    def test_different_forward_ct_gives_different_draws(self):
        """Consecutive steps do not repeat the same simulated pattern."""
        self.assertFalse(torch.equal(sample(forward_ct=1), sample(forward_ct=2)))

    def test_unknown_method_raises(self):
        """An unknown simulate_acc_method is rejected."""
        with self.assertRaises(ValueError):
            sample(simulate_acc_method="bogus")


if __name__ == "__main__":
    unittest.main()
