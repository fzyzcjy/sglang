import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_accept import (
    simulated_correct_drafts,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def simulate(**overrides) -> torch.Tensor:
    values = dict(
        simulate_acc_len=1.0,
        gamma=5,
        bs=16,
        device=torch.device("cpu"),
    )
    values.update(overrides)
    return simulated_correct_drafts(**values)


class TestSimulatedCorrectDrafts(CustomTestCase):
    def test_acc_len_one_gives_zero_correct_drafts(self):
        """Accept length 1.0 (bonus only) yields a constant zero-draft tensor."""
        correct_drafts = simulate()
        self.assertTrue(torch.equal(correct_drafts, torch.zeros(16, dtype=torch.int32)))

    def test_acc_len_counts_the_bonus_token(self):
        """Accept length 4.0 means 3 correct drafts per request."""
        self.assertTrue(bool((simulate(simulate_acc_len=4.0) == 3).all()))

    def test_acc_len_above_gamma_clamps_to_gamma(self):
        """An accept length beyond gamma+1 clamps to gamma correct drafts."""
        self.assertTrue(bool((simulate(simulate_acc_len=9.0) == 5).all()))

    def test_acc_len_below_one_clamps_to_zero(self):
        """An accept length below 1 clamps to zero correct drafts."""
        self.assertTrue(bool((simulate(simulate_acc_len=0.25) == 0).all()))

    def test_fractional_acc_len_rounds_to_nearest(self):
        """A fractional accept length rounds to the nearest integer draft count."""
        self.assertTrue(bool((simulate(simulate_acc_len=4.6) == 4).all()))


if __name__ == "__main__":
    unittest.main()
