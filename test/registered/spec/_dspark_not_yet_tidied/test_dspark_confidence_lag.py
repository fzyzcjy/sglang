import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
    _value_independent_descending_order,
    compute_verify_token_budget,
    schedule_verify_lens_topk,
)
from sglang.srt.speculative.dspark_components.dspark_sps_table import SpsCostTable
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _flat_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1], sample_steps_per_sec=[1.0], max_batch_tokens=4096
    )


def _planner(*, gamma: int, req_pool_size: int) -> HostConfidenceBudgetPlanner:
    # relay_lag_steps=0 -> the host carry supplies the full causal lag, so the test
    # can drive one confidence per step and read the lag-prior budget directly
    # (the overlap relay's natural lag-1 is exercised on GPU by the lossless harness).
    return HostConfidenceBudgetPlanner(
        sps_table=_flat_table(),
        cfg=DSparkScheduleConfig(gamma=gamma),
        req_pool_size=req_pool_size,
        relay_lag_steps=0,
    )


def _verify_all_budget(*, gamma: int, bs: int) -> int:
    cfg = DSparkScheduleConfig(gamma=gamma)
    return compute_verify_token_budget(
        history_survival_probs=torch.ones((bs, gamma), dtype=torch.float32),
        sps_table=_flat_table(),
        cfg=cfg,
    )


def _drive(planner, *, req_pool_indices, confidences, prefix_lens):
    """Feed one confidence/prefix per decode step, returning the budget K each step."""
    budgets = []
    for confidence, prefix in zip(confidences, prefix_lens):
        budgets.append(
            planner.compute_budget(
                confidence=confidence,
                seq_lens_stamp=prefix,
                prefix_lens=prefix,
                req_pool_indices_cpu=req_pool_indices,
            )
        )
    return budgets


class TestConfidenceLagBarrier(CustomTestCase):
    """The §5.2 causal barrier: a step's budget K is fixed by the confidence carried
    SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS steps earlier, never by the current step's
    confidence -- the lossless-critical timing the host carry now enforces.
    """

    def test_budget_reflects_lag_prior_not_current_confidence(self):
        """With lag=2, step 2's budget equals the lag-prior (step 0) confidence's budget."""
        with envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.override(2):
            gamma = 4
            planner = _planner(gamma=gamma, req_pool_size=2)
            idx = torch.tensor([0, 1])
            low = torch.full((2, gamma), 0.005, dtype=torch.float32)
            high = torch.full((2, gamma), 0.99, dtype=torch.float32)
            confidences = [low, high, high]
            # prefix grows by one token per step -> a valid same-request ancestor.
            prefix_lens = [torch.tensor([10 + s, 20 + s]) for s in range(3)]

            budgets = _drive(
                planner,
                req_pool_indices=idx,
                confidences=confidences,
                prefix_lens=prefix_lens,
            )

            budget_from_low = compute_verify_token_budget(
                history_survival_probs=torch.cumprod(low.float(), dim=1),
                sps_table=_flat_table(),
                cfg=DSparkScheduleConfig(gamma=gamma),
            )
            budget_from_high = compute_verify_token_budget(
                history_survival_probs=torch.cumprod(high.float(), dim=1),
                sps_table=_flat_table(),
                cfg=DSparkScheduleConfig(gamma=gamma),
            )
            self.assertNotEqual(
                budget_from_low,
                budget_from_high,
                "degenerate scenario: low and high confidence give the same budget",
            )
            self.assertEqual(budgets[2], budget_from_low)
            self.assertNotEqual(budgets[2], budget_from_high)

    def test_lag_one_uses_previous_step(self):
        """With lag=1 the carry is one deep, so step 1's budget reflects step 0."""
        with envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.override(1):
            gamma = 4
            planner = _planner(gamma=gamma, req_pool_size=2)
            idx = torch.tensor([0, 1])
            low = torch.full((2, gamma), 0.005, dtype=torch.float32)
            high = torch.full((2, gamma), 0.99, dtype=torch.float32)
            prefix_lens = [torch.tensor([10 + s, 20 + s]) for s in range(2)]

            budgets = _drive(
                planner,
                req_pool_indices=idx,
                confidences=[low, high],
                prefix_lens=prefix_lens,
            )
            budget_from_low = compute_verify_token_budget(
                history_survival_probs=torch.cumprod(low.float(), dim=1),
                sps_table=_flat_table(),
                cfg=DSparkScheduleConfig(gamma=gamma),
            )
            self.assertEqual(budgets[1], budget_from_low)


class TestFreshnessGuard(CustomTestCase):
    """H1: a carry row is keyed by req-pool row, so the lag-prior occupant may be a
    different request. The prefix_len identity stamp masks such rows (and cold starts)
    to a verify-all fallback rather than carrying a stranger's confidence into K.
    """

    def test_cold_start_falls_back_to_verify_all(self):
        """Before lag steps of history exist, the carry is unset so K is verify-all."""
        with envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.override(2):
            gamma = 4
            planner = _planner(gamma=gamma, req_pool_size=2)
            idx = torch.tensor([0, 1])
            low = torch.full((2, gamma), 0.005, dtype=torch.float32)
            budget = planner.compute_budget(
                confidence=low,
                seq_lens_stamp=torch.tensor([10, 20]),
                prefix_lens=torch.tensor([10, 20]),
                req_pool_indices_cpu=idx,
            )
            self.assertEqual(budget, _verify_all_budget(gamma=gamma, bs=2))

    def test_non_ancestor_stamp_falls_back_to_verify_all(self):
        """A carry row whose stamp is not an ancestor of the current prefix (a reused
        slot / different request) is masked to verify-all instead of using its low
        confidence (which would otherwise shrink the budget)."""
        with envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.override(2):
            gamma = 4
            planner = _planner(gamma=gamma, req_pool_size=2)
            idx = torch.tensor([0, 1])
            low = torch.full((2, gamma), 0.005, dtype=torch.float32)
            # Step 0 stamps prefix 5000; step 2's prefix is 5 -> growth negative -> stale.
            confidences = [low, low, low]
            prefix_lens = [
                torch.tensor([5000, 5000]),
                torch.tensor([5001, 5001]),
                torch.tensor([5, 5]),
            ]
            budgets = _drive(
                planner,
                req_pool_indices=idx,
                confidences=confidences,
                prefix_lens=prefix_lens,
            )
            # Masked to verify-all rather than the low-confidence budget.
            self.assertEqual(budgets[2], _verify_all_budget(gamma=gamma, bs=2))


class TestDeviceSortEquivalence(CustomTestCase):
    """The GPU-native verify-lens sort must select exactly what the old host
    keys.sort() selected; only the per-element D2H is gone. Checked on CPU tensors.
    """

    @staticmethod
    def _reference_order(probs, positions, requests, valid):
        masked = torch.where(valid, probs, torch.full_like(probs, float("-inf")))
        keys = sorted(
            range(masked.numel()),
            key=lambda i: (-float(masked[i]), int(positions[i]), int(requests[i]), i),
        )
        return torch.tensor(keys, dtype=torch.int64)

    def test_lsd_radix_matches_host_keys_sort(self):
        """The LSD-radix order equals the reference lexicographic keys.sort() order."""
        torch.manual_seed(0)
        for _ in range(50):
            bs = int(torch.randint(1, 6, (1,)).item())
            gamma = int(torch.randint(2, 6, (1,)).item())
            min_v = int(torch.randint(0, gamma - 1, (1,)).item())
            cw = ((torch.rand(bs, gamma) * 5).round() / 5)[:, min_v:gamma].double()
            ri = torch.arange(bs).view(-1, 1).expand_as(cw)
            pi = torch.arange(min_v, gamma).view(1, -1).expand_as(cw)
            valid = cw >= 1e-6
            probs = cw.reshape(-1)
            requests = ri.reshape(-1)
            positions = pi.reshape(-1)
            flat_valid = valid.reshape(-1)
            got = _value_independent_descending_order(
                probs=probs,
                positions=positions,
                requests=requests,
                valid=flat_valid,
            )
            want = self._reference_order(probs, positions, requests, flat_valid)
            self.assertTrue(torch.equal(got, want))

    def test_min_verify_len_zero_keeps_anchor(self):
        """min_verify_len=0 still clamps every verify_len to >= 1 (the anchor, B5)."""
        gamma = 4
        cfg = DSparkScheduleConfig(gamma=gamma, min_verify_len=0)
        survival = torch.cumprod(
            torch.full((3, gamma), 0.5, dtype=torch.float32), dim=1
        )
        verify_lens = schedule_verify_lens_topk(survival_probs=survival, budget=0, cfg=cfg)
        self.assertTrue(bool((verify_lens >= 1).all()))

    def test_budget_bounds_admitted_extra(self):
        """sum(verify_len - effective_floor) never exceeds the budget K."""
        gamma = 4
        cfg = DSparkScheduleConfig(gamma=gamma, min_verify_len=1)
        survival = torch.cumprod(
            torch.full((5, gamma), 0.9, dtype=torch.float32), dim=1
        )
        budget = 3
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=budget, cfg=cfg
        ).to(torch.int64)
        admitted = int((verify_lens - 1).sum())
        self.assertLessEqual(admitted, budget)


if __name__ == "__main__":
    unittest.main()
