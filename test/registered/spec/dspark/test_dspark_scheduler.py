import unittest

import torch

from sglang.srt.speculative.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
    compute_verify_token_budget,
    schedule_verify_lens_topk,
)
from sglang.srt.speculative.dspark_sps_table import SpsCostTable
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _flat_table(
    steps_per_sec: float = 1.0, max_batch_tokens: int = 4096
) -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[steps_per_sec],
        max_batch_tokens=max_batch_tokens,
    )


def _cliff_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_steps_per_sec=[1.0, 1.0, 1.0, 0.5, 0.45, 0.44, 0.43, 0.42],
        max_batch_tokens=64,
    )


def _survival_from_confidence(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence, dim=1)


def _bruteforce_budget(
    *,
    history_survival_probs: torch.Tensor,
    sps_table: SpsCostTable,
    cfg: DSparkScheduleConfig,
) -> int:
    num_requests = history_survival_probs.shape[0]
    max_len = cfg.resolved_max_verify_len()
    candidates = history_survival_probs[:, cfg.min_verify_len : max_len].flatten()
    candidates = [float(x) for x in candidates.tolist() if float(x) >= cfg.survival_eps]
    candidates.sort(reverse=True)

    best_extra, best_theta = 0, float("-inf")
    for extra in range(len(candidates) + 1):
        tau_star = num_requests + sum(candidates[:extra])
        theta = tau_star * sps_table.lookup(num_requests + extra)
        if theta > best_theta:
            best_theta, best_extra = theta, extra
    return best_extra


class TestComputeVerifyTokenBudget(CustomTestCase):
    def test_budget_argmax_matches_bruteforce_scan_flat_table(self):
        """compute_verify_token_budget K equals a naive scan with a flat SPS table."""
        torch.manual_seed(0)
        confidence = torch.rand(4, 7, dtype=torch.float32) * 0.4 + 0.55
        survival = _survival_from_confidence(confidence)
        cfg = DSparkScheduleConfig(gamma=7)
        table = _flat_table()
        expected = _bruteforce_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        actual = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        self.assertEqual(actual, expected)

    def test_budget_argmax_matches_bruteforce_scan_across_sps_cliffs(self):
        """K equals the naive global argmax even when SPS cliffs make Theta non-unimodal."""
        torch.manual_seed(1)
        cfg = DSparkScheduleConfig(gamma=7)
        table = _cliff_table()
        for _ in range(20):
            confidence = torch.rand(3, 7, dtype=torch.float32) * 0.5 + 0.45
            survival = _survival_from_confidence(confidence)
            expected = _bruteforce_budget(
                history_survival_probs=survival, sps_table=table, cfg=cfg
            )
            actual = compute_verify_token_budget(
                history_survival_probs=survival, sps_table=table, cfg=cfg
            )
            self.assertEqual(actual, expected)

    def test_budget_excludes_forced_min_positions(self):
        """Candidate pool excludes j <= min_verify_len so K counts only extra positions."""
        survival = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
        cfg_no_min = DSparkScheduleConfig(gamma=4, min_verify_len=0)
        cfg_min2 = DSparkScheduleConfig(gamma=4, min_verify_len=2)
        table = _flat_table()
        budget_no_min = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg_no_min
        )
        budget_min2 = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg_min2
        )
        self.assertEqual(budget_no_min, 4)
        self.assertEqual(budget_min2, 2)

    def test_budget_drops_candidates_below_survival_eps(self):
        """Candidates with a < survival_eps are dropped from the argmax search."""
        survival = torch.tensor([[0.9, 1e-9, 1e-12]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3, survival_eps=1e-6)
        table = _flat_table()
        budget = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        self.assertLessEqual(budget, 1)


class TestScheduleVerifyLensTopk(CustomTestCase):
    def test_topk_does_not_exceed_budget(self):
        """Total extra verify length (above the floor) never exceeds the budget."""
        torch.manual_seed(2)
        survival = _survival_from_confidence(torch.rand(5, 7) * 0.4 + 0.55)
        cfg = DSparkScheduleConfig(gamma=7)
        floor = max(cfg.min_verify_len, 1)
        for budget in (0, 1, 5, 12, 100):
            verify_lens = schedule_verify_lens_topk(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            # verify_lens counts the anchor (= 1 + ell_r); extra is the count above
            # the floor, which is what the budget bounds.
            total_extra = int((verify_lens.to(torch.int64) - floor).sum().item())
            self.assertLessEqual(total_extra, budget)
            self.assertGreaterEqual(int(verify_lens.min().item()), 1)

    def test_total_equals_anchors_plus_lens(self):
        """The forward total R + sum(verify_lens) equals an independently-derived
        R + R*min_verify_len + admitted_count (mid-pool budget, no max clamp)."""
        # One candidate per request lands above the others so budget=2 admits
        # exactly one extra per request (verify_len = 2 = min+1), staying below
        # max_len=4 so the clamp never masks the length and a +1 mutation shifts
        # the total. min/floor clamp is inert (every request gets one extra).
        survival = torch.tensor(
            [[0.90, 0.80, 0.30, 0.20], [0.85, 0.70, 0.25, 0.15]],
            dtype=torch.float32,
        )
        num_requests, max_len, budget = 2, 4, 2
        cfg = DSparkScheduleConfig(gamma=max_len, min_verify_len=1)
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=budget, cfg=cfg
        )
        actual_total = num_requests + int(verify_lens.to(torch.int64).sum().item())
        # RHS derived only from R, min_verify_len, and the admitted count (= budget
        # here, since budget < candidate count), never from sum(verify_lens).
        admitted = budget
        expected_total = num_requests + num_requests * cfg.min_verify_len + admitted
        self.assertEqual(actual_total, expected_total)

    def test_admission_is_contiguous_prefix(self):
        """Under a budget-limited scenario each request's admitted positions form
        the contiguous prefix [min, min+l_r) -- not a scattered subset (C9)."""
        # Survival is strictly monotone non-increasing per request (the cumprod
        # invariant), so the highest-survival positions are always the leftmost.
        survival = torch.tensor(
            [[0.99, 0.98, 0.50, 0.10], [0.97, 0.20, 0.05, 0.01]],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        budget = 3
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=budget, cfg=cfg
        )
        # Independently reproduce the global top-budget selection and group the
        # admitted positions by request.
        num_requests, max_len = survival.shape[0], cfg.resolved_max_verify_len()
        flat = [
            (float(survival[r, p]), p, r)
            for r in range(num_requests)
            for p in range(cfg.min_verify_len, max_len)
            if float(survival[r, p]) >= cfg.survival_eps
        ]
        flat.sort(key=lambda e: (-e[0], e[1], e[2]))
        admitted_positions: dict[int, list[int]] = {r: [] for r in range(num_requests)}
        for _prob, position, request in flat[:budget]:
            admitted_positions[request].append(position)
        for request in range(num_requests):
            positions = sorted(admitted_positions[request])
            count = int(verify_lens[request].item()) - cfg.min_verify_len
            self.assertEqual(
                len(positions),
                count,
                f"request {request}: verify_len count mismatch",
            )
            expected_prefix = list(
                range(cfg.min_verify_len, cfg.min_verify_len + count)
            )
            self.assertEqual(
                positions,
                expected_prefix,
                f"request {request} admitted {positions}, not the prefix "
                f"{expected_prefix}",
            )

    def test_higher_confidence_admitted_first(self):
        """A budget too small for both requests favors the higher-survival
        request's prefix (rank-preserving, most-confident-first) (C14)."""
        # Request 0 dominates request 1 at every position; with budget=2 both
        # extra slots must land on request 0.
        survival = torch.tensor(
            [[0.99, 0.98, 0.97, 0.96], [0.40, 0.30, 0.20, 0.10]],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=2, cfg=cfg
        )
        extra = verify_lens.to(torch.int64) - cfg.min_verify_len
        self.assertEqual(int(extra[0].item()), 2)
        self.assertEqual(int(extra[1].item()), 0)

    def test_survival_helper_matches_manual_cumprod(self):
        """_survival_from_confidence equals torch.cumprod along the time axis.

        NOTE: this guards the test helper only; the scheduler here consumes
        pre-computed survival. The worker's production cumprod is now extracted into
        DSparkWorkerV2._two_steps_prior_k_survival / _current_live_sort_survival and
        covered directly in test_dspark_confidence_lag.py.
        """
        confidence = torch.tensor(
            [[0.9, 0.8, 0.5], [0.7, 0.6, 0.4]], dtype=torch.float32
        )
        survival = _survival_from_confidence(confidence)
        expected = torch.tensor(
            [[0.9, 0.9 * 0.8, 0.9 * 0.8 * 0.5], [0.7, 0.7 * 0.6, 0.7 * 0.6 * 0.4]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(survival, expected, atol=1e-6))

    def test_min_and_max_enter_the_budget(self):
        """min_verify_len floors and max_verify_len caps every per-request length."""
        survival = torch.tensor([[0.99, 0.99, 0.99, 0.99, 0.99]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=5, min_verify_len=1, max_verify_len=3)
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=100, cfg=cfg
        )
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        self.assertLessEqual(int(verify_lens.max().item()), 3)

    def test_budget_zero_returns_min_verify_len(self):
        """budget=0 yields l_r = min_verify_len for all requests."""
        survival = torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=2, min_verify_len=1)
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=0, cfg=cfg
        )
        self.assertTrue(
            torch.equal(verify_lens, torch.tensor([1, 1], dtype=torch.int32))
        )

    def test_large_budget_selects_all_candidates(self):
        """A budget >= candidate count selects clamp(gamma, min, max) per request."""
        survival = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=1000, cfg=cfg
        )
        self.assertEqual(int(verify_lens[0].item()), 3)

    def test_tie_break_is_deterministic(self):
        """Identical inputs produce identical verify_lens across repeated calls."""
        survival = torch.full((3, 4), 0.8, dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=4)
        first = schedule_verify_lens_topk(survival_probs=survival, budget=5, cfg=cfg)
        second = schedule_verify_lens_topk(survival_probs=survival, budget=5, cfg=cfg)
        self.assertTrue(torch.equal(first, second))

    def test_tie_break_is_value_independent(self):
        """Tie-break depends only on (a, position, request), not on token values."""
        survival = torch.tensor([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        floor = max(cfg.min_verify_len, 1)
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=3, cfg=cfg
        )
        total_extra = int((verify_lens.to(torch.int64) - floor).sum().item())
        self.assertEqual(total_extra, 3)


class TestVerifyLenAnchorContract(CustomTestCase):
    """verify_lens counts the anchor (= 1 + ell_r) and must be >= 1 for every
    request: RaggedVerifyLayout rejects < 1 and _cap_correct_len reads
    ell_r = verify_lens - 1. Guards the C1 off-by-one that a non-flat SPS table
    plus small/zero budget used to trigger under the old min_verify_len=0 default.
    """

    def test_default_min_verify_len_is_one(self):
        """The default config floors verify_lens at the anchor (min_verify_len=1)."""
        self.assertEqual(DSparkScheduleConfig(gamma=4).min_verify_len, 1)

    def test_small_budget_keeps_anchor_with_default_config(self):
        """budget in {0,1,2} with the default config never drops below verify_len=1."""
        survival = _survival_from_confidence(
            torch.tensor(
                [[0.95, 0.90, 0.40], [0.80, 0.20, 0.10], [0.99, 0.97, 0.50]],
                dtype=torch.float32,
            )
        )
        cfg = DSparkScheduleConfig(gamma=3)
        for budget in (0, 1, 2):
            verify_lens = schedule_verify_lens_topk(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            self.assertGreaterEqual(int(verify_lens.min().item()), 1)

    def test_explicit_zero_min_still_clamped_to_anchor(self):
        """Even an explicit min_verify_len=0 is clamped to >= 1 (double safeguard)."""
        survival = _survival_from_confidence(
            torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=torch.float32)
        )
        cfg = DSparkScheduleConfig(gamma=3, min_verify_len=0)
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=0, cfg=cfg
        )
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        self.assertTrue(
            torch.equal(verify_lens, torch.tensor([1, 1], dtype=torch.int32))
        )

    def test_non_flat_table_small_budget_feeds_ragged_layout(self):
        """A non-flat SPS table that yields a small budget produces verify_lens
        that RaggedVerifyLayout.from_verify_lens accepts (no ValueError)."""
        # A steep SPS cliff right after the anchor batch (B = num_requests = 2)
        # makes any extra batch token collapse Theta, so the budget argmax keeps B
        # at the anchors only (K == 0).
        table = SpsCostTable(
            sample_batch_tokens=[2, 3],
            sample_steps_per_sec=[1.0, 0.1],
            max_batch_tokens=64,
        )
        cfg = DSparkScheduleConfig(gamma=3)
        scheduler = ConfidencePrefixScheduler(sps_table=table, cfg=cfg)
        survival = _survival_from_confidence(
            torch.tensor([[0.90, 0.80, 0.70], [0.85, 0.60, 0.40]], dtype=torch.float32)
        )
        budget = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        self.assertEqual(budget, 0)
        verify_lens = scheduler.compute_verify_lens(
            k_survival=survival, sort_survival=survival
        )
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        verify_lens_cpu = verify_lens.to(torch.int64).tolist()
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=torch.device("cpu"),
            grid=[sum(verify_lens_cpu)],
        )
        self.assertEqual(layout.verify_lens_cpu, verify_lens_cpu)


class TestNonAnticipating(CustomTestCase):
    def test_lens_topk_non_anticipating_under_future_perturbation(self):
        """Perturbing future-dependent a[r, k:] leaves l_r >= k unchanged (incl. ties)."""
        base = torch.tensor(
            [
                [0.95, 0.90, 0.80, 0.40],
                [0.92, 0.70, 0.30, 0.10],
                [0.99, 0.98, 0.50, 0.05],
            ],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        budget = 5
        baseline = schedule_verify_lens_topk(
            survival_probs=base, budget=budget, cfg=cfg
        )

        request, cut = 1, 2
        for delta in (-0.05, -0.2, 0.05, 0.0):
            perturbed = base.clone()
            future = perturbed[request, cut:]
            perturbed[request, cut:] = torch.clamp(
                torch.minimum(future + delta, base[request, cut - 1]), min=0.0
            )
            verify_lens = schedule_verify_lens_topk(
                survival_probs=perturbed, budget=budget, cfg=cfg
            )
            admitted_prefix_unchanged = min(int(baseline[request].item()), cut) == min(
                int(verify_lens[request].item()), cut
            )
            self.assertTrue(
                admitted_prefix_unchanged,
                msg=f"prefix admission changed under future perturbation delta={delta}",
            )

    def test_other_requests_unaffected_by_one_request_future(self):
        """Perturbing one request's future does not change other requests' lengths."""
        base = torch.tensor(
            [[0.95, 0.90, 0.20], [0.93, 0.88, 0.15]], dtype=torch.float32
        )
        cfg = DSparkScheduleConfig(gamma=3)
        budget = 2
        baseline = schedule_verify_lens_topk(
            survival_probs=base, budget=budget, cfg=cfg
        )
        perturbed = base.clone()
        perturbed[0, 2] = 0.01
        verify_lens = schedule_verify_lens_topk(
            survival_probs=perturbed, budget=budget, cfg=cfg
        )
        self.assertEqual(int(baseline[1].item()), int(verify_lens[1].item()))


class TestConfidencePrefixScheduler(CustomTestCase):
    def test_compute_verify_lens_respects_history_budget(self):
        """Different k_survival tensors yield different budgets and different verify_lens."""
        low_history = torch.tensor([[0.95, 0.30, 0.05]], dtype=torch.float32)
        high_history = torch.tensor([[0.95, 0.90, 0.85]], dtype=torch.float32)
        sort_survival = torch.tensor([[0.95, 0.90, 0.85]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        scheduler = ConfidencePrefixScheduler(sps_table=_cliff_table(), cfg=cfg)
        low_budget = compute_verify_token_budget(
            history_survival_probs=low_history, sps_table=scheduler.sps_table, cfg=cfg
        )
        high_budget = compute_verify_token_budget(
            history_survival_probs=high_history, sps_table=scheduler.sps_table, cfg=cfg
        )
        self.assertNotEqual(low_budget, high_budget, "budgets must differ for this test")
        lens_low = scheduler.compute_verify_lens(
            k_survival=low_history, sort_survival=sort_survival
        )
        lens_high = scheduler.compute_verify_lens(
            k_survival=high_history, sort_survival=sort_survival
        )
        self.assertFalse(
            torch.equal(lens_low, lens_high),
            "different k_survival budgets must produce different verify_lens",
        )

    def test_compute_verify_lens_is_pure_no_state_mutation(self):
        """compute_verify_lens does not write any new attribute to the scheduler."""
        cfg = DSparkScheduleConfig(gamma=3)
        scheduler = ConfidencePrefixScheduler(sps_table=_flat_table(), cfg=cfg)
        attrs_before = set(vars(scheduler).keys())
        k_survival = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        sort_survival = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        scheduler.compute_verify_lens(k_survival=k_survival, sort_survival=sort_survival)
        attrs_after = set(vars(scheduler).keys())
        self.assertEqual(
            attrs_before,
            attrs_after,
            f"compute_verify_lens must not mutate scheduler state; "
            f"new attrs: {attrs_after - attrs_before}",
        )
        self.assertNotIn(
            "cached_budget",
            attrs_after,
            "cached_budget must not exist on ConfidencePrefixScheduler (purity guard)",
        )


class TestDSparkScheduleConfig(CustomTestCase):
    def test_validate_rejects_min_greater_than_max(self):
        """validate rejects min_verify_len > max_verify_len."""
        with self.assertRaises(ValueError):
            DSparkScheduleConfig(gamma=4, min_verify_len=3, max_verify_len=2).validate()

    def test_validate_rejects_max_greater_than_gamma(self):
        """validate rejects max_verify_len > gamma."""
        with self.assertRaises(ValueError):
            DSparkScheduleConfig(gamma=4, max_verify_len=5).validate()

    def test_zero_max_resolves_to_gamma(self):
        """max_verify_len=0 resolves to gamma."""
        cfg = DSparkScheduleConfig(gamma=7)
        self.assertEqual(cfg.resolved_max_verify_len(), 7)


if __name__ == "__main__":
    unittest.main()
