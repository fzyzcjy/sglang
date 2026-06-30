import unittest

import torch

from sglang.srt.speculative.dspark_scheduler import (
    ConfidencePrefixScheduler,
    DSparkScheduleConfig,
    compute_verify_token_budget,
    schedule_verify_lens,
    schedule_verify_lens_greedy,
    schedule_verify_lens_topk,
)
from sglang.srt.speculative.dspark_sps_table import SpsCostTable
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
        """Total extra verify length never exceeds the supplied budget."""
        torch.manual_seed(2)
        survival = _survival_from_confidence(torch.rand(5, 7) * 0.4 + 0.55)
        cfg = DSparkScheduleConfig(gamma=7)
        for budget in (0, 1, 5, 12, 100):
            verify_lens = schedule_verify_lens_topk(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            total_extra = int((verify_lens.to(torch.int64)).sum().item())
            self.assertLessEqual(total_extra, budget)

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

        NOTE: this guards the test helper only. The worker's production
        survival = torch.cumprod(confidence) in DSparkWorkerV2._schedule_verify_lens
        is NOT yet covered by a production-path UT (it is not extracted into a
        testable pure function); the scheduler here consumes pre-computed
        survival. A cumprod->cumsum mutation in the worker would still pass.
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
        verify_lens = schedule_verify_lens_topk(
            survival_probs=survival, budget=3, cfg=cfg
        )
        total_extra = int(verify_lens.to(torch.int64).sum().item())
        self.assertEqual(total_extra, 3)


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


class TestScheduleVerifyLensGreedy(CustomTestCase):
    def test_greedy_appendix_a_early_stops_at_theta_drop(self):
        """Appendix A: a_1=0.8, SPS=(1.0,0.5,0.45) -> Theta drops at first extra so l_r=0."""
        survival = torch.tensor(
            [[0.8, 0.8 * 0.8, 0.8 * 0.8 * 0.8]], dtype=torch.float32
        )
        table = SpsCostTable(
            sample_batch_tokens=[1, 2, 3],
            sample_steps_per_sec=[1.0, 0.5, 0.45],
            max_batch_tokens=8,
        )
        cfg = DSparkScheduleConfig(gamma=3)
        verify_lens = schedule_verify_lens_greedy(
            survival_probs=survival, sps_table=table, cfg=cfg
        )
        self.assertEqual(int(verify_lens[0].item()), 0)

    def test_greedy_admits_extra_when_theta_increases(self):
        """Greedy admits one extra when a high-confidence draft keeps Theta rising."""
        survival = torch.tensor([[0.95, 0.90, 0.85]], dtype=torch.float32)
        table = SpsCostTable(
            sample_batch_tokens=[1, 2, 3, 4],
            sample_steps_per_sec=[1.0, 1.0, 0.5, 0.45],
            max_batch_tokens=8,
        )
        cfg = DSparkScheduleConfig(gamma=3)
        verify_lens = schedule_verify_lens_greedy(
            survival_probs=survival, sps_table=table, cfg=cfg
        )
        self.assertGreaterEqual(int(verify_lens[0].item()), 1)

    def test_greedy_zero_when_first_step_drops_theta(self):
        """Greedy admits nothing when adding the first candidate decreases Theta."""
        survival = torch.tensor([[0.1, 0.01]], dtype=torch.float32)
        table = SpsCostTable(
            sample_batch_tokens=[1, 2, 3],
            sample_steps_per_sec=[1.0, 0.2, 0.1],
            max_batch_tokens=8,
        )
        cfg = DSparkScheduleConfig(gamma=2)
        verify_lens = schedule_verify_lens_greedy(
            survival_probs=survival, sps_table=table, cfg=cfg
        )
        self.assertEqual(int(verify_lens[0].item()), 0)

    def test_greedy_respects_min_verify_len(self):
        """Greedy floors every per-request length at min_verify_len."""
        survival = torch.tensor([[0.05, 0.01, 0.005]], dtype=torch.float32)
        table = SpsCostTable(
            sample_batch_tokens=[1, 2, 3, 4],
            sample_steps_per_sec=[1.0, 0.1, 0.05, 0.01],
            max_batch_tokens=8,
        )
        cfg = DSparkScheduleConfig(gamma=3, min_verify_len=1)
        verify_lens = schedule_verify_lens_greedy(
            survival_probs=survival, sps_table=table, cfg=cfg
        )
        self.assertGreaterEqual(int(verify_lens[0].item()), 1)


class TestConfidencePrefixScheduler(CustomTestCase):
    def test_cold_start_verifies_all(self):
        """With no cached budget the scheduler falls back to verify-all (max per request)."""
        survival = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        scheduler = ConfidencePrefixScheduler(sps_table=_flat_table(), cfg=cfg)
        verify_lens = scheduler.compute_verify_lens(survival_probs=survival)
        self.assertEqual(int(verify_lens[0].item()), 3)

    def test_update_budget_then_compute_uses_cached_budget(self):
        """update_budget_from_history caches K and compute_verify_lens respects it."""
        history = torch.tensor([[0.95, 0.30, 0.05]], dtype=torch.float32)
        now = torch.tensor([[0.95, 0.90, 0.85]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        scheduler = ConfidencePrefixScheduler(sps_table=_cliff_table(), cfg=cfg)
        budget = scheduler.update_budget_from_history(history_survival_probs=history)
        self.assertEqual(scheduler.cached_budget, budget)
        verify_lens = scheduler.compute_verify_lens(survival_probs=now)
        total_extra = int(verify_lens.to(torch.int64).sum().item())
        self.assertLessEqual(total_extra, budget)


class TestScheduleVerifyLensHook(CustomTestCase):
    def test_hook_returns_none_without_scheduler(self):
        """schedule_verify_lens returns None (uniform fallback) when scheduler is None."""
        survival = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
        self.assertIsNone(schedule_verify_lens(scheduler=None, survival_probs=survival))

    def test_hook_returns_none_without_survival_probs(self):
        """schedule_verify_lens returns None when survival_probs is None."""
        cfg = DSparkScheduleConfig(gamma=2)
        scheduler = ConfidencePrefixScheduler(sps_table=_flat_table(), cfg=cfg)
        self.assertIsNone(
            schedule_verify_lens(scheduler=scheduler, survival_probs=None)
        )

    def test_hook_returns_verify_lens_tensor(self):
        """schedule_verify_lens returns a per-request verify_lens tensor when wired."""
        survival = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        scheduler = ConfidencePrefixScheduler(sps_table=_flat_table(), cfg=cfg)
        verify_lens = schedule_verify_lens(scheduler=scheduler, survival_probs=survival)
        self.assertIsNotNone(verify_lens)
        self.assertEqual(tuple(verify_lens.shape), (1,))


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
