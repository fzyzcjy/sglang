import functools
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    DSparkScheduleConfig,
    compute_verify_token_budget,
    schedule_verify_lens_topk_from_survival,
)
from sglang.srt.speculative.dspark_components.dspark_sps_table import SpsCostTable
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


def schedule_verify_lens_topk_vanilla(
    *,
    survival_probs: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    """Readable plain-Python reference for ``schedule_verify_lens_topk``.

    Same contract as the production function: spend a global ``budget`` of extra
    verify positions across all requests, always taking the highest-survival
    candidates first, then turn each request's admitted count into a per-request
    ``verify_len`` (anchor + admitted drafts). The production version expresses this
    with GPU tensor ops plus a value-independent argsort; this one is a flat list
    sort so the equivalence is easy to read and check.
    """
    cfg.validate()
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device

    # Gate candidates in the input dtype (matches the production
    # ``candidate_window >= survival_eps``) and read survival as float64 for ranking.
    valid_rows = (survival_probs >= cfg.survival_eps).tolist()
    survival_rows = survival_probs.to(torch.float64).tolist()

    # One candidate per (request, extra position) in the window [min_verify_len,
    # max_len) whose survival clears the eps gate.
    candidates: list[tuple[float, int, int]] = []
    for request in range(num_requests):
        for position in range(cfg.min_verify_len, max_len):
            if valid_rows[request][position]:
                candidates.append((survival_rows[request][position], position, request))

    # Rank by survival descending, breaking ties by (position, request) ascending so
    # the order depends only on coordinates, never on token values (value-independent
    # tie-break). (position, request) is unique per candidate, so no further key is
    # needed.
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))

    selected_extra = [0] * num_requests
    for _survival, _position, request in candidates[: max(int(budget), 0)]:
        selected_extra[request] += 1

    # verify_len counts the anchor (= 1 + admitted drafts); floor at
    # max(min_verify_len, 1) so an explicit min_verify_len=0 still keeps the anchor,
    # and cap at max_len.
    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = [
        min(max(cfg.min_verify_len + extra, lower_bound), max_len)
        for extra in selected_extra
    ]
    return torch.tensor(verify_lens, dtype=torch.int32, device=device)


# Every property test below runs against both the production function and the
# vanilla reference (parameterized via subTest), so the readable reference is held
# to exactly the same contract.
_TOPK_IMPLS = (
    schedule_verify_lens_topk_from_survival,
    schedule_verify_lens_topk_vanilla,
)


def _for_each_impl(test_method):
    @functools.wraps(test_method)
    def wrapper(self):
        for impl in _TOPK_IMPLS:
            with self.subTest(impl=impl.__name__):
                test_method(self, impl)

    return wrapper


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
    @_for_each_impl
    def test_topk_does_not_exceed_budget(self, impl):
        """Total extra verify length (above the floor) never exceeds the budget."""
        torch.manual_seed(2)
        survival = _survival_from_confidence(torch.rand(5, 7) * 0.4 + 0.55)
        cfg = DSparkScheduleConfig(gamma=7)
        floor = max(cfg.min_verify_len, 1)
        for budget in (0, 1, 5, 12, 100):
            verify_lens = impl(survival_probs=survival, budget=budget, cfg=cfg)
            # verify_lens counts the anchor (= 1 + ell_r); extra is the count above
            # the floor, which is what the budget bounds.
            total_extra = int((verify_lens.to(torch.int64) - floor).sum().item())
            self.assertLessEqual(total_extra, budget)
            self.assertGreaterEqual(int(verify_lens.min().item()), 1)

    @_for_each_impl
    def test_total_equals_anchors_plus_lens(self, impl):
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
        verify_lens = impl(survival_probs=survival, budget=budget, cfg=cfg)
        actual_total = num_requests + int(verify_lens.to(torch.int64).sum().item())
        # RHS derived only from R, min_verify_len, and the admitted count (= budget
        # here, since budget < candidate count), never from sum(verify_lens).
        admitted = budget
        expected_total = num_requests + num_requests * cfg.min_verify_len + admitted
        self.assertEqual(actual_total, expected_total)

    @_for_each_impl
    def test_admission_is_contiguous_prefix(self, impl):
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
        verify_lens = impl(survival_probs=survival, budget=budget, cfg=cfg)
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

    @_for_each_impl
    def test_higher_confidence_admitted_first(self, impl):
        """A budget too small for both requests favors the higher-survival
        request's prefix (rank-preserving, most-confident-first) (C14)."""
        # Request 0 dominates request 1 at every position; with budget=2 both
        # extra slots must land on request 0.
        survival = torch.tensor(
            [[0.99, 0.98, 0.97, 0.96], [0.40, 0.30, 0.20, 0.10]],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        verify_lens = impl(survival_probs=survival, budget=2, cfg=cfg)
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

    @_for_each_impl
    def test_min_and_max_enter_the_budget(self, impl):
        """min_verify_len floors and max_verify_len caps every per-request length."""
        survival = torch.tensor([[0.99, 0.99, 0.99, 0.99, 0.99]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=5, min_verify_len=1, max_verify_len=3)
        verify_lens = impl(survival_probs=survival, budget=100, cfg=cfg)
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        self.assertLessEqual(int(verify_lens.max().item()), 3)

    @_for_each_impl
    def test_budget_zero_returns_min_verify_len(self, impl):
        """budget=0 yields l_r = min_verify_len for all requests."""
        survival = torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=2, min_verify_len=1)
        verify_lens = impl(survival_probs=survival, budget=0, cfg=cfg)
        self.assertTrue(
            torch.equal(verify_lens, torch.tensor([1, 1], dtype=torch.int32))
        )

    @_for_each_impl
    def test_large_budget_selects_all_candidates(self, impl):
        """A budget >= candidate count selects clamp(gamma, min, max) per request."""
        survival = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        verify_lens = impl(survival_probs=survival, budget=1000, cfg=cfg)
        self.assertEqual(int(verify_lens[0].item()), 3)

    @_for_each_impl
    def test_tie_break_is_deterministic(self, impl):
        """Identical inputs produce identical verify_lens across repeated calls."""
        survival = torch.full((3, 4), 0.8, dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=4)
        first = impl(survival_probs=survival, budget=5, cfg=cfg)
        second = impl(survival_probs=survival, budget=5, cfg=cfg)
        self.assertTrue(torch.equal(first, second))

    @_for_each_impl
    def test_tie_break_is_value_independent(self, impl):
        """Tie-break depends only on (a, position, request), not on token values."""
        survival = torch.tensor([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        floor = max(cfg.min_verify_len, 1)
        verify_lens = impl(survival_probs=survival, budget=3, cfg=cfg)
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

    @_for_each_impl
    def test_small_budget_keeps_anchor_with_default_config(self, impl):
        """budget in {0,1,2} with the default config never drops below verify_len=1."""
        survival = _survival_from_confidence(
            torch.tensor(
                [[0.95, 0.90, 0.40], [0.80, 0.20, 0.10], [0.99, 0.97, 0.50]],
                dtype=torch.float32,
            )
        )
        cfg = DSparkScheduleConfig(gamma=3)
        for budget in (0, 1, 2):
            verify_lens = impl(survival_probs=survival, budget=budget, cfg=cfg)
            self.assertGreaterEqual(int(verify_lens.min().item()), 1)

    @_for_each_impl
    def test_explicit_zero_min_still_clamped_to_anchor(self, impl):
        """Even an explicit min_verify_len=0 is clamped to >= 1 (double safeguard)."""
        survival = _survival_from_confidence(
            torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=torch.float32)
        )
        cfg = DSparkScheduleConfig(gamma=3, min_verify_len=0)
        verify_lens = impl(survival_probs=survival, budget=0, cfg=cfg)
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
        survival = _survival_from_confidence(
            torch.tensor([[0.90, 0.80, 0.70], [0.85, 0.60, 0.40]], dtype=torch.float32)
        )
        budget = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        self.assertEqual(budget, 0)
        verify_lens = schedule_verify_lens_topk_from_survival(
            survival_probs=survival, budget=budget, cfg=cfg
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
    @_for_each_impl
    def test_lens_topk_non_anticipating_under_future_perturbation(self, impl):
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
        baseline = impl(survival_probs=base, budget=budget, cfg=cfg)

        request, cut = 1, 2
        for delta in (-0.05, -0.2, 0.05, 0.0):
            perturbed = base.clone()
            future = perturbed[request, cut:]
            perturbed[request, cut:] = torch.clamp(
                torch.minimum(future + delta, base[request, cut - 1]), min=0.0
            )
            verify_lens = impl(survival_probs=perturbed, budget=budget, cfg=cfg)
            admitted_prefix_unchanged = min(int(baseline[request].item()), cut) == min(
                int(verify_lens[request].item()), cut
            )
            self.assertTrue(
                admitted_prefix_unchanged,
                msg=f"prefix admission changed under future perturbation delta={delta}",
            )

    @_for_each_impl
    def test_other_requests_unaffected_by_one_request_future(self, impl):
        """Perturbing one request's future does not change other requests' lengths."""
        base = torch.tensor(
            [[0.95, 0.90, 0.20], [0.93, 0.88, 0.15]], dtype=torch.float32
        )
        cfg = DSparkScheduleConfig(gamma=3)
        budget = 2
        baseline = impl(survival_probs=base, budget=budget, cfg=cfg)
        perturbed = base.clone()
        perturbed[0, 2] = 0.01
        verify_lens = impl(survival_probs=perturbed, budget=budget, cfg=cfg)
        self.assertEqual(int(baseline[1].item()), int(verify_lens[1].item()))


class TestVanillaMatchesReference(CustomTestCase):
    def test_random_inputs_match_reference(self):
        """schedule_verify_lens_topk_vanilla matches schedule_verify_lens_topk
        bit-for-bit across many randomized (survival, budget, cfg) inputs."""
        torch.manual_seed(20260630)
        num_trials = 4000
        for trial in range(num_trials):
            num_requests = int(torch.randint(1, 6, ()).item())
            gamma = int(torch.randint(1, 9, ()).item())

            # Mix dtypes (stresses the eps-gate dtype path), continuous vs coarse
            # confidence (coarse -> exact ties), and an occasional saturated row.
            dtype = torch.float32 if trial % 2 == 0 else torch.float64
            confidence = torch.rand(num_requests, gamma, dtype=dtype)
            if trial % 3 == 0:
                confidence = (confidence * 4).round() / 4
            if trial % 7 == 0:
                confidence = torch.ones(num_requests, gamma, dtype=dtype)
            survival = torch.cumprod(confidence, dim=1)

            min_verify_len = int(torch.randint(0, gamma + 1, ()).item())
            # 0 is the "resolve to gamma" sentinel; otherwise a concrete cap in
            # [min_verify_len, gamma].
            if torch.rand(()).item() < 0.5:
                max_verify_len = 0
            else:
                max_verify_len = int(
                    torch.randint(min_verify_len, gamma + 1, ()).item()
                )
            survival_eps = float(
                [1e-6, 1e-3, 0.1, 0.5][int(torch.randint(0, 4, ()).item())]
            )
            budget = int(torch.randint(0, num_requests * gamma + 3, ()).item())

            cfg = DSparkScheduleConfig(
                gamma=gamma,
                min_verify_len=min_verify_len,
                max_verify_len=max_verify_len,
                survival_eps=survival_eps,
            )
            reference = schedule_verify_lens_topk_from_survival(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            vanilla = schedule_verify_lens_topk_vanilla(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            self.assertTrue(
                torch.equal(reference, vanilla),
                msg=(
                    f"mismatch on trial {trial}: budget={budget} "
                    f"min={min_verify_len} max={max_verify_len} eps={survival_eps} "
                    f"survival={survival.tolist()} "
                    f"reference={reference.tolist()} vanilla={vanilla.tolist()}"
                ),
            )


class TestVerifyLensComposition(CustomTestCase):
    def test_verify_lens_respects_history_budget(self):
        """Different two_steps_prior_k_survival tensors yield different budgets and
        thus different verify_lens (the production compose: budget from history
        survival, verify_lens from the current sort survival)."""
        low_history = torch.tensor([[0.95, 0.30, 0.05]], dtype=torch.float32)
        high_history = torch.tensor([[0.95, 0.90, 0.85]], dtype=torch.float32)
        sort_survival = torch.tensor([[0.95, 0.90, 0.85]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        sps_table = _cliff_table()
        low_budget = compute_verify_token_budget(
            history_survival_probs=low_history, sps_table=sps_table, cfg=cfg
        )
        high_budget = compute_verify_token_budget(
            history_survival_probs=high_history, sps_table=sps_table, cfg=cfg
        )
        self.assertNotEqual(
            low_budget, high_budget, "budgets must differ for this test"
        )
        lens_low = schedule_verify_lens_topk_from_survival(
            survival_probs=sort_survival, budget=low_budget, cfg=cfg
        )
        lens_high = schedule_verify_lens_topk_from_survival(
            survival_probs=sort_survival, budget=high_budget, cfg=cfg
        )
        self.assertFalse(
            torch.equal(lens_low, lens_high),
            "different two_steps_prior_k_survival budgets must produce different verify_lens",
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
