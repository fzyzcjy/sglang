import functools
import unittest
from typing import Callable

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

_CUDA_AVAILABLE = torch.cuda.is_available()

_RESIDUAL_ATOL = 1e-5
_RESIDUAL_RTOL = 1e-5
_RATE_SAMPLES = 20000
_RATE_ATOL = 0.02


def _requires_cuda(test_method: Callable) -> Callable:
    """Decorator: skip the test when CUDA is unavailable (kernel is Triton/GPU)."""

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if not _CUDA_AVAILABLE:
            self.skipTest("CUDA not available; chain rejection kernel is GPU-only.")
        return test_method(self, *args, **kwargs)

    return wrapper


def _residual_reference(
    target_probs_row: torch.Tensor, draft_probs_row: torch.Tensor
) -> torch.Tensor:
    """Closed-form (p-q)+ normalized residual oracle (mirrors sample_residual)."""
    residual = torch.clamp(target_probs_row - draft_probs_row, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    if torch.any(residual_mass <= 1e-8):
        residual = torch.where(residual_mass <= 1e-8, target_probs_row, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    return residual / residual_mass.clamp_min(1e-8)


class _RejectionHarness:
    """Driver for chain_speculative_sampling_triton with controlled coins.

    Builds the exact tensor layout the kernel consumes (see
    reject_sampling.py: chain_speculative_sampling_triton) and exposes a small
    API to assert closed-form acceptance and residual behavior. No coin stream
    is shared with any reference: the harness FEEDS the coins, so verification
    is analytic per-token accounting, not seed-matched bit-equality.
    """

    def __init__(
        self,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        candidates: torch.Tensor,
        uniform_samples: torch.Tensor,
        uniform_samples_final: torch.Tensor,
    ) -> None:
        self.device = target_probs.device
        self.batch_size, self.num_slots, self.vocab_size = target_probs.shape
        self.gamma = self.num_slots - 1
        self.target_probs = target_probs
        self.draft_probs = draft_probs
        self.candidates = candidates
        self.uniform_samples = uniform_samples
        self.uniform_samples_final = uniform_samples_final

    def run(self) -> dict:
        from sglang.srt.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        bs = self.batch_size
        num_slots = self.num_slots
        device = self.device

        # predicts is a flat global buffer; retrive_index maps (batch, slot) ->
        # a global flat slot. Use one contiguous block of num_slots per request.
        predicts = torch.full((bs * num_slots,), -1, dtype=torch.int64, device=device)
        accept_index = torch.full((bs, num_slots), -1, dtype=torch.int64, device=device)
        accept_token_num = torch.zeros(bs, dtype=torch.int64, device=device)
        retrive_index = torch.arange(
            bs * num_slots, dtype=torch.int64, device=device
        ).reshape(bs, num_slots)
        # retrive_next_token / retrive_next_sibling are unused on the chain path.
        retrive_next_token = torch.full(
            (bs, num_slots), -1, dtype=torch.int64, device=device
        )
        retrive_next_sibling = torch.full(
            (bs, num_slots), -1, dtype=torch.int64, device=device
        )

        chain_speculative_sampling_triton(
            predicts,
            accept_index,
            accept_token_num,
            self.candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            self.uniform_samples,
            self.uniform_samples_final,
            self.target_probs,
            self.draft_probs,
            None,
            None,
            False,
        )
        return {
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
            "retrive_index": retrive_index,
        }

    def expected_num_accept(self) -> torch.Tensor:
        """Closed-form per-request accept count from coin*q < p, first-fail-stops."""
        counts = torch.zeros(self.batch_size, dtype=torch.int64)
        for i in range(self.batch_size):
            accepted = 0
            for k in range(self.gamma):
                token = int(self.candidates[i, k + 1].item())
                p = float(self.target_probs[i, k, token].item())
                q = float(self.draft_probs[i, k, token].item())
                coin = float(self.uniform_samples[i, k].item())
                if coin * q < p:
                    accepted += 1
                else:
                    break
            counts[i] = accepted
        return counts


def _make_uniform_probs(
    bs: int, rows: int, vocab: int, device: torch.device, seed: int
) -> torch.Tensor:
    """Random row-normalized probability tensor [bs, rows, vocab] (fp32)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(bs, rows, vocab, generator=g)
    return torch.softmax(logits, dim=-1).to(device=device, dtype=torch.float32)


def _apply_top_k_top_p(probs: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k then top-p truncation per row and renormalize (fp32)."""
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    keep = torch.zeros_like(sorted_probs, dtype=torch.bool)
    keep[..., :top_k] = True
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = keep & (cumulative - sorted_probs < top_p)
    keep[..., 0] = True
    masked_sorted = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    masked = torch.zeros_like(probs).scatter_(-1, sorted_idx, masked_sorted)
    return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class TestDSparkRejectionAcceptRule(CustomTestCase):
    """Invariant 1: the chain kernel accepts iff coin*q < p, first-fail stops.

    These tests REPLACE the hollow server-tier stub
    `test_temperature_sampling_lossless_with_top_k_top_p` (which only re-ran one
    prompt and asserted no-crash). The kernel does not share a coin stream with
    any reference, so verification is analytic: we feed controlled coins and
    assert the closed-form acceptance count / index, not seed-matched equality.
    """

    @_requires_cuda
    def test_all_accept_when_coins_force_accept(self):
        """All gamma drafts accepted when every coin*q < p (coins forced to 0)."""
        device = torch.device("cuda")
        bs, gamma, vocab = 3, 5, 64
        target_probs = _make_uniform_probs(bs, gamma + 1, vocab, device, seed=1)
        draft_probs = _make_uniform_probs(bs, gamma, vocab, device, seed=2)
        candidates = torch.randint(
            0, vocab, (bs, gamma + 1), dtype=torch.int64, device=device
        )
        uniform_samples = torch.zeros(bs, gamma, dtype=torch.float32, device=device)
        uniform_final = torch.rand(bs, dtype=torch.float32, device=device)

        harness = _RejectionHarness(
            target_probs, draft_probs, candidates, uniform_samples, uniform_final
        )
        result = harness.run()
        self.assertTrue(
            torch.equal(
                result["accept_token_num"].cpu(),
                torch.full((bs,), gamma, dtype=torch.int64),
            ),
            "Forcing coin=0 must accept all gamma drafts.",
        )

    @_requires_cuda
    def test_all_reject_when_coins_force_reject(self):
        """Zero drafts accepted when coin=1 and q>p makes coin*q >= p at slot 0."""
        device = torch.device("cuda")
        bs, gamma, vocab = 2, 4, 32
        # Make draft mass concentrate where target mass is low: q >> p on the
        # candidate token so coin (≈1) * q >= p forces rejection at slot 0.
        target_probs = torch.full(
            (bs, gamma + 1, vocab), 1.0 / vocab, dtype=torch.float32, device=device
        )
        draft_probs = torch.full(
            (bs, gamma, vocab), 1.0 / vocab, dtype=torch.float32, device=device
        )
        candidates = torch.zeros(bs, gamma + 1, dtype=torch.int64, device=device)
        # Token 0 gets large q, tiny p -> coin*q >= p.
        draft_probs[:, :, 0] = 0.9
        draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
        target_probs[:, :, 0] = 1e-6
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
        uniform_samples = torch.full(
            (bs, gamma), 1.0 - 1e-6, dtype=torch.float32, device=device
        )
        uniform_final = torch.rand(bs, dtype=torch.float32, device=device)

        harness = _RejectionHarness(
            target_probs, draft_probs, candidates, uniform_samples, uniform_final
        )
        result = harness.run()
        self.assertTrue(
            torch.equal(
                result["accept_token_num"].cpu(),
                torch.zeros(bs, dtype=torch.int64),
            ),
            "Forcing coin*q >= p at slot 0 must accept zero drafts.",
        )

    @_requires_cuda
    def test_accept_count_matches_closed_form(self):
        """Kernel accept_token_num equals analytic coin*q<p first-fail count."""
        device = torch.device("cuda")
        bs, gamma, vocab = 8, 6, 96
        target_probs = _make_uniform_probs(bs, gamma + 1, vocab, device, seed=11)
        draft_probs = _make_uniform_probs(bs, gamma, vocab, device, seed=12)
        candidates = torch.randint(
            0, vocab, (bs, gamma + 1), dtype=torch.int64, device=device
        )
        g = torch.Generator(device="cpu").manual_seed(13)
        uniform_samples = torch.rand(bs, gamma, generator=g).to(
            device=device, dtype=torch.float32
        )
        uniform_final = torch.rand(bs, generator=g).to(
            device=device, dtype=torch.float32
        )

        harness = _RejectionHarness(
            target_probs, draft_probs, candidates, uniform_samples, uniform_final
        )
        result = harness.run()
        expected = harness.expected_num_accept()
        self.assertTrue(
            torch.equal(result["accept_token_num"].cpu(), expected),
            f"Accept count mismatch: kernel={result['accept_token_num'].cpu()} "
            f"expected={expected}",
        )

    @_requires_cuda
    def test_accept_index_matches_retrive_index_prefix(self):
        """Accepted slots' accept_index equal the retrive_index of accepted prefix."""
        device = torch.device("cuda")
        bs, gamma, vocab = 4, 5, 48
        target_probs = _make_uniform_probs(bs, gamma + 1, vocab, device, seed=21)
        draft_probs = _make_uniform_probs(bs, gamma, vocab, device, seed=22)
        candidates = torch.randint(
            0, vocab, (bs, gamma + 1), dtype=torch.int64, device=device
        )
        g = torch.Generator(device="cpu").manual_seed(23)
        uniform_samples = torch.rand(bs, gamma, generator=g).to(
            device=device, dtype=torch.float32
        )
        uniform_final = torch.rand(bs, generator=g).to(
            device=device, dtype=torch.float32
        )

        harness = _RejectionHarness(
            target_probs, draft_probs, candidates, uniform_samples, uniform_final
        )
        result = harness.run()
        expected = harness.expected_num_accept()
        accept_index = result["accept_index"].cpu()
        retrive_index = result["retrive_index"].cpu()
        for i in range(bs):
            num = int(expected[i].item())
            # Root slot 0 always carries retrive_index[i, 0]; accepted prefix
            # 1..num carry retrive_index[i, 1..num].
            for slot in range(num + 1):
                self.assertEqual(
                    int(accept_index[i, slot].item()),
                    int(retrive_index[i, slot].item()),
                    f"req {i} slot {slot}: accept_index must equal retrive_index.",
                )


class TestDSparkRejectionResidual(CustomTestCase):
    """Invariant 1: rejected-slot final token follows (p-q)+ normalized residual."""

    @_requires_cuda
    def test_residual_distribution_matches_oracle(self):
        """Empirical final-token histogram at the rejected slot matches (p-q)+."""
        device = torch.device("cuda")
        bs, gamma, vocab = 1, 1, 16
        # Single request, single draft slot; force rejection at slot 0 so the
        # final token is drawn from the residual at row cur_prob_row == 0.
        target_probs = _make_uniform_probs(1, gamma + 1, vocab, device, seed=31)
        draft_probs = _make_uniform_probs(1, gamma, vocab, device, seed=32)
        # Candidate token chosen so coin (≈1) * q >= p -> reject at slot 0.
        cand_token = int(torch.argmax(draft_probs[0, 0]).item())
        candidates = torch.zeros(1, gamma + 1, dtype=torch.int64, device=device)
        candidates[0, 1] = cand_token
        # Ensure rejection: q(cand) large relative to p(cand).
        target_probs[0, 0, cand_token] = 1e-6
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)

        oracle = _residual_reference(target_probs[0, 0:1], draft_probs[0, 0:1]).squeeze(
            0
        )

        counts = torch.zeros(vocab, dtype=torch.float64)
        n = 4000
        g = torch.Generator(device="cpu").manual_seed(99)
        for _ in range(n):
            batch_target = target_probs.clone()
            batch_draft = draft_probs.clone()
            uniform_samples = torch.full(
                (1, gamma), 1.0 - 1e-6, dtype=torch.float32, device=device
            )
            uniform_final = torch.rand(1, generator=g).to(
                device=device, dtype=torch.float32
            )
            harness = _RejectionHarness(
                batch_target,
                batch_draft,
                candidates,
                uniform_samples,
                uniform_final,
            )
            result = harness.run()
            # On rejection at slot 0, final token is stored at predicts[
            # retrive_index[0, 0]] (last_accepted_global_idx == root).
            root_global = int(result["retrive_index"][0, 0].item())
            final_token = int(result["predicts"][root_global].item())
            self.assertGreaterEqual(final_token, 0)
            counts[final_token] += 1

        empirical = counts / counts.sum()
        oracle_cpu = oracle.double().cpu()
        # Distribution match within sampling tolerance (chi-square-ish L1 bound).
        l1 = float((empirical - oracle_cpu).abs().sum().item())
        self.assertLess(
            l1,
            0.12,
            f"Residual histogram L1 {l1:.4f} too far from (p-q)+ oracle.",
        )


class TestDSparkRejectionTopKTopP(CustomTestCase):
    """Invariant 1: losslessness holds with target top-k/top-p masking applied."""

    @_requires_cuda
    def test_accept_count_with_target_top_k_top_p(self):
        """Top-k/top-p masked target_probs still gives closed-form accept count."""
        device = torch.device("cuda")
        bs, gamma, vocab = 6, 5, 128
        target_probs = _make_uniform_probs(bs, gamma + 1, vocab, device, seed=41)
        target_probs = _apply_top_k_top_p(target_probs, top_k=20, top_p=0.9)
        draft_probs = _make_uniform_probs(bs, gamma, vocab, device, seed=42)
        # Candidates must lie in the surviving support so p(cand) > 0 is possible.
        candidates = torch.zeros(bs, gamma + 1, dtype=torch.int64, device=device)
        for i in range(bs):
            for s in range(gamma + 1):
                row = min(s, gamma)
                candidates[i, s] = int(torch.argmax(target_probs[i, row]).item())
        g = torch.Generator(device="cpu").manual_seed(43)
        uniform_samples = torch.rand(bs, gamma, generator=g).to(
            device=device, dtype=torch.float32
        )
        uniform_final = torch.rand(bs, generator=g).to(
            device=device, dtype=torch.float32
        )

        harness = _RejectionHarness(
            target_probs, draft_probs, candidates, uniform_samples, uniform_final
        )
        result = harness.run()
        expected = harness.expected_num_accept()
        self.assertTrue(
            torch.equal(result["accept_token_num"].cpu(), expected),
            "Top-k/top-p target masking must not break the accept rule.",
        )


class TestDSparkRejectionEdgeCases(CustomTestCase):
    """Invariant 1 edge cases: zero draft positions (verify_len == 1)."""

    @_requires_cuda
    def test_zero_draft_positions_samples_target(self):
        """gamma==0 (verify_len 1) accepts nothing and the kernel samples target.

        Mirrors the ragged edge case ell_r == 0: candidates carries only the
        anchor, accept_token_num == 0, and the all-accepted branch samples pure
        target p. infra guarantees min(verify_lens) >= 1, but the harness must
        cover this slot explicitly.
        """
        device = torch.device("cuda")
        bs, gamma, vocab = 3, 0, 32
        num_slots = gamma + 1  # 1
        target_probs = _make_uniform_probs(bs, num_slots, vocab, device, seed=51)
        # draft_probs has gamma == 0 rows; allocate a 0-row tensor.
        draft_probs = torch.empty(bs, 0, vocab, dtype=torch.float32, device=device)
        candidates = torch.randint(
            0, vocab, (bs, num_slots), dtype=torch.int64, device=device
        )
        uniform_samples = torch.empty(bs, 0, dtype=torch.float32, device=device)
        uniform_final = torch.rand(bs, dtype=torch.float32, device=device)

        harness = _RejectionHarness(
            target_probs, draft_probs, candidates, uniform_samples, uniform_final
        )
        result = harness.run()
        self.assertTrue(
            torch.equal(
                result["accept_token_num"].cpu(),
                torch.zeros(bs, dtype=torch.int64),
            ),
            "With zero draft positions no draft can be accepted.",
        )
        # Each request stores a final token at its root slot.
        retrive_index = result["retrive_index"].cpu()
        predicts = result["predicts"].cpu()
        for i in range(bs):
            root_global = int(retrive_index[i, 0].item())
            self.assertGreaterEqual(
                int(predicts[root_global].item()),
                0,
                "Final target token must be stored at the root slot.",
            )


if __name__ == "__main__":
    unittest.main()
