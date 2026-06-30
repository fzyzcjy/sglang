import unittest

import torch

from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")


def _make_worker(
    *,
    gamma: int,
    mode: RaggedVerifyMode = RaggedVerifyMode.CAP_ACCEPT,
    scheduler: object = object(),
) -> DSparkWorkerV2:
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.gamma = gamma
    worker.verify_num_draft_tokens = gamma + 1
    worker._ragged_verify_mode = mode
    worker._verify_scheduler = scheduler
    worker._confidence_ready = None
    worker._confidence_cpu_pinned = None
    return worker


def _cutoff_layout(verify_lens_cpu: list[int]) -> RaggedVerifyLayout:
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens_cpu,
        device=_DEVICE,
        grid=[sum(verify_lens_cpu)],
    )


class TestCutoffCapMath(CustomTestCase):
    def test_cap_is_elementwise_min_with_ell_r(self):
        """_cap_correct_len caps each request's accept count at ell_r = verify_len - 1."""
        worker = _make_worker(gamma=4)
        layout = _cutoff_layout([5, 2, 4])
        correct_len = torch.tensor([4, 3, 1], dtype=torch.int32)
        capped = worker._cap_correct_len(correct_len=correct_len, layout=layout)
        self.assertEqual(capped.tolist(), [4, 1, 1])

    def test_cap_no_op_when_already_below_ell_r(self):
        """Capping leaves accept counts that are already <= ell_r unchanged."""
        worker = _make_worker(gamma=4)
        layout = _cutoff_layout([5, 5, 5])
        correct_len = torch.tensor([0, 2, 4], dtype=torch.int32)
        capped = worker._cap_correct_len(correct_len=correct_len, layout=layout)
        self.assertEqual(capped.tolist(), [0, 2, 4])


class TestCutoffGreedyAccept(CustomTestCase):
    def _greedy_logits(self, target_predict: torch.Tensor, vocab: int) -> torch.Tensor:
        bs, window = target_predict.shape
        logits = torch.zeros((bs * window, vocab), dtype=torch.float32)
        flat = target_predict.reshape(-1)
        logits[torch.arange(bs * window), flat] = 1.0
        return logits

    def test_full_verify_len_is_byte_identical_to_no_cutoff(self):
        """A cutoff layout with verify_len == gamma+1 is byte-identical to no cutoff."""
        gamma = 4
        worker = _make_worker(gamma=gamma)
        vocab = 128
        candidates = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        target_predict = torch.tensor([[2, 3, 99, 50, 60], [7, 8, 9, 10, 11]])
        logits = self._greedy_logits(target_predict, vocab)

        base_len, base_bonus = worker._accept_greedy(
            candidates=candidates, target_logits=logits
        )
        layout = _cutoff_layout([gamma + 1, gamma + 1])
        cut_len, cut_bonus = worker._accept_greedy(
            candidates=candidates, target_logits=logits, cutoff_layout=layout
        )
        self.assertEqual(base_len.tolist(), cut_len.tolist())
        self.assertEqual(base_bonus.tolist(), cut_bonus.tolist())

    def test_truncation_caps_accept_and_bonus_is_target_at_cap(self):
        """Capping accept yields the target's true next token at the capped index (lossless)."""
        gamma = 4
        worker = _make_worker(gamma=gamma)
        vocab = 64
        # Request 0 would accept 3 drafts uncapped (candidates[:,1:]==target_predict[:,:-1]).
        candidates = torch.tensor([[1, 2, 3, 4, 5]])
        target_predict = torch.tensor([[2, 3, 4, 41, 42]])
        logits = self._greedy_logits(target_predict, vocab)

        base_len, _ = worker._accept_greedy(candidates=candidates, target_logits=logits)
        self.assertEqual(base_len.tolist(), [3])

        layout = _cutoff_layout([3])  # ell_r = 2
        cut_len, cut_bonus = worker._accept_greedy(
            candidates=candidates, target_logits=logits, cutoff_layout=layout
        )
        self.assertEqual(cut_len.tolist(), [2])
        # Lossless: the bonus is the target argmax at the capped position (index 2).
        self.assertEqual(cut_bonus.tolist(), [int(target_predict[0, 2])])

    def test_anchor_only_cap_commits_only_bonus(self):
        """ell_r = 0 (verify_len 1) caps accept to 0 and the bonus is the target's first token."""
        gamma = 4
        worker = _make_worker(gamma=gamma)
        vocab = 64
        candidates = torch.tensor([[1, 2, 3, 4, 5]])
        target_predict = torch.tensor([[2, 3, 4, 5, 6]])
        logits = self._greedy_logits(target_predict, vocab)

        layout = _cutoff_layout([1])  # ell_r = 0
        cut_len, cut_bonus = worker._accept_greedy(
            candidates=candidates, target_logits=logits, cutoff_layout=layout
        )
        self.assertEqual(cut_len.tolist(), [0])
        self.assertEqual(cut_bonus.tolist(), [int(target_predict[0, 0])])


class TestRaggedGating(CustomTestCase):
    def test_off_mode_returns_none(self):
        """OFF mode never schedules a ragged layout (uniform path stays byte-identical)."""
        worker = _make_worker(gamma=4, mode=RaggedVerifyMode.STATIC, scheduler=object())
        layout = worker._maybe_schedule_ragged_layout(
            req_pool_indices=torch.tensor([0, 1]), device=_DEVICE
        )
        self.assertIsNone(layout)

    def test_full_mode_without_confidence_returns_none(self):
        """FULL with no confidence history yet falls back to the uniform block (lossless)."""
        worker = _make_worker(gamma=4, mode=RaggedVerifyMode.COMPACT, scheduler=object())
        layout = worker._maybe_schedule_ragged_layout(
            req_pool_indices=torch.tensor([0, 1]), device=_DEVICE
        )
        self.assertIsNone(layout)

    def test_cutoff_mode_without_scheduler_returns_none(self):
        """cap-accept with no scheduler (no confidence head) falls back to uniform."""
        worker = _make_worker(
            gamma=4, mode=RaggedVerifyMode.CAP_ACCEPT, scheduler=None
        )
        layout = worker._maybe_schedule_ragged_layout(
            req_pool_indices=torch.tensor([0, 1]), device=_DEVICE
        )
        self.assertIsNone(layout)


if __name__ == "__main__":
    unittest.main()
