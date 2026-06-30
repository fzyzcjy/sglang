import types
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_verify_planner import (
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
    _DraftBlockResult,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")


def _make_worker(*, gamma: int, mode: RaggedVerifyMode = RaggedVerifyMode.COMPACT):
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.gamma = gamma
    worker.verify_num_draft_tokens = gamma + 1
    worker.device = _DEVICE
    planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
    planner.gamma = gamma
    planner.verify_num_draft_tokens = gamma + 1
    planner._ragged_verify_mode = mode
    worker._verify_planner = planner
    return worker


def _full_layout(verify_lens_cpu, grid=None):
    grid = grid or [sum(verify_lens_cpu)]
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens_cpu,
        device=_DEVICE,
        grid=grid,
    )


class TestCompactVerifyIds(CustomTestCase):
    def test_packs_anchor_then_scheduled_prefix_per_request(self):
        """Compact verify ids pack [anchor, s_0..s_{ell_r-1}] back to back per request."""
        worker = _make_worker(gamma=4)
        # anchor = draft_block_ids[:, 0]; drafts s_0..s_3 = draft_tokens[r].
        draft_block_ids = torch.tensor([[10, 0, 0, 0, 0], [20, 0, 0, 0, 0]])
        draft_tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
        verify_lens_cpu = [3, 1]  # req0 verifies anchor + s_0,s_1; req1 anchor only
        verify_ids = worker._compact_verify_ids(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            verify_lens_cpu=verify_lens_cpu,
            total=sum(verify_lens_cpu),
            device=_DEVICE,
        )
        self.assertEqual(verify_ids.tolist(), [10, 11, 12, 20])

    def test_anchor_only_request_emits_only_anchor(self):
        """A verify_len == 1 request contributes only its anchor token."""
        worker = _make_worker(gamma=4)
        draft_block_ids = torch.tensor([[7, 0, 0, 0, 0]])
        draft_tokens = torch.tensor([[1, 2, 3, 4]])
        verify_ids = worker._compact_verify_ids(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            verify_lens_cpu=[1],
            total=1,
            device=_DEVICE,
        )
        self.assertEqual(verify_ids.tolist(), [7])


class TestCompactToStridedScatter(CustomTestCase):
    def test_scatter_places_each_request_at_its_stride_offset(self):
        """Compact rows land at i*(gamma+1)+k; padded strided rows take the fill value."""
        worker = _make_worker(gamma=4)
        layout = _full_layout([3, 1])  # total = 4
        compact = torch.tensor(
            [[1.0], [2.0], [3.0], [9.0]], dtype=torch.float32
        )  # req0: rows 0..2, req1: row 3
        strided = worker._scatter_compact_to_strided(
            compact=compact, layout=layout, bs=2, fill_value=-1.0
        )
        self.assertEqual(strided.shape, (2 * 5, 1))
        # req0 occupies strided rows 0,1,2 (then 3,4 padded); req1 occupies row 5.
        self.assertEqual(strided[:3, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(strided[3:5, 0].tolist(), [-1.0, -1.0])
        self.assertEqual(strided[5, 0].item(), 9.0)
        self.assertEqual(strided[6:10, 0].tolist(), [-1.0, -1.0, -1.0, -1.0])


class TestResolveGreedyMask(CustomTestCase):
    def test_mask_is_per_request_from_top_k(self):
        """A row is greedy iff its top_k <= 1 (mirrors is_all_greedy per request)."""
        worker = _make_worker(gamma=4)
        sampling_info = types.SimpleNamespace(
            top_ks=torch.tensor([1, 5, 1, 50], dtype=torch.int32)
        )
        mask = worker._resolve_greedy_mask(bs=4, sampling_info=sampling_info)
        self.assertEqual(mask.tolist(), [True, False, True, False])

    def test_none_sampling_info_is_all_greedy(self):
        """No sampling info means the whole batch is greedy."""
        worker = _make_worker(gamma=4)
        mask = worker._resolve_greedy_mask(bs=3, sampling_info=None)
        self.assertEqual(mask.tolist(), [True, True, True])


class TestAcceptBlockPerRequest(CustomTestCase):
    def _greedy_logits(self, target_predict, vocab):
        bs, window = target_predict.shape
        logits = torch.zeros((bs * window, vocab), dtype=torch.float32)
        logits[torch.arange(bs * window), target_predict.reshape(-1)] = 1.0
        return logits

    def test_all_greedy_uses_argmax_match_path(self):
        """An all-greedy mask routes _accept_draft_tokens to the argmax-match (DFlash) rule."""
        gamma = 4
        worker = _make_worker(gamma=gamma)
        vocab = 64
        candidates = torch.tensor([[1, 2, 3, 4, 5]])
        target_predict = torch.tensor([[2, 3, 4, 41, 42]])
        logits = self._greedy_logits(target_predict, vocab)
        draft_block = _DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=torch.zeros((1, gamma, vocab)),
            greedy_mask=torch.tensor([True]),
            temperatures=torch.ones(1),
        )
        correct_len, bonus = worker._accept_draft_tokens(
            candidates=candidates,
            target_logits=logits,
            draft_block=draft_block,
            sampling_info=None,
            draft_input=object(),
        )
        self.assertEqual(correct_len.tolist(), [3])
        self.assertEqual(bonus.tolist(), [41])

    def test_all_sampling_uses_only_chain_kernel_path(self):
        """An all-sampling mask routes _accept_draft_tokens to _accept_sampling exactly once (no greedy call)."""
        gamma = 4
        worker = _make_worker(gamma=gamma)
        bs = 2
        candidates = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        target_logits = torch.zeros((bs * (gamma + 1), 8))
        draft_block = _DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=torch.zeros((bs, gamma, 8)),
            greedy_mask=torch.tensor([False, False]),
            temperatures=torch.ones(bs),
        )
        calls = {"greedy": 0, "sampling": 0}

        def fake_greedy(**kwargs):
            calls["greedy"] += 1
            return torch.zeros(bs, dtype=torch.int32), torch.zeros(
                bs, dtype=torch.int64
            )

        def fake_sampling(**kwargs):
            calls["sampling"] += 1
            return (
                torch.tensor([1, 2], dtype=torch.int32),
                torch.tensor([3, 4], dtype=torch.int64),
            )

        worker._accept_greedy = fake_greedy
        worker._accept_sampling = fake_sampling
        correct_len, bonus = worker._accept_draft_tokens(
            candidates=candidates,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=object(),
            draft_input=object(),
        )
        self.assertEqual(calls, {"greedy": 0, "sampling": 1})
        self.assertEqual(correct_len.tolist(), [1, 2])
        self.assertEqual(bonus.tolist(), [3, 4])

    def test_mixed_batch_selects_per_row_results(self):
        """A mixed batch selects greedy rows from argmax-match and sampling rows from the chain kernel."""
        gamma = 4
        worker = _make_worker(gamma=gamma)
        bs = 2
        candidates = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        target_logits = torch.zeros((bs * (gamma + 1), 8))
        draft_block = _DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=torch.zeros((bs, gamma, 8)),
            greedy_mask=torch.tensor([True, False]),
            temperatures=torch.ones(bs),
        )

        captured = {}

        def fake_greedy(*, candidates, target_logits, cutoff_layout=None):
            return (
                torch.tensor([3, 99], dtype=torch.int32),
                torch.tensor([10, 11], dtype=torch.int64),
            )

        def fake_sampling(
            *,
            candidates,
            target_logits,
            draft_probs,
            sampling_info,
            draft_input,
            cutoff_layout=None,
        ):
            captured["draft_probs_shape"] = tuple(draft_probs.shape)
            return (
                torch.tensor([88, 2], dtype=torch.int32),
                torch.tensor([20, 21], dtype=torch.int64),
            )

        worker._accept_greedy = fake_greedy
        worker._accept_sampling = fake_sampling

        correct_len, bonus = worker._accept_draft_tokens(
            candidates=candidates,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=object(),
            draft_input=object(),
        )
        # Row 0 (greedy) takes the greedy result; row 1 (sampling) the chain result.
        self.assertEqual(correct_len.tolist(), [3, 2])
        self.assertEqual(bonus.tolist(), [10, 21])
        self.assertEqual(captured["draft_probs_shape"], (bs, gamma, 8))


class TestVerifyLayoutGrid(CustomTestCase):
    def test_cutoff_grid_is_total_only(self):
        """cap-accept never selects a token-keyed graph, so the grid is just the total."""
        worker = _make_worker(gamma=4, mode=RaggedVerifyMode.CAP_ACCEPT)
        grid = worker._verify_layout_grid(verify_lens_cpu=[5, 2, 1])
        self.assertEqual(grid, [8])

    def test_full_grid_uses_runner_token_buckets(self):
        """full aligns the layout grid with the decode runner's token-keyed capture buckets."""
        worker = _make_worker(gamma=4, mode=RaggedVerifyMode.COMPACT)
        runner = types.SimpleNamespace(
            ragged_verify_mode=True,
            capture_num_tokens=[4, 8, 16, 32],
        )
        worker.model_runner = types.SimpleNamespace(decode_cuda_graph_runner=runner)
        grid = worker._verify_layout_grid(verify_lens_cpu=[5, 2, 1])
        self.assertEqual(grid, [4, 8, 16, 32])
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[5, 2, 1], device=_DEVICE, grid=grid
        )
        self.assertEqual(layout.total_verify_tokens, 8)
        self.assertEqual(layout.graph_num_tokens, 8)

    def test_full_grid_falls_back_to_total_without_token_graph(self):
        """full with no token-keyed graph (eager / cuda-graph off) runs on exactly total."""
        worker = _make_worker(gamma=4, mode=RaggedVerifyMode.COMPACT)
        worker.model_runner = types.SimpleNamespace(decode_cuda_graph_runner=None)
        grid = worker._verify_layout_grid(verify_lens_cpu=[5, 2, 1])
        self.assertEqual(grid, [8])

    def test_floor_lifts_bucket_when_token_graph_present(self):
        """The bs-derived floor only applies when a token-keyed graph exists."""
        worker = _make_worker(gamma=4, mode=RaggedVerifyMode.COMPACT)
        runner = types.SimpleNamespace(
            ragged_verify_mode=True,
            capture_num_tokens=[5, 10, 15],
        )
        worker.model_runner = types.SimpleNamespace(decode_cuda_graph_runner=runner)
        # 3 requests, gamma+1 = 5, so the floor is 15; a tiny total still keys the
        # 15-token tier (3 capture slots), not the smaller 5-token tier (1 slot).
        floor = worker._verify_layout_graph_num_tokens_floor(num_reqs=3)
        self.assertEqual(floor, 15)
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1, 1, 1],
            device=_DEVICE,
            grid=runner.capture_num_tokens,
            graph_num_tokens_floor=floor,
        )
        self.assertEqual(layout.total_verify_tokens, 3)
        self.assertEqual(layout.graph_num_tokens, 15)

    def test_floor_is_zero_without_token_graph(self):
        """No token-keyed graph means no bs-derived floor (eager runs on exact total)."""
        worker = _make_worker(gamma=4, mode=RaggedVerifyMode.COMPACT)
        worker.model_runner = types.SimpleNamespace(decode_cuda_graph_runner=None)
        self.assertEqual(worker._verify_layout_graph_num_tokens_floor(num_reqs=3), 0)


class TestLayoutLessCompactCarriesDegenerateLayout(CustomTestCase):
    def test_uniform_ragged_layout_matches_static_full_block(self):
        """The C3 fallback layout is a uniform gamma+1 block keyed token-wise."""
        worker = _make_worker(gamma=4, mode=RaggedVerifyMode.COMPACT)
        runner = types.SimpleNamespace(
            ragged_verify_mode=True,
            capture_num_tokens=[5, 10, 15],
        )
        worker.model_runner = types.SimpleNamespace(decode_cuda_graph_runner=runner)
        layout = worker._uniform_ragged_layout(bs=2, device=_DEVICE)
        self.assertEqual(layout.verify_lens_cpu, [5, 5])
        self.assertEqual(layout.total_verify_tokens, 10)
        self.assertEqual(layout.graph_num_tokens, 10)


class TestRaggedLayoutSkippedWhenBatchExceedsCapturedGrid(CustomTestCase):
    def _compact_worker(self, *, gamma: int, capture_num_tokens: list[int]):
        worker = _make_worker(gamma=gamma, mode=RaggedVerifyMode.COMPACT)
        runner = types.SimpleNamespace(
            ragged_verify_mode=True,
            capture_num_tokens=capture_num_tokens,
        )
        worker.model_runner = types.SimpleNamespace(decode_cuda_graph_runner=runner)
        worker._verify_planner.model_runner = worker.model_runner
        return worker

    def test_uniform_layout_skipped_when_bs_exceeds_max_capture_bs(self):
        """A layout-less compact verify with bs > max_capture_bs returns None, not a raise."""
        # capture max tier 15 -> max_capture_bs = 15 // 5 = 3; bs=4 floors to 20 > 15.
        worker = self._compact_worker(gamma=4, capture_num_tokens=[5, 10, 15])
        self.assertIsNone(worker._uniform_ragged_layout(bs=4, device=_DEVICE))

    def test_uniform_layout_built_when_bs_at_max_capture_bs(self):
        """bs == max_capture_bs still builds the layout (the guard is inert when it fits)."""
        worker = self._compact_worker(gamma=4, capture_num_tokens=[5, 10, 15])
        layout = worker._uniform_ragged_layout(bs=3, device=_DEVICE)
        self.assertIsNotNone(layout)
        self.assertEqual(layout.total_verify_tokens, 15)

    def test_confidence_ready_layout_skipped_when_num_reqs_exceeds_max_capture_bs(self):
        """A confidence-ready compact batch with num_reqs > max_capture_bs returns None, not a raise."""
        worker = self._compact_worker(gamma=4, capture_num_tokens=[5, 10, 15])
        worker._verify_planner._schedule_verify_lens = (
            lambda *, req_pool_indices, prefix_lens, device: torch.tensor(
                [5, 4, 5, 3], dtype=torch.int32, device=device
            )
        )
        req_pool_indices = torch.arange(4, device=_DEVICE)
        self.assertIsNone(
            worker._verify_planner.schedule_layout(
                req_pool_indices=req_pool_indices,
                prefix_lens=torch.full((4,), 8, device=_DEVICE),
                device=_DEVICE,
            )
        )


if __name__ == "__main__":
    unittest.main()
