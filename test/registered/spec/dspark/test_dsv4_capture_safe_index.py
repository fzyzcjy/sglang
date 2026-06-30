import random
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
from sglang.srt.layers.attention.deepseek_v4_backend import (
    PAGE_INDEX_ALIGNED_SIZE,
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    _compact_dspark_window_then_block,
    build_dspark_swa_page_indices,
)
from sglang.srt.utils import ceil_align
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _reference_compact_with_boolean_mask(
    *,
    window_swa_locs: torch.Tensor,
    block_swa_locs: torch.Tensor,
    context_lens: torch.Tensor,
    target_width: int,
    block_size: int,
) -> torch.Tensor:
    """The original boolean-mask left-pack, kept here only as the equivalence oracle."""
    bs = window_swa_locs.shape[0]
    device = window_swa_locs.device
    out = torch.full((bs, target_width), -1, dtype=torch.int32, device=device)

    col = torch.arange(SWA_WINDOW, device=device, dtype=torch.int32).view(1, -1)
    valid_window = col >= (SWA_WINDOW - context_lens.view(-1, 1))
    packed_window_col = col - (SWA_WINDOW - context_lens.view(-1, 1))
    rows = torch.arange(bs, device=device).view(-1, 1).expand(-1, SWA_WINDOW)
    out[rows[valid_window], packed_window_col[valid_window]] = window_swa_locs[
        valid_window
    ]

    block_col = context_lens.view(-1, 1) + torch.arange(
        block_size, device=device, dtype=torch.int32
    ).view(1, -1)
    block_rows = torch.arange(bs, device=device).view(-1, 1).expand(-1, block_size)
    out[block_rows, block_col] = block_swa_locs
    return out


def _make_window_block(
    *,
    context_lens: list[int],
    block_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs = len(context_lens)
    window = torch.full((bs, SWA_WINDOW), -1, dtype=torch.int32)
    for r, cl in enumerate(context_lens):
        for j in range(SWA_WINDOW - cl, SWA_WINDOW):
            window[r, j] = rng.randint(0, 9999)
    block = torch.randint(0, 9999, (bs, block_size), dtype=torch.int32)
    return window, block


class TestCompactWindowThenBlockEquivalence(CustomTestCase):
    def test_matches_boolean_mask_reference_over_random_context_lens(self):
        """The gather+where left-pack is element-wise equal to the boolean-mask oracle."""
        rng = random.Random(0)
        for _ in range(200):
            bs = rng.randint(1, 6)
            block_size = rng.randint(1, 8)
            target_width = ceil_align(SWA_WINDOW + block_size, PAGE_INDEX_ALIGNED_SIZE)
            context_lens = []
            for _r in range(bs):
                roll = rng.random()
                if roll < 0.15:
                    context_lens.append(0)
                elif roll < 0.30:
                    context_lens.append(SWA_WINDOW)
                else:
                    context_lens.append(rng.randint(0, SWA_WINDOW))
            window, block = _make_window_block(
                context_lens=context_lens, block_size=block_size, rng=rng
            )
            context = torch.tensor(context_lens, dtype=torch.int32)
            new_out = _compact_dspark_window_then_block(
                window_swa_locs=window,
                block_swa_locs=block,
                context_lens=context,
                target_width=target_width,
                block_size=block_size,
            )
            ref_out = _reference_compact_with_boolean_mask(
                window_swa_locs=window,
                block_swa_locs=block,
                context_lens=context,
                target_width=target_width,
                block_size=block_size,
            )
            self.assertTrue(torch.equal(new_out, ref_out))

    def test_context_len_zero_packs_only_block_slots(self):
        """context_len=0 yields an all -1 window and the block slots at [0, block_size)."""
        block_size = 4
        target_width = ceil_align(SWA_WINDOW + block_size, PAGE_INDEX_ALIGNED_SIZE)
        window = torch.full((1, SWA_WINDOW), -1, dtype=torch.int32)
        block = torch.tensor([[7, 8, 9, 10]], dtype=torch.int32)
        out = _compact_dspark_window_then_block(
            window_swa_locs=window,
            block_swa_locs=block,
            context_lens=torch.tensor([0], dtype=torch.int32),
            target_width=target_width,
            block_size=block_size,
        )
        self.assertEqual(out[0, :block_size].tolist(), [7, 8, 9, 10])
        self.assertTrue(torch.all(out[0, block_size:] == -1).item())

    def test_context_len_full_window_keeps_whole_window_then_block(self):
        """context_len=SWA_WINDOW keeps the full window then places block at [W, W+block_size)."""
        block_size = 3
        target_width = ceil_align(SWA_WINDOW + block_size, PAGE_INDEX_ALIGNED_SIZE)
        window = torch.arange(SWA_WINDOW, dtype=torch.int32).view(1, SWA_WINDOW)
        block = torch.tensor([[100, 101, 102]], dtype=torch.int32)
        out = _compact_dspark_window_then_block(
            window_swa_locs=window,
            block_swa_locs=block,
            context_lens=torch.tensor([SWA_WINDOW], dtype=torch.int32),
            target_width=target_width,
            block_size=block_size,
        )
        self.assertEqual(out[0, :SWA_WINDOW].tolist(), list(range(SWA_WINDOW)))
        self.assertEqual(
            out[0, SWA_WINDOW : SWA_WINDOW + block_size].tolist(), [100, 101, 102]
        )


def _fake_backend(*, gamma: int, is_dspark_draft: bool) -> DeepseekV4AttnBackend:
    """A DeepseekV4AttnBackend with only the fields make_core_attn_metadata reads."""
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.swa_page_size = SWA_WINDOW
    backend.page_size = 256
    backend.is_dspark_draft = is_dspark_draft
    backend.speculative_num_draft_tokens = gamma + 1
    backend.cuda_int32_kwargs = {"device": torch.device("cpu"), "dtype": torch.int32}
    backend.c4_topk = 1
    num_reqs, max_cols = 8, 512
    backend.req_to_token = (
        torch.arange(num_reqs * max_cols, dtype=torch.int64).view(num_reqs, max_cols)
    )
    backend.token_to_kv_pool = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda x: x
    )
    return backend


class TestMakeCoreAttnMetadataGeometryGating(CustomTestCase):
    def test_dspark_block_size_routes_to_noncausal_builder(self):
        """A non-None dspark_block_size routes the uniform-gamma path to the non-causal index."""
        gamma, bs = 4, 2
        backend = _fake_backend(gamma=gamma, is_dspark_draft=True)
        num_q = bs * gamma
        page_indices = torch.zeros((num_q, SWA_WINDOW), dtype=torch.int32)
        topk = torch.full((num_q,), gamma, dtype=torch.int32)
        backend.get_dspark_swa_page_indices = mock.MagicMock(
            return_value=(page_indices, topk)
        )
        backend.get_swa_page_indices = mock.MagicMock()
        with mock.patch.object(dsv4, "_create_flashmla_metadata", return_value=None):
            backend.make_core_attn_metadata(
                req_to_token=backend.req_to_token,
                req_pool_indices_repeated=torch.zeros(num_q, dtype=torch.int64),
                seq_lens_casual=torch.ones(num_q, dtype=torch.int32),
                max_seq_len=512,
                out_loc=torch.arange(num_q, dtype=torch.int32),
                need_compress=False,
                dspark_block_size=gamma,
            )
        backend.get_dspark_swa_page_indices.assert_called_once()
        self.assertEqual(
            backend.get_dspark_swa_page_indices.call_args.kwargs["block_size"], gamma
        )
        backend.get_swa_page_indices.assert_not_called()

    def test_none_block_size_stays_causal_even_on_dspark_draft(self):
        """The gamma+1 capture geometry (num_q not a gamma multiple) is routed causal, not asserted."""
        gamma, bs = 4, 1
        backend = _fake_backend(gamma=gamma, is_dspark_draft=True)
        num_q = (gamma + 1) * bs
        backend.get_swa_page_indices = mock.MagicMock(
            return_value=torch.zeros((num_q, SWA_WINDOW), dtype=torch.int32)
        )
        backend.get_dspark_swa_page_indices = mock.MagicMock()
        with mock.patch.object(dsv4, "_create_flashmla_metadata", return_value=None):
            backend.make_core_attn_metadata(
                req_to_token=backend.req_to_token,
                req_pool_indices_repeated=torch.zeros(num_q, dtype=torch.int64),
                seq_lens_casual=torch.arange(1, num_q + 1, dtype=torch.int32),
                max_seq_len=512,
                out_loc=torch.arange(num_q, dtype=torch.int32),
                need_compress=False,
                dspark_block_size=None,
            )
        backend.get_swa_page_indices.assert_called_once()
        backend.get_dspark_swa_page_indices.assert_not_called()

    def test_block_size_must_equal_gamma(self):
        """dspark_block_size that is not speculative_num_draft_tokens-1 trips the gamma-lock assert."""
        gamma, bs = 4, 2
        backend = _fake_backend(gamma=gamma, is_dspark_draft=True)
        num_q = bs * gamma
        backend.get_dspark_swa_page_indices = mock.MagicMock()
        with self.assertRaises(AssertionError):
            backend.make_core_attn_metadata(
                req_to_token=backend.req_to_token,
                req_pool_indices_repeated=torch.zeros(num_q, dtype=torch.int64),
                seq_lens_casual=torch.ones(num_q, dtype=torch.int32),
                max_seq_len=512,
                out_loc=torch.arange(num_q, dtype=torch.int32),
                need_compress=False,
                dspark_block_size=gamma + 1,
            )


class TestDsparkSwaPageIndicesGeometry(CustomTestCase):
    def test_uniform_gamma_geometry_builds_replicated_noncausal_rows(self):
        """Uniform-gamma num_q yields per-request topk = context_len + gamma, identical across block rows."""
        gamma, bs = 3, 2
        backend = _fake_backend(gamma=gamma, is_dspark_draft=True)
        prefix_lens = [5, 130]
        seq_lens_casual = torch.tensor(
            [p + 1 for p in prefix_lens for _ in range(gamma)], dtype=torch.int32
        )
        req_pool_indices_repeated = torch.tensor(
            [r for r in range(bs) for _ in range(gamma)], dtype=torch.int32
        )
        out_loc = torch.arange(bs * gamma, dtype=torch.int32) + 100000
        page_indices, topk = backend.get_dspark_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            out_loc=out_loc,
            block_size=gamma,
        )
        self.assertEqual(page_indices.shape[0], bs * gamma)
        for r, prefix in enumerate(prefix_lens):
            expected_topk = min(prefix, SWA_WINDOW) + gamma
            rows = topk[r * gamma : (r + 1) * gamma]
            self.assertTrue(torch.all(rows == expected_topk).item())
            block = page_indices[r * gamma : (r + 1) * gamma]
            self.assertTrue(torch.equal(block[0], block[-1]))

    def test_non_gamma_multiple_geometry_trips_uniform_assert(self):
        """The gamma+1 verify geometry that crashed capture still trips the uniform-gamma assert here."""
        gamma, bs = 4, 1
        backend = _fake_backend(gamma=gamma, is_dspark_draft=True)
        num_q = (gamma + 1) * bs
        seq_lens_casual = torch.arange(1, num_q + 1, dtype=torch.int32)
        req_pool_indices_repeated = torch.zeros(num_q, dtype=torch.int32)
        out_loc = torch.arange(num_q, dtype=torch.int32) + 100000
        with self.assertRaises(AssertionError):
            backend.get_dspark_swa_page_indices(
                seq_lens_casual=seq_lens_casual,
                req_pool_indices_repeated=req_pool_indices_repeated,
                out_loc=out_loc,
                block_size=gamma,
            )


if __name__ == "__main__":
    unittest.main()
