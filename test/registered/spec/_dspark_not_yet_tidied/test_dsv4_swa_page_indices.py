import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    PAGE_INDEX_ALIGNED_SIZE,
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    _compact_dspark_window_then_block,
    build_dspark_swa_page_indices,
    compute_dspark_window_gather,
)
from sglang.srt.utils import ceil_align
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _distinct_window(*, bs: int, base: int = 1) -> torch.Tensor:
    """Window with every column distinct and >= base, so a wrong gather slice is caught."""
    return (base + torch.arange(bs * SWA_WINDOW, dtype=torch.int32)).view(
        bs, SWA_WINDOW
    )


def _front_padded_window(*, context_lens: list[int], base: int = 1) -> torch.Tensor:
    """Window with the pre-start prefix left-padded -1 and the valid suffix distinct >= base."""
    bs = len(context_lens)
    window = torch.full((bs, SWA_WINDOW), -1, dtype=torch.int32)
    for r, cl in enumerate(context_lens):
        for j in range(SWA_WINDOW - cl, SWA_WINDOW):
            window[r, j] = base + r * SWA_WINDOW + j
    return window


def _distinct_block(*, bs: int, block_size: int, base: int = 900000) -> torch.Tensor:
    """Block slots distinct from the window slots so placement is identifiable."""
    return (base + torch.arange(bs * block_size, dtype=torch.int32)).view(
        bs, block_size
    )


def _oracle_build_page_indices(
    *,
    window_swa_locs: torch.Tensor,
    block_swa_locs: torch.Tensor,
    context_lens: list[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-consistent contract rebuild over the INPUT slot arrays (no reference integers).

    Each request row = its last context window slots, then its whole block, then -1; the row
    is replicated across all block_size non-causal query rows and topk = context + block_size.
    """
    bs = window_swa_locs.shape[0]
    target_width = ceil_align(SWA_WINDOW + block_size, PAGE_INDEX_ALIGNED_SIZE)
    rows: list[list[int]] = []
    topks: list[int] = []
    for r in range(bs):
        cl = int(context_lens[r])
        context_slots = window_swa_locs[r, SWA_WINDOW - cl : SWA_WINDOW].tolist()
        block_slots = block_swa_locs[r].tolist()
        row = context_slots + block_slots + [-1] * (target_width - cl - block_size)
        for _ in range(block_size):
            rows.append(row)
            topks.append(cl + block_size)
    return (
        torch.tensor(rows, dtype=torch.int32),
        torch.tensor(topks, dtype=torch.int32),
    )


class TestBuildDsparkSwaPageIndicesPlumbing(CustomTestCase):
    def test_plumbing_oracle_matches_over_block_sizes_and_contexts(self):
        """build matches the contract rebuild for every block_size and mixed per-request context."""
        for block_size in (1, 4, 63, 64, 65):
            context_lens = [0, 1, SWA_WINDOW - 1, SWA_WINDOW]
            bs = len(context_lens)
            window = _distinct_window(bs=bs)
            block = _distinct_block(bs=bs, block_size=block_size)
            context = torch.tensor(context_lens, dtype=torch.int32)
            page_indices, topk = build_dspark_swa_page_indices(
                window_swa_locs=window,
                block_swa_locs=block,
                context_lens=context,
                block_size=block_size,
            )
            exp_indices, exp_topk = _oracle_build_page_indices(
                window_swa_locs=window,
                block_swa_locs=block,
                context_lens=context_lens,
                block_size=block_size,
            )
            self.assertTrue(
                torch.equal(page_indices, exp_indices), msg=f"{block_size=}"
            )
            self.assertTrue(torch.equal(topk, exp_topk), msg=f"{block_size=}")
            self.assertEqual(page_indices.dtype, torch.int32)
            self.assertEqual(topk.dtype, torch.int32)
            self.assertEqual(page_indices.device, window.device)

    def test_completeness_no_minus_one_holes_in_valid_prefix(self):
        """The valid topk prefix never contains a -1 hole; only the tail past topk is -1."""
        block_size = 4
        context_lens = [0, 2, SWA_WINDOW]
        window = _front_padded_window(context_lens=context_lens)
        block = _distinct_block(bs=len(context_lens), block_size=block_size)
        context = torch.tensor(context_lens, dtype=torch.int32)
        page_indices, topk = build_dspark_swa_page_indices(
            window_swa_locs=window,
            block_swa_locs=block,
            context_lens=context,
            block_size=block_size,
        )
        for q in range(page_indices.shape[0]):
            t = int(topk[q])
            self.assertTrue((page_indices[q, :t] >= 0).all().item(), msg=f"{q=}")
            self.assertTrue((page_indices[q, t:] == -1).all().item(), msg=f"{q=}")

    def test_block_size_64_context_128_fills_whole_row_no_tail(self):
        """block_size=64 with full window fills target_width=192 entirely, no -1 tail."""
        block_size = 64
        context = torch.tensor([SWA_WINDOW], dtype=torch.int32)
        window = _distinct_window(bs=1)
        block = _distinct_block(bs=1, block_size=block_size)
        page_indices, topk = build_dspark_swa_page_indices(
            window_swa_locs=window,
            block_swa_locs=block,
            context_lens=context,
            block_size=block_size,
        )
        target_width = ceil_align(SWA_WINDOW + block_size, PAGE_INDEX_ALIGNED_SIZE)
        self.assertEqual(target_width, SWA_WINDOW + block_size)
        self.assertEqual(int(topk[0]), target_width)
        self.assertTrue((page_indices[0] >= 0).all().item())

    def test_row_replication_is_non_causal(self):
        """All block_size query rows of a request are byte-identical and share the same topk."""
        block_size = 3
        context_lens = [5, SWA_WINDOW]
        bs = len(context_lens)
        window = _distinct_window(bs=bs)
        block = _distinct_block(bs=bs, block_size=block_size)
        context = torch.tensor(context_lens, dtype=torch.int32)
        page_indices, topk = build_dspark_swa_page_indices(
            window_swa_locs=window,
            block_swa_locs=block,
            context_lens=context,
            block_size=block_size,
        )
        for r, cl in enumerate(context_lens):
            block_rows = page_indices[r * block_size : (r + 1) * block_size]
            self.assertTrue(torch.equal(block_rows[0], block_rows[-1]))
            block_topk = topk[r * block_size : (r + 1) * block_size]
            self.assertTrue((block_topk == cl + block_size).all().item())

    def test_bs_one_single_request(self):
        """A single-request batch packs context then block with no broadcast leakage."""
        block_size = 5
        context_lens = [3]
        window = _distinct_window(bs=1)
        block = _distinct_block(bs=1, block_size=block_size)
        context = torch.tensor(context_lens, dtype=torch.int32)
        page_indices, topk = build_dspark_swa_page_indices(
            window_swa_locs=window,
            block_swa_locs=block,
            context_lens=context,
            block_size=block_size,
        )
        exp_indices, exp_topk = _oracle_build_page_indices(
            window_swa_locs=window,
            block_swa_locs=block,
            context_lens=context_lens,
            block_size=block_size,
        )
        self.assertEqual(page_indices.shape[0], block_size)
        self.assertTrue(torch.equal(page_indices, exp_indices))
        self.assertTrue(torch.equal(topk, exp_topk))

    def test_window_wrong_shape_raises(self):
        """A window that is not [bs, SWA_WINDOW] is rejected loudly."""
        block_size = 4
        with self.assertRaises(ValueError):
            build_dspark_swa_page_indices(
                window_swa_locs=torch.zeros((2, SWA_WINDOW - 1), dtype=torch.int32),
                block_swa_locs=torch.zeros((2, block_size), dtype=torch.int32),
                context_lens=torch.zeros(2, dtype=torch.int32),
                block_size=block_size,
            )

    def test_block_wrong_shape_raises(self):
        """A block whose width disagrees with block_size is rejected loudly."""
        block_size = 4
        with self.assertRaises(ValueError):
            build_dspark_swa_page_indices(
                window_swa_locs=torch.zeros((2, SWA_WINDOW), dtype=torch.int32),
                block_swa_locs=torch.zeros((2, block_size + 1), dtype=torch.int32),
                context_lens=torch.zeros(2, dtype=torch.int32),
                block_size=block_size,
            )


class TestCompactWindowThenBlockContract(CustomTestCase):
    def _compact(
        self,
        *,
        window: torch.Tensor,
        block: torch.Tensor,
        context_lens: list[int],
        block_size: int,
    ) -> torch.Tensor:
        target_width = ceil_align(SWA_WINDOW + block_size, PAGE_INDEX_ALIGNED_SIZE)
        return _compact_dspark_window_then_block(
            window_swa_locs=window,
            block_swa_locs=block,
            context_lens=torch.tensor(context_lens, dtype=torch.int32),
            target_width=target_width,
            block_size=block_size,
        )

    def test_left_pack_gathers_last_context_entries(self):
        """out[r, :cl] equals the last cl window entries (index oracle, not the boolean mask)."""
        block_size = 4
        context_lens = [0, 1, 7, SWA_WINDOW]
        bs = len(context_lens)
        window = _distinct_window(bs=bs)
        block = _distinct_block(bs=bs, block_size=block_size)
        out = self._compact(
            window=window, block=block, context_lens=context_lens, block_size=block_size
        )
        for r, cl in enumerate(context_lens):
            expected = window[r, SWA_WINDOW - cl : SWA_WINDOW]
            self.assertTrue(torch.equal(out[r, :cl], expected), msg=f"{r=}")

    def test_block_slots_placed_after_context(self):
        """out[r, cl:cl+block_size] equals the request's block slots."""
        block_size = 3
        context_lens = [0, 2, SWA_WINDOW]
        bs = len(context_lens)
        window = _distinct_window(bs=bs)
        block = _distinct_block(bs=bs, block_size=block_size)
        out = self._compact(
            window=window, block=block, context_lens=context_lens, block_size=block_size
        )
        for r, cl in enumerate(context_lens):
            self.assertTrue(
                torch.equal(out[r, cl : cl + block_size], block[r]), msg=f"{r=}"
            )

    def test_tail_after_block_is_minus_one(self):
        """out[r, cl+block_size:] is all -1 for a mixed-context batch."""
        block_size = 5
        context_lens = [0, 2, SWA_WINDOW]
        bs = len(context_lens)
        window = _distinct_window(bs=bs)
        block = _distinct_block(bs=bs, block_size=block_size)
        out = self._compact(
            window=window, block=block, context_lens=context_lens, block_size=block_size
        )
        for r, cl in enumerate(context_lens):
            self.assertTrue((out[r, cl + block_size :] == -1).all().item(), msg=f"{r=}")

    def test_bs_one_single_request(self):
        """A single-request compact packs context then block with no broadcast leakage."""
        block_size = 6
        context_lens = [4]
        window = _distinct_window(bs=1)
        block = _distinct_block(bs=1, block_size=block_size)
        out = self._compact(
            window=window, block=block, context_lens=context_lens, block_size=block_size
        )
        cl = context_lens[0]
        self.assertTrue(
            torch.equal(out[0, :cl], window[0, SWA_WINDOW - cl : SWA_WINDOW])
        )
        self.assertTrue(torch.equal(out[0, cl : cl + block_size], block[0]))
        self.assertTrue((out[0, cl + block_size :] == -1).all().item())


def _scrambled_req_to_token(
    *, num_reqs: int, max_cols: int, base: int = 10
) -> torch.Tensor:
    """A ring-style req_to_token: each request scatters its positions across the slot range.

    ``37`` is coprime to ``max_cols``, so within any request the column->slot map is a
    bijection whose consecutive positions are non-monotonic (the window [prefix-W, prefix-1]
    is scattered, i.e. wraps across the ring), exercising the gather under cross-boundary
    layouts while keeping every position's slot distinct (full-information SET assertions).
    """
    table = torch.empty((num_reqs, max_cols), dtype=torch.int64)
    for req in range(num_reqs):
        for p in range(max_cols):
            table[req, p] = base + req * max_cols + (p * 37) % max_cols
    return table


def _affine_translate(slots: torch.Tensor) -> torch.Tensor:
    """Injective, non-identity full->swa map so a forgotten/wrong translate is caught."""
    return slots * 3 + 5


class TestGetDsparkSwaPageIndicesOrchestrator(CustomTestCase):
    def _expected_attended_set(
        self,
        *,
        req_to_token: torch.Tensor,
        translate,
        prefix: int,
        req: int,
        out_loc_block: torch.Tensor,
    ) -> set:
        context = min(prefix, SWA_WINDOW)
        context_slots = {
            int(translate(int(req_to_token[req, p])))
            for p in range(prefix - context, prefix)
        }
        block_slots = {int(translate(int(slot))) for slot in out_loc_block.tolist()}
        return context_slots | block_slots

    def test_attended_slot_sets_match_for_both_caller_gammas(self):
        """Each query row attends exactly {recent context slots} u {block slots}, both gammas."""
        num_reqs, max_cols = 8, 600
        req_to_token = _scrambled_req_to_token(num_reqs=num_reqs, max_cols=max_cols)
        backend = SimpleNamespace(
            req_to_token=req_to_token,
            token_to_kv_pool=SimpleNamespace(
                translate_loc_from_full_to_swa=_affine_translate
            ),
        )
        # The eager make_core_attn_metadata caller passes block_size=num_draft-1; the
        # in-graph replay caller passes draft_token_num. Cover both gamma geometries.
        for block_size in (3, 4):
            prefixes = [0, 5, 130, 200]
            req_ids = [1, 2, 4, 7]
            bs = len(prefixes)
            seq_lens: list[int] = []
            req_pool: list[int] = []
            for prefix, req in zip(prefixes, req_ids):
                for k in range(block_size):
                    seq_lens.append(prefix + 1 + k)
                    req_pool.append(req)
            seq_lens_casual = torch.tensor(seq_lens, dtype=torch.int32)
            req_pool_indices_repeated = torch.tensor(req_pool, dtype=torch.int32)
            num_q = bs * block_size
            out_loc = 500000 + torch.arange(num_q, dtype=torch.int64)
            page_indices, topk = DeepseekV4AttnBackend.get_dspark_swa_page_indices(
                backend,
                seq_lens_casual=seq_lens_casual,
                req_pool_indices_repeated=req_pool_indices_repeated,
                out_loc=out_loc,
                block_size=block_size,
            )
            for r, (prefix, req) in enumerate(zip(prefixes, req_ids)):
                expected = self._expected_attended_set(
                    req_to_token=req_to_token,
                    translate=_affine_translate,
                    prefix=prefix,
                    req=req,
                    out_loc_block=out_loc[r * block_size : (r + 1) * block_size],
                )
                for j in range(block_size):
                    q = r * block_size + j
                    t = int(topk[q])
                    self.assertEqual(t, min(prefix, SWA_WINDOW) + block_size)
                    attended = page_indices[q, :t]
                    self.assertTrue(
                        (attended >= 0).all().item(), msg=f"{block_size=} {r=}"
                    )
                    self.assertTrue(
                        (page_indices[q, t:] == -1).all().item(),
                        msg=f"{block_size=} {r=}",
                    )
                    self.assertEqual(
                        set(attended.tolist()), expected, msg=f"{block_size=} {r=} {j=}"
                    )
                rows = page_indices[r * block_size : (r + 1) * block_size]
                self.assertTrue(torch.equal(rows[0], rows[-1]))

    def test_invalid_window_positions_do_not_leak_into_prefix(self):
        """Pre-start window positions are filled 0 -> translated -> -1, never leaking into topk.

        With a prefix < SWA_WINDOW the front window columns are invalid. The method fills them
        0 in full space, translates, then masks them -1 in swa space (this masked_fill order:
        fill-0 -> translate -> fill-(-1)). The translate maps 0 -> 0, a value made distinct
        from every legitimate slot, so its absence from the valid prefix pins both the masking
        order and the gather-the-last-context direction (a front gather would surface 0 / -1).
        """
        swa_size = 997

        def translate(slots: torch.Tensor) -> torch.Tensor:
            return slots % swa_size

        block_size = 3
        prefix, req = 2, 4
        num_reqs, max_cols = 8, 64
        req_to_token = torch.zeros((num_reqs, max_cols), dtype=torch.int64)
        # Position-0 and position-1 slots translate to 1 and 2; the masked invalid value is 0.
        req_to_token[req, 0] = 1
        req_to_token[req, 1] = 2
        backend = SimpleNamespace(
            req_to_token=req_to_token,
            token_to_kv_pool=SimpleNamespace(translate_loc_from_full_to_swa=translate),
        )
        seq_lens_casual = torch.tensor(
            [prefix + 1 + k for k in range(block_size)], dtype=torch.int32
        )
        req_pool_indices_repeated = torch.full((block_size,), req, dtype=torch.int32)
        out_loc = torch.tensor([10, 20, 30], dtype=torch.int64)
        page_indices, topk = DeepseekV4AttnBackend.get_dspark_swa_page_indices(
            backend,
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            out_loc=out_loc,
            block_size=block_size,
        )
        expected = {1, 2, 10, 20, 30}
        for j in range(block_size):
            t = int(topk[j])
            self.assertEqual(t, prefix + block_size)
            attended = page_indices[j, :t]
            self.assertTrue((attended >= 0).all().item())
            self.assertNotIn(0, attended.tolist())
            self.assertEqual(set(attended.tolist()), expected)


def _uniform_layout(
    *, prefixes: list[int], block_size: int, req_ids: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the uniform-gamma per-token (seq_lens_casual, req_pool) for the given prefixes."""
    if req_ids is None:
        req_ids = list(range(len(prefixes)))
    seq: list[int] = []
    req_pool: list[int] = []
    for prefix, req in zip(prefixes, req_ids):
        for k in range(block_size):
            seq.append(prefix + 1 + k)
            req_pool.append(req)
    return (
        torch.tensor(seq, dtype=torch.int32),
        torch.tensor(req_pool, dtype=torch.int32),
    )


def _oracle_window_gather(
    *,
    seq_lens_casual: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference (prefix, context, offsets, invalid) replicating the PRE-clamp ordering."""
    s = seq_lens_casual.to(torch.int32)
    bs = s.numel() // block_size
    first = torch.arange(bs, dtype=torch.int64) * block_size
    prefix = (s[first] - 1).to(torch.int32)
    context = torch.clamp(prefix, max=SWA_WINDOW).to(torch.int32)
    offsets_raw = (
        prefix.to(torch.int64).unsqueeze(1)
        - SWA_WINDOW
        + torch.arange(SWA_WINDOW, dtype=torch.int64).unsqueeze(0)
    )
    invalid = offsets_raw < 0
    offsets = offsets_raw.clamp(min=0)
    return prefix, context, offsets, invalid


class TestComputeDsparkWindowGather(CustomTestCase):
    def test_fields_match_preclamp_oracle(self):
        """offsets / invalid / context_lens match the pre-clamp reference for mixed prefixes."""
        block_size = 3
        prefixes = [0, 5, SWA_WINDOW, SWA_WINDOW + 7]
        req_ids = [2, 5, 1, 6]
        seq_lens_casual, req_pool = _uniform_layout(
            prefixes=prefixes, block_size=block_size, req_ids=req_ids
        )
        gather = compute_dspark_window_gather(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool,
            block_size=block_size,
        )
        _, context, offsets, invalid = _oracle_window_gather(
            seq_lens_casual=seq_lens_casual, block_size=block_size
        )
        self.assertEqual(gather.num_q, len(prefixes) * block_size)
        self.assertEqual(gather.bs, len(prefixes))
        self.assertTrue(torch.equal(gather.context_lens, context))
        self.assertTrue(torch.equal(gather.offsets, offsets))
        self.assertTrue(torch.equal(gather.invalid, invalid))
        self.assertEqual(gather.offsets.dtype, torch.int64)
        self.assertEqual(gather.invalid.dtype, torch.bool)
        self.assertEqual(gather.context_lens.dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                gather.req_pool_indices_per_request,
                torch.tensor(req_ids, dtype=torch.int32),
            )
        )

    def test_context_lens_equals_valid_offset_count(self):
        """context_lens binds to the actual -1 positions: context == (~invalid).sum(dim=1)."""
        block_size = 2
        prefixes = [0, SWA_WINDOW - 1, SWA_WINDOW, SWA_WINDOW + 5]
        seq_lens_casual, req_pool = _uniform_layout(
            prefixes=prefixes, block_size=block_size
        )
        gather = compute_dspark_window_gather(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool,
            block_size=block_size,
        )
        self.assertTrue(
            torch.equal(
                gather.context_lens, (~gather.invalid).sum(dim=1).to(torch.int32)
            )
        )

    def test_invalid_marks_exactly_pre_start_positions(self):
        """invalid is True iff prefix - SWA_WINDOW + arange(W) < 0 (the front W-prefix cols)."""
        block_size = 4
        prefixes = [3, SWA_WINDOW]
        seq_lens_casual, req_pool = _uniform_layout(
            prefixes=prefixes, block_size=block_size
        )
        gather = compute_dspark_window_gather(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool,
            block_size=block_size,
        )
        arange = torch.arange(SWA_WINDOW, dtype=torch.int64)
        for r, prefix in enumerate(prefixes):
            expected = (prefix - SWA_WINDOW + arange) < 0
            self.assertTrue(torch.equal(gather.invalid[r], expected), msg=f"{prefix=}")

    def test_offsets_slide_with_prefix_not_clamped_dead(self):
        """For prefix >= W context stays W but offsets keep sliding (W+1 drops the oldest)."""
        block_size = 1
        arange = torch.arange(SWA_WINDOW, dtype=torch.int64)
        offsets_by_prefix: dict[int, torch.Tensor] = {}
        for prefix in (SWA_WINDOW - 1, SWA_WINDOW, SWA_WINDOW + 1, SWA_WINDOW + 5):
            seq_lens_casual, req_pool = _uniform_layout(
                prefixes=[prefix], block_size=block_size
            )
            gather = compute_dspark_window_gather(
                seq_lens_casual=seq_lens_casual,
                req_pool_indices_repeated=req_pool,
                block_size=block_size,
            )
            expected = (prefix - SWA_WINDOW + arange).clamp(min=0)
            self.assertTrue(torch.equal(gather.offsets[0], expected), msg=f"{prefix=}")
            offsets_by_prefix[prefix] = gather.offsets[0]
        self.assertTrue(
            torch.equal(
                offsets_by_prefix[SWA_WINDOW + 1], offsets_by_prefix[SWA_WINDOW] + 1
            )
        )

    def test_bs_one_single_request(self):
        """A single-request batch yields bs==1 and a [1, SWA_WINDOW] offsets/invalid."""
        block_size = 5
        seq_lens_casual, req_pool = _uniform_layout(prefixes=[9], block_size=block_size)
        gather = compute_dspark_window_gather(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool,
            block_size=block_size,
        )
        self.assertEqual(gather.bs, 1)
        self.assertEqual(gather.num_q, block_size)
        self.assertEqual(tuple(gather.offsets.shape), (1, SWA_WINDOW))
        self.assertEqual(int(gather.context_lens[0]), 9)

    def test_non_divisible_num_q_raises(self):
        """A num_q that is not a block_size multiple trips the uniform-gamma assert."""
        block_size = 4
        seq_lens_casual = torch.arange(1, (block_size + 1) + 1, dtype=torch.int32)
        req_pool = torch.zeros(block_size + 1, dtype=torch.int32)
        with self.assertRaises(AssertionError):
            compute_dspark_window_gather(
                seq_lens_casual=seq_lens_casual,
                req_pool_indices_repeated=req_pool,
                block_size=block_size,
            )


if __name__ == "__main__":
    unittest.main()
