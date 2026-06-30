import types
import unittest

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    _resolve_ragged_verify_layout,
    build_ragged_target_verify_geometry,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GRID = [8, 16, 24, 32, 64]


class TestRaggedVerifyGraphCapability(CustomTestCase):
    def test_base_backend_defaults_false(self):
        """The base attention backend does not advertise ragged verify graph support."""
        self.assertFalse(AttentionBackend.supports_ragged_verify_graph)

    def test_trtllm_mha_supports_ragged_verify_graph(self):
        """trtllm_mha advertises the ragged verify graph capability (drives runner admission)."""
        self.assertTrue(TRTLLMHAAttnBackend.supports_ragged_verify_graph)

    def test_dsv4_supports_ragged_verify_graph(self):
        """The DSV4 backend keeps ragged verify graph support after the probe refactor (no regression)."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        self.assertTrue(DeepseekV4AttnBackend.supports_ragged_verify_graph)


class TestResolveRaggedVerifyLayout(CustomTestCase):
    def test_none_without_spec_info(self):
        """A forward batch without spec_info has no ragged layout."""
        fb = types.SimpleNamespace(spec_info=None)
        self.assertIsNone(_resolve_ragged_verify_layout(fb))

    def test_none_without_layout_attr(self):
        """A spec_info missing the ragged_verify_layout attribute resolves to None."""
        fb = types.SimpleNamespace(spec_info=types.SimpleNamespace())
        self.assertIsNone(_resolve_ragged_verify_layout(fb))

    def test_returns_attached_layout(self):
        """An attached ragged_verify_layout is returned as-is (the per-batch geometry key)."""
        layout = RaggedVerifyLayout.uniform(
            bs=2, num_draft_tokens=8, device=_DEVICE, grid=_GRID
        )
        fb = types.SimpleNamespace(
            spec_info=types.SimpleNamespace(ragged_verify_layout=layout)
        )
        self.assertIs(_resolve_ragged_verify_layout(fb), layout)


class TestRaggedTargetVerifyGeometry(CustomTestCase):
    def test_mixed_verify_lens_geometry(self):
        """Mixed verify_lens build per-request cache_seqlens, variable qo_indptr, kv cumsum, and max q."""
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(geometry.cache_seqlens_int32.tolist(), [18, 21, 33])
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 9, 12])
        self.assertEqual(geometry.cu_seqlens_k.tolist(), [0, 18, 39, 72])
        self.assertEqual(geometry.max_seq_len_q, 8)

    def test_geometry_dtypes_are_int32(self):
        """The verify geometry tensors are int32 (the trtllm-gen kernel contract)."""
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int64)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(geometry.cache_seqlens_int32.dtype, torch.int32)
        self.assertEqual(geometry.cu_seqlens_q.dtype, torch.int32)
        self.assertEqual(geometry.cu_seqlens_k.dtype, torch.int32)

    def test_qo_indptr_total_matches_input_tokens(self):
        """cu_seqlens_q ends at the total verify-token count (the packed query length)."""
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(int(geometry.cu_seqlens_q[-1]), layout.total_verify_tokens)


class TestPaddedRaggedVerifyGeometry(CustomTestCase):
    def test_padded_layout_grows_bs_and_fills_bucket(self):
        """padded_to_bucket grows bs to graph_num_tokens//num_draft and the geometry covers every padded slot."""
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        )
        self.assertEqual(raw.graph_num_tokens, 32)
        padded = raw.padded_to_bucket(num_draft_tokens=8)
        self.assertEqual(padded.bs, 4)
        self.assertEqual(padded.verify_lens_cpu, [8, 1, 3, 20])
        self.assertEqual(padded.qo_indptr_device.tolist(), [0, 8, 9, 12, 32])
        # seq_lens carries the capture fill value (1) on the padded slot.
        seq_lens = torch.tensor([10, 20, 30, 1], dtype=torch.int32)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=padded)
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 9, 12, 32])
        self.assertEqual(geometry.cache_seqlens_int32.tolist(), [18, 21, 33, 21])
        self.assertEqual(int(geometry.cu_seqlens_k[-1]), 18 + 21 + 33 + 21)

    def test_padded_qo_indptr_reaches_graph_num_tokens(self):
        """The padded qo_indptr ends exactly at the frozen bucket (so batch_size == padded_bs)."""
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        )
        padded = raw.padded_to_bucket(num_draft_tokens=8)
        self.assertEqual(int(padded.qo_indptr_device[-1]), padded.graph_num_tokens)
        self.assertEqual(padded.qo_indptr_device.numel(), padded.bs + 1)


class TestNegativeSeamGeometry(CustomTestCase):
    def test_uniform_capture_geometry_diverges_from_ragged(self):
        """Forced-uniform geometry ([gamma+1]*bs) differs from a mixed ragged layout, so the negative seam has teeth."""
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        ragged = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        uniform = RaggedVerifyLayout.uniform(
            bs=3, num_draft_tokens=8, device=_DEVICE, grid=_GRID
        )
        g_ragged = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=ragged)
        g_uniform = build_ragged_target_verify_geometry(
            seq_lens=seq_lens, layout=uniform
        )
        self.assertNotEqual(
            g_ragged.cu_seqlens_q.tolist(), g_uniform.cu_seqlens_q.tolist()
        )
        self.assertNotEqual(
            g_ragged.cache_seqlens_int32.tolist(),
            g_uniform.cache_seqlens_int32.tolist(),
        )

    def test_uniform_layout_geometry_matches_legacy_arange(self):
        """A uniform layout reproduces the legacy fixed-stride qo_indptr (byte-identical to the old uniform verify)."""
        seq_lens = torch.tensor([5, 7, 9], dtype=torch.int32)
        uniform = RaggedVerifyLayout.uniform(
            bs=3, num_draft_tokens=8, device=_DEVICE, grid=_GRID
        )
        geometry = build_ragged_target_verify_geometry(
            seq_lens=seq_lens, layout=uniform
        )
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 16, 24])
        self.assertEqual(geometry.max_seq_len_q, 8)


class TestSwaPaddedSlotGuard(CustomTestCase):
    def test_clamp_keeps_gather_index_in_range(self):
        """The SWA gather clamp (mirrors the triton kernel) keeps a stale slot-0 slot inside [0, numel)."""
        full_to_swa_numel = 17
        slots = torch.tensor([-5, 0, 16, 100000, -1], dtype=torch.int64)
        clamped = torch.minimum(
            torch.maximum(slots, torch.zeros_like(slots)),
            torch.full_like(slots, full_to_swa_numel - 1),
        )
        self.assertTrue(bool((clamped >= 0).all()))
        self.assertTrue(bool((clamped < full_to_swa_numel).all()))
        self.assertEqual(clamped.tolist(), [0, 0, 16, 16, 0])


if __name__ == "__main__":
    unittest.main()
