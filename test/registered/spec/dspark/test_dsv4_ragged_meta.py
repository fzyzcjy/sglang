import types
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
from sglang.srt.layers.attention.deepseek_v4_backend import (
    RAGGED_VERIFY_CHOICES,
    RAGGED_VERIFY_CUTOFF_ONLY,
    RAGGED_VERIFY_FULL,
    RAGGED_VERIFY_OFF,
    DeepseekV4AttnBackend,
    _ragged_verify_mode,
    _resolve_ragged_verify_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_layout(
    verify_lens_cpu: list[int],
    *,
    graph_num_tokens: int | None = None,
) -> types.SimpleNamespace:
    verify_lens = torch.tensor(verify_lens_cpu, dtype=torch.int32)
    extend_start_loc = torch.tensor(
        [0, *torch.cumsum(verify_lens, dim=0).tolist()[:-1]], dtype=torch.int32
    )
    total = int(verify_lens.sum())
    return types.SimpleNamespace(
        verify_lens=verify_lens,
        verify_lens_cpu=list(verify_lens_cpu),
        total_verify_tokens=total,
        extend_start_loc=extend_start_loc,
        graph_num_tokens=total if graph_num_tokens is None else graph_num_tokens,
    )


def _make_forward_batch(layout) -> types.SimpleNamespace:
    spec_info = types.SimpleNamespace(ragged_verify_layout=layout)
    return types.SimpleNamespace(spec_info=spec_info)


def _stub_backend(
    *,
    num_draft_tokens: int = 6,
    online_c128_enabled: bool = False,
) -> types.SimpleNamespace:
    online = types.SimpleNamespace(enabled=lambda: online_c128_enabled)
    return types.SimpleNamespace(
        speculative_num_draft_tokens=num_draft_tokens,
        online_c128_mtp=online,
    )


class _patched_cp_size:
    def __init__(self, cp_size: int):
        self._cp_size = cp_size
        self._saved = None

    def __enter__(self):
        self._saved = dsv4.get_parallel
        dsv4.get_parallel = lambda: types.SimpleNamespace(attn_cp_size=self._cp_size)
        return self

    def __exit__(self, *exc):
        dsv4.get_parallel = self._saved
        return False


class TestRaggedVerifyMode(CustomTestCase):
    def test_unset_mode_returns_off(self):
        """Unset SGLANG_RAGGED_VERIFY resolves to the off sentinel."""
        with envs.SGLANG_RAGGED_VERIFY.override(""):
            self.assertEqual(_ragged_verify_mode(), RAGGED_VERIFY_OFF)

    def test_cutoff_only_mode_parsed(self):
        """cutoff-only is an accepted mode value."""
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_CUTOFF_ONLY):
            self.assertEqual(_ragged_verify_mode(), RAGGED_VERIFY_CUTOFF_ONLY)

    def test_full_mode_parsed(self):
        """full is an accepted mode value."""
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_FULL):
            self.assertEqual(_ragged_verify_mode(), RAGGED_VERIFY_FULL)

    def test_invalid_mode_fails_loud(self):
        """An unrecognised mode value raises instead of silently defaulting."""
        with envs.SGLANG_RAGGED_VERIFY.override("bogus"):
            with self.assertRaises(AssertionError):
                _ragged_verify_mode()

    def test_choices_are_off_cutoff_full(self):
        """The accepted mode set is exactly off / cutoff-only / full."""
        self.assertEqual(
            set(RAGGED_VERIFY_CHOICES),
            {RAGGED_VERIFY_OFF, RAGGED_VERIFY_CUTOFF_ONLY, RAGGED_VERIFY_FULL},
        )


class TestResolveRaggedVerifyLayout(CustomTestCase):
    def test_none_when_no_spec_info(self):
        """A forward batch without spec_info has no ragged layout."""
        fb = types.SimpleNamespace(spec_info=None)
        self.assertIsNone(_resolve_ragged_verify_layout(fb))

    def test_none_when_attr_absent(self):
        """A spec_info without the ragged_verify_layout attr resolves to None."""
        fb = types.SimpleNamespace(spec_info=types.SimpleNamespace())
        self.assertIsNone(_resolve_ragged_verify_layout(fb))

    def test_returns_attached_layout(self):
        """An attached ragged_verify_layout is returned as-is."""
        layout = _make_layout([6, 3, 1])
        fb = _make_forward_batch(layout)
        self.assertIs(_resolve_ragged_verify_layout(fb), layout)


class TestTargetVerifyGraphKey(CustomTestCase):
    def test_bs_keyed_when_no_layout(self):
        """Without a ragged layout the key stays (bs, num_draft*bs), byte-identical."""
        backend = _stub_backend(num_draft_tokens=6)
        key, num_tokens = DeepseekV4AttnBackend._target_verify_graph_key(
            backend, bs=3, ragged_layout=None
        )
        self.assertEqual(key, 3)
        self.assertEqual(num_tokens, 18)

    def test_token_keyed_when_full_layout(self):
        """A full ragged layout keys the graph by graph_num_tokens."""
        backend = _stub_backend(num_draft_tokens=6)
        layout = _make_layout([6, 3, 1])
        key, num_tokens = DeepseekV4AttnBackend._target_verify_graph_key(
            backend, bs=3, ragged_layout=layout
        )
        self.assertEqual(key, 10)
        self.assertEqual(num_tokens, 10)

    def test_graph_num_tokens_above_full_block_fails_loud(self):
        """graph_num_tokens may never exceed the full-block num_draft*bs budget."""
        backend = _stub_backend(num_draft_tokens=6)
        layout = _make_layout([6, 3, 1], graph_num_tokens=24)
        with self.assertRaises(AssertionError):
            DeepseekV4AttnBackend._target_verify_graph_key(
                backend, bs=3, ragged_layout=layout
            )

    def test_graph_num_tokens_must_equal_total(self):
        """A graph_num_tokens != total_verify_tokens (round-up bucket) fails loud here."""
        backend = _stub_backend(num_draft_tokens=6)
        layout = _make_layout([6, 3, 1], graph_num_tokens=12)
        with self.assertRaises(AssertionError):
            DeepseekV4AttnBackend._target_verify_graph_key(
                backend, bs=3, ragged_layout=layout
            )

    def test_uniform_full_block_layout_matches_bs_block(self):
        """A uniform full-block layout (all num_draft) yields num_draft*bs tokens."""
        backend = _stub_backend(num_draft_tokens=6)
        layout = _make_layout([6, 6, 6])
        key, num_tokens = DeepseekV4AttnBackend._target_verify_graph_key(
            backend, bs=3, ragged_layout=layout
        )
        self.assertEqual(key, 18)
        self.assertEqual(num_tokens, 18)

    def test_worked_example_geometry(self):
        """DSK worked example: verify_lens=[6,3,1] -> total 10, exclusive-cumsum starts."""
        layout = _make_layout([6, 3, 1])
        self.assertEqual(layout.total_verify_tokens, 10)
        self.assertEqual(layout.extend_start_loc.tolist(), [0, 6, 9])
        backend = _stub_backend(num_draft_tokens=6)
        key, num_tokens = DeepseekV4AttnBackend._target_verify_graph_key(
            backend, bs=3, ragged_layout=layout
        )
        self.assertEqual((key, num_tokens), (10, 10))


class TestResolveVerifyLayoutGating(CustomTestCase):
    def test_none_when_no_layout_attached(self):
        """No layout on spec_info means the backend stays on the full-block path."""
        backend = _stub_backend()
        fb = _make_forward_batch(None)
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_FULL):
            self.assertIsNone(
                DeepseekV4AttnBackend._resolve_verify_layout(backend, fb, bs=3)
            )

    def test_none_when_mode_not_full(self):
        """cutoff-only keeps the bs-keyed full-block graph (layout ignored here)."""
        backend = _stub_backend()
        layout = _make_layout([6, 3, 1])
        fb = _make_forward_batch(layout)
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_CUTOFF_ONLY):
            self.assertIsNone(
                DeepseekV4AttnBackend._resolve_verify_layout(backend, fb, bs=3)
            )

    def test_returns_layout_when_full(self):
        """full mode plus an attached layout resolves to the ragged layout."""
        backend = _stub_backend()
        layout = _make_layout([6, 3, 1])
        fb = _make_forward_batch(layout)
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_FULL):
            with _patched_cp_size(1):
                resolved = DeepseekV4AttnBackend._resolve_verify_layout(
                    backend, fb, bs=3
                )
        self.assertIs(resolved, layout)

    def test_context_parallel_fails_loud(self):
        """Context parallel combined with ragged verify must raise NotImplementedError."""
        backend = _stub_backend()
        layout = _make_layout([6, 3, 1])
        fb = _make_forward_batch(layout)
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_FULL):
            with _patched_cp_size(2):
                with self.assertRaises(NotImplementedError):
                    DeepseekV4AttnBackend._resolve_verify_layout(backend, fb, bs=3)

    def test_online_c128_fails_loud(self):
        """Online c128 MTP combined with ragged verify must raise NotImplementedError."""
        backend = _stub_backend(online_c128_enabled=True)
        layout = _make_layout([6, 3, 1])
        fb = _make_forward_batch(layout)
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_FULL):
            with _patched_cp_size(1):
                with self.assertRaises(NotImplementedError):
                    DeepseekV4AttnBackend._resolve_verify_layout(backend, fb, bs=3)

    def test_anchor_assertion_rejects_zero_verify_len(self):
        """A verify_len < 1 (no anchor verified) trips the resolve assertion."""
        backend = _stub_backend()
        layout = _make_layout([6, 3, 0])
        fb = _make_forward_batch(layout)
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_FULL):
            with _patched_cp_size(1):
                with self.assertRaises(AssertionError):
                    DeepseekV4AttnBackend._resolve_verify_layout(backend, fb, bs=3)

    def test_bs_mismatch_fails_loud(self):
        """A layout whose verify_lens_cpu length disagrees with bs trips an assertion."""
        backend = _stub_backend()
        layout = _make_layout([6, 3, 1])
        fb = _make_forward_batch(layout)
        with envs.SGLANG_RAGGED_VERIFY.override(RAGGED_VERIFY_FULL):
            with _patched_cp_size(1):
                with self.assertRaises(AssertionError):
                    DeepseekV4AttnBackend._resolve_verify_layout(backend, fb, bs=2)


if __name__ == "__main__":
    unittest.main()
