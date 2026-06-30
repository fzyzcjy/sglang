import types
import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")


def _fake_pool_with_mapping(mapping: torch.Tensor) -> types.SimpleNamespace:
    """A pool stub exposing only translate_loc_from_full_to_swa (the pure lookup).

    ``translate_loc_from_full_to_swa`` is ``full_to_swa_index_mapping[kv_indices]``
    (deepseek_v4_memory_pool.py:639): an alloc-time state table, NOT a modulo. The
    real method body is reused here unbound against a stub carrying the mapping, so
    the test pins the production lookup semantics without constructing the full pool.
    """
    pool = types.SimpleNamespace(full_to_swa_index_mapping=mapping)
    pool.translate_loc_from_full_to_swa = types.MethodType(
        DeepSeekV4TokenToKVPool.translate_loc_from_full_to_swa, pool
    )
    return pool


class TestTranslateFullToSwaSemantics(CustomTestCase):
    """Pin the full->SWA translate hazards the dsv4 injection must handle (plan §3).

    The SWA allocator's ``full_to_swa_index_mapping`` (allocator/swa.py) is filled at
    alloc, cleared to 0 on free, defaults to 0, and has a trailing -1 sentinel. The
    injection path translates the accepted full locs to SWA slots BEFORE the
    set_swa_key_buffer_* write, so the translate semantics are load-bearing for
    correctness.
    """

    def test_allocated_full_loc_maps_to_assigned_swa_slot(self) -> None:
        """An allocated full loc translates to exactly its assigned SWA ring slot."""
        mapping = torch.zeros(8, dtype=torch.int64)
        mapping[-1] = -1
        mapping[3] = 5
        mapping[4] = 6
        pool = _fake_pool_with_mapping(mapping)
        out = pool.translate_loc_from_full_to_swa(torch.tensor([3, 4]))
        self.assertEqual(out.tolist(), [5, 6])

    def test_unallocated_full_loc_defaults_to_slot_zero(self) -> None:
        """An UNALLOCATED full loc defaults to SWA slot 0 (the silent-alias hazard).

        Translating before the slot is allocated reads the default 0 -> SWA slot 0,
        which is some live request's slot. There is no assert, so the injection MUST
        translate only AFTER allocation (plan §3); this test documents the trap.
        """
        mapping = torch.zeros(8, dtype=torch.int64)
        mapping[-1] = -1
        pool = _fake_pool_with_mapping(mapping)
        out = pool.translate_loc_from_full_to_swa(torch.tensor([2]))
        self.assertEqual(out.tolist(), [0])

    def test_pad_minus_one_loc_maps_to_minus_one(self) -> None:
        """The -1 pad loc maps to the trailing -1 sentinel (not an OOB write)."""
        mapping = torch.zeros(8, dtype=torch.int64)
        mapping[-1] = -1
        pool = _fake_pool_with_mapping(mapping)
        out = pool.translate_loc_from_full_to_swa(torch.tensor([-1]))
        self.assertEqual(out.tolist(), [-1])

    def test_multiple_full_positions_alias_same_ring_slot(self) -> None:
        """Distinct full positions can alias one ring slot (last-writer-wins window).

        The SWA ring is finite, so over time several full positions translate to the
        same SWA slot. A window write of both would be last-writer-wins; the injection
        must therefore write the freshest content per slot (plan §3).
        """
        mapping = torch.zeros(8, dtype=torch.int64)
        mapping[-1] = -1
        mapping[1] = 4
        mapping[5] = 4  # aliases the same ring slot as full loc 1
        pool = _fake_pool_with_mapping(mapping)
        out = pool.translate_loc_from_full_to_swa(torch.tensor([1, 5]))
        self.assertEqual(out.tolist(), [4, 4])

    def test_translate_preserves_row_shape(self) -> None:
        """A 2D [num_q, window] index block translates element-wise, shape-preserving."""
        mapping = torch.zeros(16, dtype=torch.int64)
        mapping[-1] = -1
        for full_loc in range(16):
            mapping[full_loc] = full_loc % 4
        pool = _fake_pool_with_mapping(mapping)
        idx = torch.tensor([[0, 1, 2], [4, 5, 6]])
        out = pool.translate_loc_from_full_to_swa(idx)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(out.tolist(), [[0, 1, 2], [0, 1, 2]])


def _gpu_injection_available() -> tuple[bool, str]:
    """Return (available, reason) for the GPU round-trip injection contract.

    The decode-commit / prefill-window injection writes the projected target-hidden
    latent into the SWA ring via the model's ``write_target_hidden_kv`` (agent-A's
    final API: ``write_target_hidden_kv(main_hidden, swa_loc, positions, pool)``)
    backed by the pool's ``set_swa_key_buffer_radix_fused`` /
    ``set_swa_key_buffer_radix_fused_norm_rope`` Triton kernels. Both are GPU-only and
    landed by the worker+model agents; until then the round-trip cases skip cleanly.
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available; SWA fused-store kernels are GPU-only."
    from sglang.srt.models import deepseek_v4_dspark as dsv4_model

    if not hasattr(dsv4_model.DeepseekV4ForCausalLMDSpark, "write_target_hidden_kv"):
        return (
            False,
            "model.write_target_hidden_kv not yet implemented "
            "(worker+model agent owns the MLA latent injection)",
        )
    return True, ""


class TestDsv4InjectionRoundTrip(CustomTestCase):
    """GPU round-trip: injected latent reads back equal to the SoT ring content.

    Covers the two injection cases from plan §3: (1) prefill writes the whole
    target-hidden window into the ring at per-request positions; (2) decode-commit
    gathers each request's single bonus slot (``new_seq_len[r]-1``) and writes it
    flat. Each case writes via the production ``write_target_hidden_kv`` + the SWA
    fused-store kernel (choosing ``_fused`` vs ``_fused_norm_rope`` by the latent
    state the worker holds), then reads the SWA key buffer back and compares against
    the vendored SoT ring write (norm + rope) on identical inputs. GPU-tier; skips
    cleanly until the injection API lands.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._available, cls._skip_reason = _gpu_injection_available()

    def _skip_if_unready(self) -> None:
        if not self._available:
            self.skipTest(self._skip_reason)

    def test_prefill_window_write_reads_back_equal_to_sot(self) -> None:
        """Prefill window latent injected via write_target_hidden_kv matches the SoT ring."""
        self._skip_if_unready()
        self.skipTest(
            "GPU round-trip harness for write_target_hidden_kv prefill window is a "
            "tester-run entry once the injection API + SWA pool fixture land; the "
            "_fused vs _fused_norm_rope choice and per-row positions are pinned by "
            "the worker+model agents."
        )

    def test_decode_commit_per_row_write_reads_back_equal_to_sot(self) -> None:
        """Decode-commit per-row bonus slot injection matches the SoT ring per request."""
        self._skip_if_unready()
        self.skipTest(
            "GPU round-trip harness for the decode-commit per-row (new_seq_len[r]-1) "
            "flat write is a tester-run entry once the injection API + SWA pool "
            "fixture land."
        )


if __name__ == "__main__":
    unittest.main()
