import unittest
from types import SimpleNamespace

import torch

import sglang.srt.speculative.dflash_info as dflash_info
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_verify_input(
    *, draft_token_num: int, ragged_verify_layout=None
) -> DFlashVerifyInput:
    return DFlashVerifyInput(
        draft_token=torch.zeros((1,), dtype=torch.int64),
        positions=torch.zeros((1,), dtype=torch.int64),
        draft_token_num=draft_token_num,
        custom_mask=None,
        ragged_verify_layout=ragged_verify_layout,
    )


def _fake_layout(verify_lens_cpu: list[int]):
    verify_lens = torch.tensor(verify_lens_cpu, dtype=torch.int32)
    qo_indptr_device = torch.zeros((len(verify_lens_cpu) + 1,), dtype=torch.int32)
    qo_indptr_device[1:] = torch.cumsum(verify_lens, dim=0)
    return SimpleNamespace(
        verify_lens=verify_lens,
        verify_lens_cpu=verify_lens_cpu,
        total_verify_tokens=sum(verify_lens_cpu),
        qo_indptr_device=qo_indptr_device,
    )


class TestDFlashRaggedVerifyMetadata(CustomTestCase):
    def setUp(self):
        # generate_attn_arg_prefill calls a Triton kv-indices kernel that needs
        # CUDA; stub it so the host-side qo_indptr / sizing logic is testable on
        # CPU. The kernel only fills kv_indices (already allocated by the caller).
        self._orig_kernel = dflash_info.create_flashinfer_kv_indices_triton

        class _NoopKernel:
            def __getitem__(self, _grid):
                return lambda *args, **kwargs: None

        dflash_info.create_flashinfer_kv_indices_triton = _NoopKernel()

    def tearDown(self):
        dflash_info.create_flashinfer_kv_indices_triton = self._orig_kernel

    def test_qo_indptr_is_per_req_cumsum_when_ragged(self):
        """A ragged layout makes generate_attn_arg_prefill emit an exclusive-cumsum qo_indptr."""
        layout = _fake_layout([6, 3, 1])
        verify_input = _make_verify_input(
            draft_token_num=8, ragged_verify_layout=layout
        )
        bs = 3
        req_pool_indices = torch.arange(bs, dtype=torch.int64)
        paged_kernel_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        paged_kernel_lens_sum = int(paged_kernel_lens.sum())
        req_to_token = torch.zeros((bs, 64), dtype=torch.int32)

        kv_indices, cum_kv_seq_len, qo_indptr, mask = (
            verify_input.generate_attn_arg_prefill(
                req_pool_indices=req_pool_indices,
                paged_kernel_lens=paged_kernel_lens,
                paged_kernel_lens_sum=paged_kernel_lens_sum,
                req_to_token=req_to_token,
            )
        )

        self.assertEqual(qo_indptr.tolist(), [0, 6, 9, 10])
        # kv_indices is sized paged_sum + Σ(1+ℓ_r) = 60 + 10.
        self.assertEqual(kv_indices.numel(), paged_kernel_lens_sum + 10)
        # cum_kv_seq_len adds per-request verify_lens, not a uniform stride.
        self.assertEqual(cum_kv_seq_len.tolist(), [0, 16, 39, 70])
        self.assertIsNone(mask)

    def test_qo_indptr_bit_identical_when_uniform(self):
        """Without a layout, generate_attn_arg_prefill reproduces today's uniform-stride geometry byte-for-byte."""
        draft_token_num = 4
        bs = 3
        req_pool_indices = torch.arange(bs, dtype=torch.int64)
        paged_kernel_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        paged_kernel_lens_sum = int(paged_kernel_lens.sum())
        req_to_token = torch.zeros((bs, 64), dtype=torch.int32)

        verify_input = _make_verify_input(draft_token_num=draft_token_num)
        kv_indices, cum_kv_seq_len, qo_indptr, _ = (
            verify_input.generate_attn_arg_prefill(
                req_pool_indices=req_pool_indices,
                paged_kernel_lens=paged_kernel_lens,
                paged_kernel_lens_sum=paged_kernel_lens_sum,
                req_to_token=req_to_token,
            )
        )

        expected_qo = torch.arange(
            0, (bs + 1) * draft_token_num, step=draft_token_num, dtype=torch.int32
        )
        self.assertTrue(torch.equal(qo_indptr, expected_qo))
        self.assertEqual(
            kv_indices.numel(), paged_kernel_lens_sum + draft_token_num * bs
        )
        expected_cum = torch.zeros((bs + 1,), dtype=torch.int32)
        expected_cum[1:] = torch.cumsum(paged_kernel_lens + draft_token_num, dim=0)
        self.assertTrue(torch.equal(cum_kv_seq_len, expected_cum))

    def test_uniform_layout_matches_no_layout(self):
        """A degenerate verify_lens == [draft_token_num]*bs layout matches the no-layout uniform path."""
        draft_token_num = 4
        bs = 3
        req_pool_indices = torch.arange(bs, dtype=torch.int64)
        paged_kernel_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        paged_kernel_lens_sum = int(paged_kernel_lens.sum())
        req_to_token = torch.zeros((bs, 64), dtype=torch.int32)

        no_layout = _make_verify_input(draft_token_num=draft_token_num)
        _, cum_no, qo_no, _ = no_layout.generate_attn_arg_prefill(
            req_pool_indices=req_pool_indices,
            paged_kernel_lens=paged_kernel_lens,
            paged_kernel_lens_sum=paged_kernel_lens_sum,
            req_to_token=req_to_token,
        )

        uniform_layout = _fake_layout([draft_token_num] * bs)
        with_layout = _make_verify_input(
            draft_token_num=draft_token_num, ragged_verify_layout=uniform_layout
        )
        _, cum_with, qo_with, _ = with_layout.generate_attn_arg_prefill(
            req_pool_indices=req_pool_indices,
            paged_kernel_lens=paged_kernel_lens,
            paged_kernel_lens_sum=paged_kernel_lens_sum,
            req_to_token=req_to_token,
        )

        self.assertTrue(torch.equal(qo_no, qo_with))
        self.assertTrue(torch.equal(cum_no, cum_with))


class TestRaggedVerifyFullModeGate(CustomTestCase):
    def test_disabled_for_non_block_draft_algorithm(self):
        """EAGLE (not block-draft-with-target-kv) never enables ragged-verify-compact mode."""
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            ragged_verify_full_mode_enabled,
        )

        self.assertFalse(ragged_verify_full_mode_enabled(SpeculativeAlgorithm.EAGLE))
        self.assertFalse(ragged_verify_full_mode_enabled(SpeculativeAlgorithm.NONE))

    def test_dspark_gate_follows_infra_env_helper(self):
        """For DSpark the gate defers to the infra ragged_verify_compact_enabled helper."""
        from sglang.srt.model_executor.runner import decode_cuda_graph_runner as runner

        # DSpark is block-draft-with-target-kv, so the result mirrors the infra
        # env helper. When the infra module is absent the gate is False.
        result = runner.ragged_verify_full_mode_enabled(SpeculativeAlgorithm.DSPARK)
        try:
            from sglang.srt.speculative.ragged_verify import (
                ragged_verify_compact_enabled,
            )

            self.assertEqual(result, ragged_verify_compact_enabled())
        except ImportError:
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
