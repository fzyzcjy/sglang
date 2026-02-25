import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    AuxTensorsForStep,
    _infer_positions,
    _normalize_megatron,
    _normalize_sglang,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestNormalizeSGLang:
    """Tests for SGLang aux tensor normalization."""

    def test_passthrough(self):
        """SGLang tensors pass through without transformation."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30]),
            "positions": torch.tensor([0, 1, 2]),
            "seq_lens": torch.tensor([3]),
            "req_pool_indices": torch.tensor([7]),
            "rids": ["A"],
        }

        result: AuxTensorsForStep = _normalize_sglang(step_data)

        assert torch.equal(result.input_ids, step_data["input_ids"])
        assert torch.equal(result.positions, step_data["positions"])
        assert torch.equal(result.seq_lens, step_data["seq_lens"])
        assert torch.equal(result.req_pool_indices, step_data["req_pool_indices"])
        assert result.rids == ("A",)

    def test_rids_none(self):
        """Missing rids results in None."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20]),
            "positions": torch.tensor([0, 1]),
            "seq_lens": torch.tensor([2]),
        }

        result: AuxTensorsForStep = _normalize_sglang(step_data)
        assert result.rids is None
        assert result.req_pool_indices is None


class TestNormalizeMegatron:
    """Tests for Megatron aux tensor normalization."""

    def test_cu_seqlens_to_seq_lens(self):
        """cu_seqlens_q is converted to seq_lens via differencing."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: AuxTensorsForStep = _normalize_megatron(step_data, layout="thd")

        assert torch.equal(result.seq_lens, torch.tensor([3, 2]))

    def test_positions_inferred_thd(self):
        """Positions inferred from seq_lens in thd layout."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: AuxTensorsForStep = _normalize_megatron(step_data, layout="thd")

        expected_positions = torch.tensor([0, 1, 2, 0, 1])
        assert torch.equal(result.positions, expected_positions)

    def test_positions_inferred_bshd(self):
        """Positions inferred in bshd layout: [B, S] with arange per row."""
        step_data: dict = {
            "input_ids": torch.tensor([[10, 20, 30], [40, 50, 60]]),
            "cu_seqlens_q": torch.tensor([0, 3, 6]),
        }

        result: AuxTensorsForStep = _normalize_megatron(step_data, layout="bshd")

        expected_positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
        assert torch.equal(result.positions, expected_positions)

    def test_position_ids_passthrough(self):
        """Explicit position_ids used directly instead of inference."""
        explicit_positions = torch.tensor([5, 6, 7, 8, 9])
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "position_ids": explicit_positions,
            "cu_seqlens_q": torch.tensor([0, 5]),
        }

        result: AuxTensorsForStep = _normalize_megatron(step_data, layout="thd")

        assert torch.equal(result.positions, explicit_positions)


class TestInferPositions:
    """Tests for position inference helper."""

    def test_thd_multiple_sequences(self):
        """thd: positions reset to 0 for each sequence."""
        result = _infer_positions(
            seq_lens=torch.tensor([2, 3]),
            input_ids=torch.tensor([10, 20, 30, 40, 50]),
            layout="thd",
        )
        assert torch.equal(result, torch.tensor([0, 1, 0, 1, 2]))

    def test_bshd_broadcast(self):
        """bshd: positions are 0..S-1 broadcast across batch."""
        result = _infer_positions(
            seq_lens=torch.tensor([3, 3]),
            input_ids=torch.tensor([[10, 20, 30], [40, 50, 60]]),
            layout="bshd",
        )
        expected = torch.tensor([[0, 1, 2], [0, 1, 2]])
        assert torch.equal(result, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
