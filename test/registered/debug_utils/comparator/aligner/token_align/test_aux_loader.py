import sys
from pathlib import Path

import polars as pl
import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    _infer_positions,
    _normalize_megatron,
    _normalize_sglang,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    StepAux,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestNormalizeSGLang:
    """Tests for SGLang aux tensor normalization."""

    def test_with_rids(self):
        """SGLang tensors with rids produce string seq_ids."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30]),
            "positions": torch.tensor([0, 1, 2]),
            "seq_lens": torch.tensor([3]),
            "rids": ["A"],
        }

        result: StepAux = _normalize_sglang(step_data, step=0)

        assert torch.equal(result.input_ids, step_data["input_ids"])
        assert torch.equal(result.positions, step_data["positions"])
        assert torch.equal(result.seq_lens, step_data["seq_lens"])
        assert result.seq_ids == ("A",)

    def test_rids_none_fallback(self):
        """Missing rids results in (step, index) fallback seq_ids."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20]),
            "positions": torch.tensor([0, 1]),
            "seq_lens": torch.tensor([2]),
        }

        result: StepAux = _normalize_sglang(step_data, step=3)
        assert result.seq_ids == ((3, 0),)

    def test_multiple_seqs_with_rids(self):
        """Multiple sequences with rids."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "positions": torch.tensor([0, 1, 2, 0, 1]),
            "seq_lens": torch.tensor([3, 2]),
            "rids": ["A", "B"],
        }

        result: StepAux = _normalize_sglang(step_data, step=0)
        assert result.seq_ids == ("A", "B")


class TestNormalizeMegatron:
    """Tests for Megatron aux tensor normalization."""

    def test_cu_seqlens_to_seq_lens(self):
        """cu_seqlens_q is converted to seq_lens via differencing."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: StepAux = _normalize_megatron(step_data, layout="thd", step=0)

        assert torch.equal(result.seq_lens, torch.tensor([3, 2]))

    def test_positions_inferred_thd(self):
        """Positions inferred from seq_lens in thd layout."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: StepAux = _normalize_megatron(step_data, layout="thd", step=0)

        expected_positions = torch.tensor([0, 1, 2, 0, 1])
        assert torch.equal(result.positions, expected_positions)

    def test_position_ids_passthrough(self):
        """Explicit position_ids used directly instead of inference."""
        explicit_positions = torch.tensor([5, 6, 7, 8, 9])
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "position_ids": explicit_positions,
            "cu_seqlens_q": torch.tensor([0, 5]),
        }

        result: StepAux = _normalize_megatron(step_data, layout="thd", step=0)

        assert torch.equal(result.positions, explicit_positions)

    def test_seq_ids_are_step_index_tuples(self):
        """Megatron seq_ids are (step, seq_index) tuples."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: StepAux = _normalize_megatron(step_data, layout="thd", step=5)
        assert result.seq_ids == ((5, 0), (5, 1))


class TestInferPositions:
    """Tests for position inference helper."""

    def test_thd_multiple_sequences(self):
        """thd: positions reset to 0 for each sequence."""
        result = _infer_positions(
            seq_lens=torch.tensor([2, 3]),
        )
        assert torch.equal(result, torch.tensor([0, 1, 0, 1, 2]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
