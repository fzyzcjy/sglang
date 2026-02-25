import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_loader import (
    _infer_aux_dims,
    _infer_positions,
    _normalize_step_megatron,
    _normalize_step_sglang,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    MegatronSeqId,
    SGLangSeqId,
    TokenAlignerStepAux,
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

        result: TokenAlignerStepAux = _normalize_step_sglang(step_data, step=0)

        assert torch.equal(result.input_ids, step_data["input_ids"])
        assert torch.equal(result.positions, step_data["positions"])
        assert torch.equal(result.seq_lens, step_data["seq_lens"])
        assert result.seq_ids == (SGLangSeqId(rid="A"),)

    def test_rids_none_fallback(self):
        """Missing rids results in (step, index) fallback seq_ids."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20]),
            "positions": torch.tensor([0, 1]),
            "seq_lens": torch.tensor([2]),
        }

        result: TokenAlignerStepAux = _normalize_step_sglang(step_data, step=3)
        assert result.seq_ids == (MegatronSeqId(step=3, seq_index=0),)

    def test_multiple_seqs_with_rids(self):
        """Multiple sequences with rids."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "positions": torch.tensor([0, 1, 2, 0, 1]),
            "seq_lens": torch.tensor([3, 2]),
            "rids": ["A", "B"],
        }

        result: TokenAlignerStepAux = _normalize_step_sglang(step_data, step=0)
        assert result.seq_ids == (SGLangSeqId(rid="A"), SGLangSeqId(rid="B"))


class TestNormalizeMegatron:
    """Tests for Megatron aux tensor normalization."""

    def test_cu_seqlens_to_seq_lens(self):
        """cu_seqlens_q is converted to seq_lens via differencing."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: TokenAlignerStepAux = _normalize_step_megatron(
            step_data, layout="thd", step=0
        )

        assert torch.equal(result.seq_lens, torch.tensor([3, 2]))

    def test_positions_inferred_thd(self):
        """Positions inferred from seq_lens in thd layout."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: TokenAlignerStepAux = _normalize_step_megatron(
            step_data, layout="thd", step=0
        )

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

        result: TokenAlignerStepAux = _normalize_step_megatron(
            step_data, layout="thd", step=0
        )

        assert torch.equal(result.positions, explicit_positions)

    def test_seq_ids_are_step_index_tuples(self):
        """Megatron seq_ids are (step, seq_index) tuples."""
        step_data: dict = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "cu_seqlens_q": torch.tensor([0, 3, 5]),
        }

        result: TokenAlignerStepAux = _normalize_step_megatron(
            step_data, layout="thd", step=5
        )
        assert result.seq_ids == (
            MegatronSeqId(step=5, seq_index=0),
            MegatronSeqId(step=5, seq_index=1),
        )


class TestInferPositions:
    """Tests for position inference helper."""

    def test_thd_multiple_sequences(self):
        """thd: positions reset to 0 for each sequence."""
        result = _infer_positions(
            seq_lens=torch.tensor([2, 3]),
        )
        assert torch.equal(result, torch.tensor([0, 1, 0, 1, 2]))


class TestInferAuxDims:
    """Tests for _infer_aux_dims inference logic."""

    def _make_meta(self, *, cp_size: int = 1, cp_rank: int = 0) -> dict:
        return {
            "sglang_parallel_info": {
                "tp_rank": 0,
                "tp_size": 1,
                "cp_rank": cp_rank,
                "cp_size": cp_size,
            }
        }

    def test_no_cp_returns_none(self):
        """Without CP parallelism, _infer_aux_dims returns None."""
        metas: list[dict] = [self._make_meta(cp_size=1)]
        result = _infer_aux_dims(name="input_ids", framework="sglang", metas=metas)
        assert result is None

    def test_cp_sharded_sglang_input_ids_raises(self):
        """CP + input_ids in sglang raises NotImplementedError."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        with pytest.raises(NotImplementedError, match="CP-sharded"):
            _infer_aux_dims(name="input_ids", framework="sglang", metas=metas)

    def test_cp_sharded_sglang_positions_raises(self):
        """CP + positions in sglang raises NotImplementedError."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        with pytest.raises(NotImplementedError, match="CP-sharded"):
            _infer_aux_dims(name="positions", framework="sglang", metas=metas)

    def test_cp_sharded_megatron_input_ids_raises(self):
        """CP + input_ids in megatron raises NotImplementedError."""
        metas: list[dict] = [
            {"megatron_parallel_info": {"cp_rank": 0, "cp_size": 2}},
            {"megatron_parallel_info": {"cp_rank": 1, "cp_size": 2}},
        ]
        with pytest.raises(NotImplementedError, match="CP-sharded"):
            _infer_aux_dims(name="input_ids", framework="megatron", metas=metas)

    def test_cp_non_sharded_name_returns_none(self):
        """CP + non-sharded tensor name (seq_lens) returns None."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        result = _infer_aux_dims(name="seq_lens", framework="sglang", metas=metas)
        assert result is None

    def test_unknown_framework_returns_none(self):
        """CP + unknown framework returns None (no match in _CP_SHARDED_AUX_NAMES)."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        result = _infer_aux_dims(name="input_ids", framework="unknown", metas=metas)
        assert result is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
