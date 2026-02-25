import sys

import pytest
from pydantic import ValidationError

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    PositionalSeqId,
    TokenAlignerPlan,
    TokenAlignerSeqInfo,
    TokenAlignerStepAux,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.output_types import SummaryRecord
from sglang.srt.debug_utils.comparator.utils import Pair, _check_equal_lengths
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestCheckEqualLengths:
    def test_all_equal(self):
        _check_equal_lengths(a=[1, 2], b=[3, 4])

    def test_empty_lists(self):
        _check_equal_lengths(a=[], b=[])

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            _check_equal_lengths(a=[1, 2], b=[3])


class TestTokenAlignerStepAux:
    def test_valid(self):
        aux = TokenAlignerStepAux(
            input_ids=[10, 20, 30],
            positions=[0, 1, 2],
            seq_lens=[2, 1],
            seq_ids=[PositionalSeqId(step=0, seq_index=0), PositionalSeqId(step=0, seq_index=1)],
        )
        assert len(aux.input_ids) == 3

    def test_token_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            TokenAlignerStepAux(
                input_ids=[10, 20, 30],
                positions=[0, 1],
                seq_lens=[2, 1],
                seq_ids=[PositionalSeqId(step=0, seq_index=0), PositionalSeqId(step=0, seq_index=1)],
            )

    def test_seq_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            TokenAlignerStepAux(
                input_ids=[10, 20, 30],
                positions=[0, 1, 2],
                seq_lens=[2, 1],
                seq_ids=[PositionalSeqId(step=0, seq_index=0)],
            )

    def test_sum_seq_lens_mismatch(self):
        with pytest.raises(ValueError, match="sum\\(seq_lens\\)"):
            TokenAlignerStepAux(
                input_ids=[10, 20, 30],
                positions=[0, 1, 2],
                seq_lens=[1, 1],
                seq_ids=[PositionalSeqId(step=0, seq_index=0), PositionalSeqId(step=0, seq_index=1)],
            )


class TestTokenAlignerSeqInfo:
    def test_valid(self):
        info = TokenAlignerSeqInfo(
            input_ids=[10, 20, 30],
            positions=[0, 1, 2],
            locator=TokenLocator(steps=[0, 0, 1], token_index_in_step=[0, 1, 0]),
        )
        assert len(info.input_ids) == 3

    def test_length_mismatch(self):
        with pytest.raises(ValidationError):
            TokenAlignerSeqInfo(
                input_ids=[10, 20, 30],
                positions=[0, 1, 2],
                locator=TokenLocator(steps=[0, 0], token_index_in_step=[0, 1, 0]),
            )

    def test_positions_not_sequential(self):
        with pytest.raises(ValidationError, match="positions must be"):
            TokenAlignerSeqInfo(
                input_ids=[10, 20, 30],
                positions=[0, 2, 1],
                locator=TokenLocator(steps=[0, 0, 1], token_index_in_step=[0, 1, 0]),
            )


class TestTokenAlignerPlan:
    def test_valid(self):
        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[0, 0, 1], token_index_in_step=[0, 1, 0]),
                y=TokenLocator(steps=[0, 1, 1], token_index_in_step=[0, 0, 1]),
            ),
        )
        assert len(plan.locators.x.steps) == 3

    def test_length_mismatch(self):
        with pytest.raises(ValidationError, match="Length mismatch"):
            TokenAlignerPlan(
                locators=Pair(
                    x=TokenLocator(steps=[0, 0], token_index_in_step=[0, 1]),
                    y=TokenLocator(steps=[0, 1, 1], token_index_in_step=[0, 0, 1]),
                ),
            )


class TestSummaryRecord:
    def test_valid(self):
        record = SummaryRecord(total=10, passed=7, failed=2, skipped=1)
        assert record.total == 10

    def test_total_mismatch(self):
        with pytest.raises(ValidationError, match="total=10"):
            SummaryRecord(total=10, passed=5, failed=2, skipped=1)


class TestAxisInfo:
    def test_valid(self):
        info = AxisInfo(axis_rank=0, axis_size=4)
        assert info.axis_rank == 0

    def test_axis_size_zero(self):
        with pytest.raises(ValidationError, match="axis_size must be > 0"):
            AxisInfo(axis_rank=0, axis_size=0)

    def test_axis_size_negative(self):
        with pytest.raises(ValidationError, match="axis_size must be > 0"):
            AxisInfo(axis_rank=0, axis_size=-1)

    def test_axis_rank_negative(self):
        with pytest.raises(ValidationError, match="axis_rank must be in"):
            AxisInfo(axis_rank=-1, axis_size=4)

    def test_axis_rank_too_large(self):
        with pytest.raises(ValidationError, match="axis_rank must be in"):
            AxisInfo(axis_rank=4, axis_size=4)

    def test_axis_rank_equals_size_minus_one(self):
        info = AxisInfo(axis_rank=3, axis_size=4)
        assert info.axis_rank == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
