import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    AuxTensorsForStep,
    SideAux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.planner import (
    build_token_index,
    compute_alignment_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    SequenceRecord,
    SideTokenIndex,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


class TestBuildTokenIndexSGLangThd:
    """Tests for SGLang thd token index building."""

    def test_single_step_prefill(self):
        """Single prefill step with two sequences."""
        side_aux = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([10, 20, 30, 40, 50]),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    seq_lens=torch.tensor([3, 2]),
                    req_pool_indices=torch.tensor([7, 3]),
                    rids=("A", "B"),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_token_index(side_aux)
        assert len(index.sequences) == 2

        seq0 = index.sequences[0]
        assert seq0.input_ids == (10, 20, 30)
        assert seq0.positions == (0, 1, 2)
        assert seq0.steps == (0, 0, 0)
        assert seq0.indices == (0, 1, 2)

        seq1 = index.sequences[1]
        assert seq1.input_ids == (40, 50)
        assert seq1.positions == (0, 1)
        assert seq1.indices == (3, 4)

    def test_multi_step_prefill_decode(self):
        """Prefill step followed by decode steps, sequences accumulate tokens."""
        side_aux = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([10, 20, 30, 40, 50]),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    seq_lens=torch.tensor([3, 2]),
                    req_pool_indices=torch.tensor([7, 3]),
                    rids=("A", "B"),
                ),
                1: AuxTensorsForStep(
                    input_ids=torch.tensor([31, 51]),
                    positions=torch.tensor([3, 2]),
                    seq_lens=torch.tensor([1, 1]),
                    req_pool_indices=torch.tensor([7, 3]),
                    rids=("A", "B"),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_token_index(side_aux)
        assert len(index.sequences) == 2

        seq0 = index.sequences[0]
        assert seq0.input_ids == (10, 20, 30, 31)
        assert seq0.positions == (0, 1, 2, 3)
        assert seq0.steps == (0, 0, 0, 1)

        seq1 = index.sequences[1]
        assert seq1.input_ids == (40, 50, 51)
        assert seq1.positions == (0, 1, 2)

    def test_sequence_exit_and_join(self):
        """Sequence A exits, new sequence D joins with different rpi."""
        side_aux = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([10, 20, 30]),
                    positions=torch.tensor([0, 1, 2]),
                    seq_lens=torch.tensor([3]),
                    req_pool_indices=torch.tensor([7]),
                    rids=("A",),
                ),
                1: AuxTensorsForStep(
                    input_ids=torch.tensor([100, 200]),
                    positions=torch.tensor([0, 1]),
                    seq_lens=torch.tensor([2]),
                    req_pool_indices=torch.tensor([12]),
                    rids=("D",),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_token_index(side_aux)
        assert len(index.sequences) == 2

    def test_slot_reuse_detection(self):
        """Same rpi but different rid → slot reuse → two separate sequences."""
        side_aux = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([10, 20]),
                    positions=torch.tensor([0, 1]),
                    seq_lens=torch.tensor([2]),
                    req_pool_indices=torch.tensor([7]),
                    rids=("A",),
                ),
                1: AuxTensorsForStep(
                    input_ids=torch.tensor([100, 200, 300]),
                    positions=torch.tensor([0, 1, 2]),
                    seq_lens=torch.tensor([3]),
                    req_pool_indices=torch.tensor([7]),
                    rids=("D",),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_token_index(side_aux)
        assert len(index.sequences) == 2

        all_input_ids = {
            seq_id: rec.input_ids for seq_id, rec in index.sequences.items()
        }
        assert (10, 20) in all_input_ids.values()
        assert (100, 200, 300) in all_input_ids.values()


class TestBuildTokenIndexMegatronBshd:
    """Tests for Megatron bshd token index building."""

    def test_single_step_no_padding(self):
        """Single step, B=2, S=3, no padding."""
        side_aux = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([[10, 20, 30], [40, 50, 60]]),
                    positions=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                    seq_lens=torch.tensor([3, 3]),
                    req_pool_indices=None,
                    rids=None,
                ),
            },
            framework="megatron",
            layout="bshd",
        )

        index = build_token_index(side_aux)
        assert len(index.sequences) == 2

        seq0 = index.sequences[0]
        assert seq0.input_ids == (10, 20, 30)
        assert seq0.indices == (0, 1, 2)

        seq1 = index.sequences[1]
        assert seq1.input_ids == (40, 50, 60)
        assert seq1.indices == (3, 4, 5)

    def test_single_step_with_padding(self):
        """Padding tokens excluded: seq_lens=[2,1] but S=3."""
        side_aux = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([[10, 20, 0], [40, 0, 0]]),
                    positions=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                    seq_lens=torch.tensor([2, 1]),
                    req_pool_indices=None,
                    rids=None,
                ),
            },
            framework="megatron",
            layout="bshd",
        )

        index = build_token_index(side_aux)
        assert len(index.sequences) == 2

        seq0 = index.sequences[0]
        assert len(seq0.positions) == 2

        seq1 = index.sequences[1]
        assert len(seq1.positions) == 1


class TestMatchSequences:
    """Tests for sequence matching logic."""

    def test_exact_match(self):
        """Two sides with identical input_ids match exactly."""
        index_a = _make_index(
            sequences={0: (10, 20, 30), 1: (40, 50)},
            framework="sglang",
        )
        index_b = _make_index(
            sequences={0: (10, 20, 30), 1: (40, 50)},
            framework="megatron",
        )

        plan = compute_alignment_plan(index_a=index_a, index_b=index_b)
        assert plan.summary.num_matched_tokens == 5
        assert len(plan.summary.sequence_matches) == 2

    def test_different_order(self):
        """Sequences in different order still match by input_ids content."""
        index_a = _make_index(
            sequences={0: (10, 20, 30), 1: (40, 50)},
            framework="sglang",
        )
        index_b = _make_index(
            sequences={0: (40, 50), 1: (10, 20, 30)},
            framework="megatron",
        )

        plan = compute_alignment_plan(index_a=index_a, index_b=index_b)
        assert plan.summary.num_matched_tokens == 5

    def test_prefix_match(self):
        """A-side has shorter sequence (prefix of B-side), still matches."""
        index_a = _make_index(
            sequences={0: (10, 20)},
            framework="sglang",
        )
        index_b = _make_index(
            sequences={0: (10, 20, 30)},
            framework="megatron",
        )

        plan = compute_alignment_plan(index_a=index_a, index_b=index_b)
        assert plan.summary.num_matched_tokens == 2
        assert plan.summary.sequence_matches[0].num_matched == 2

    def test_no_match(self):
        """Completely different sequences produce no matches."""
        index_a = _make_index(
            sequences={0: (10, 20)},
            framework="sglang",
        )
        index_b = _make_index(
            sequences={0: (99, 88)},
            framework="megatron",
        )

        plan = compute_alignment_plan(index_a=index_a, index_b=index_b)
        assert plan.summary.num_matched_tokens == 0
        assert len(plan.summary.unmatched_seq_ids_a) == 1
        assert len(plan.summary.unmatched_seq_ids_b) == 1

    def test_ambiguous_same_input_ids(self):
        """Two sequences with identical input_ids: greedy match, no error."""
        index_a = _make_index(
            sequences={0: (10, 20), 1: (10, 20)},
            framework="sglang",
        )
        index_b = _make_index(
            sequences={0: (10, 20), 1: (10, 20)},
            framework="megatron",
        )

        plan = compute_alignment_plan(index_a=index_a, index_b=index_b)
        assert plan.summary.num_matched_tokens == 4
        assert len(plan.summary.sequence_matches) == 2


class TestComputeAlignmentPlanCrossLayout:
    """Tests for alignment plan across different layouts/step distributions."""

    def test_thd_multi_step_vs_bshd_single_step(self):
        """SGLang multi-step thd aligned with Megatron single-step bshd."""
        side_aux_a = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([10, 20, 30, 40, 50]),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    seq_lens=torch.tensor([3, 2]),
                    req_pool_indices=torch.tensor([7, 3]),
                    rids=("A", "B"),
                ),
                1: AuxTensorsForStep(
                    input_ids=torch.tensor([31, 51]),
                    positions=torch.tensor([3, 2]),
                    seq_lens=torch.tensor([1, 1]),
                    req_pool_indices=torch.tensor([7, 3]),
                    rids=("A", "B"),
                ),
            },
            framework="sglang",
            layout="thd",
        )
        side_aux_b = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([[10, 20, 30, 31], [40, 50, 51, 0]]),
                    positions=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
                    seq_lens=torch.tensor([4, 3]),
                    req_pool_indices=None,
                    rids=None,
                ),
            },
            framework="megatron",
            layout="bshd",
        )

        index_a = build_token_index(side_aux_a)
        index_b = build_token_index(side_aux_b)

        plan = compute_alignment_plan(index_a=index_a, index_b=index_b)

        assert plan.summary.num_matched_tokens == 7
        assert len(plan.summary.sequence_matches) == 2

    def test_thd_vs_thd_different_step_splits(self):
        """Two thd sides with same tokens but different step distributions."""
        side_aux_a = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([10, 20]),
                    positions=torch.tensor([0, 1]),
                    seq_lens=torch.tensor([2]),
                    req_pool_indices=torch.tensor([5]),
                    rids=("X",),
                ),
                1: AuxTensorsForStep(
                    input_ids=torch.tensor([30]),
                    positions=torch.tensor([2]),
                    seq_lens=torch.tensor([1]),
                    req_pool_indices=torch.tensor([5]),
                    rids=("X",),
                ),
            },
            framework="sglang",
            layout="thd",
        )
        side_aux_b = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([10, 20, 30]),
                    positions=torch.tensor([0, 1, 2]),
                    seq_lens=torch.tensor([3]),
                    req_pool_indices=torch.tensor([9]),
                    rids=("X",),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index_a = build_token_index(side_aux_a)
        index_b = build_token_index(side_aux_b)

        plan = compute_alignment_plan(index_a=index_a, index_b=index_b)
        assert plan.summary.num_matched_tokens == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(
    *,
    sequences: dict[int, tuple[int, ...]],
    framework: str = "sglang",
    layout: str = "thd",
) -> SideTokenIndex:
    """Create a SideTokenIndex from simplified input_ids-only specification."""
    records: dict[int, SequenceRecord] = {}
    for seq_id, input_ids in sequences.items():
        num_tokens = len(input_ids)
        records[seq_id] = SequenceRecord(
            input_ids=input_ids,
            positions=tuple(range(num_tokens)),
            steps=tuple([0] * num_tokens),
            indices=tuple(range(num_tokens)),
        )
    return SideTokenIndex(sequences=records, framework=framework, layout=layout)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
