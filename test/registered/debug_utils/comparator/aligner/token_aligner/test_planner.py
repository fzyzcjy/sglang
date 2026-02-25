import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.indexer import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.planner import (
    compute_token_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    MegatronSeqId,
    SGLangSeqId,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
    TokenAlignerStepAux,
    TokenAlignerGlobalAux,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


class TestBuildTokenIndexSGLangThd:
    """Tests for SGLang thd token index building."""

    def test_single_step_prefill(self):
        """Single prefill step with two sequences."""
        side_aux = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 40, 50]),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    seq_lens=torch.tensor([3, 2]),
                    seq_ids=(SGLangSeqId(rid="A"), SGLangSeqId(rid="B")),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        assert len(index.sequences) == 2

        seq0 = index.sequences[0]
        assert seq0.input_ids == [10, 20, 30]
        assert seq0.positions == [0, 1, 2]
        assert seq0.steps == [0, 0, 0]
        assert seq0.indices == [0, 1, 2]

        seq1 = index.sequences[1]
        assert seq1.input_ids == [40, 50]
        assert seq1.positions == [0, 1]
        assert seq1.indices == [3, 4]

    def test_multi_step_prefill_decode(self):
        """Prefill step followed by decode steps, sequences accumulate tokens."""
        side_aux = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 40, 50]),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    seq_lens=torch.tensor([3, 2]),
                    seq_ids=(SGLangSeqId(rid="A"), SGLangSeqId(rid="B")),
                ),
                1: TokenAlignerStepAux(
                    input_ids=torch.tensor([31, 51]),
                    positions=torch.tensor([3, 2]),
                    seq_lens=torch.tensor([1, 1]),
                    seq_ids=(SGLangSeqId(rid="A"), SGLangSeqId(rid="B")),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        assert len(index.sequences) == 2

        seq0 = index.sequences[0]
        assert seq0.input_ids == [10, 20, 30, 31]
        assert seq0.positions == [0, 1, 2, 3]
        assert seq0.steps == [0, 0, 0, 1]

        seq1 = index.sequences[1]
        assert seq1.input_ids == [40, 50, 51]
        assert seq1.positions == [0, 1, 2]

    def test_sequence_exit_and_join(self):
        """Sequence A exits, new sequence D joins with different seq_id."""
        side_aux = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30]),
                    positions=torch.tensor([0, 1, 2]),
                    seq_lens=torch.tensor([3]),
                    seq_ids=(SGLangSeqId(rid="A"),),
                ),
                1: TokenAlignerStepAux(
                    input_ids=torch.tensor([100, 200]),
                    positions=torch.tensor([0, 1]),
                    seq_lens=torch.tensor([2]),
                    seq_ids=(SGLangSeqId(rid="D"),),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        assert len(index.sequences) == 2

    def test_different_seq_ids_produce_separate_sequences(self):
        """Different seq_ids at different steps → separate sequences."""
        side_aux = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20]),
                    positions=torch.tensor([0, 1]),
                    seq_lens=torch.tensor([2]),
                    seq_ids=(SGLangSeqId(rid="A"),),
                ),
                1: TokenAlignerStepAux(
                    input_ids=torch.tensor([100, 200, 300]),
                    positions=torch.tensor([0, 1, 2]),
                    seq_lens=torch.tensor([3]),
                    seq_ids=(SGLangSeqId(rid="D"),),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        assert len(index.sequences) == 2

        all_input_ids = {
            seq_id: rec.input_ids for seq_id, rec in index.sequences.items()
        }
        assert [10, 20] in all_input_ids.values()
        assert [100, 200, 300] in all_input_ids.values()


class TestBuildTokenIndexMegatronThd:
    """Tests for Megatron thd token index building."""

    def test_single_step_two_sequences(self):
        """Single step with two sequences in thd layout."""
        side_aux = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 40, 50]),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    seq_lens=torch.tensor([3, 2]),
                    seq_ids=(MegatronSeqId(step=0, seq_index=0), MegatronSeqId(step=0, seq_index=1)),
                ),
            },
            framework="megatron",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        assert len(index.sequences) == 2

        seq0 = index.sequences[0]
        assert seq0.input_ids == [10, 20, 30]
        assert seq0.positions == [0, 1, 2]
        assert seq0.steps == [0, 0, 0]
        assert seq0.indices == [0, 1, 2]

        seq1 = index.sequences[1]
        assert seq1.input_ids == [40, 50]
        assert seq1.positions == [0, 1]
        assert seq1.indices == [3, 4]

    def test_multi_step_accumulation(self):
        """Two steps with different seq_ids produce separate sequences."""
        side_aux = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 40]),
                    positions=torch.tensor([0, 1, 0, 1]),
                    seq_lens=torch.tensor([2, 2]),
                    seq_ids=(MegatronSeqId(step=0, seq_index=0), MegatronSeqId(step=0, seq_index=1)),
                ),
                1: TokenAlignerStepAux(
                    input_ids=torch.tensor([50, 60, 70, 80]),
                    positions=torch.tensor([0, 1, 0, 1]),
                    seq_lens=torch.tensor([2, 2]),
                    seq_ids=(MegatronSeqId(step=1, seq_index=0), MegatronSeqId(step=1, seq_index=1)),
                ),
            },
            framework="megatron",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        assert len(index.sequences) == 4

        seq0 = index.sequences[0]
        assert seq0.input_ids == [10, 20]
        assert seq0.steps == [0, 0]

        seq2 = index.sequences[2]
        assert seq2.input_ids == [50, 60]
        assert seq2.steps == [1, 1]


class TestMatchSequences:
    """Tests for sequence matching logic."""

    def test_exact_match(self):
        """Two sides with identical input_ids match exactly."""
        index_a = _make_index(
            sequences={0: (10, 20, 30), 1: (40, 50)},
        )
        index_b = _make_index(
            sequences={0: (10, 20, 30), 1: (40, 50)},
        )

        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index_a, y=index_b))
        assert len(plan.match_steps.x) == 5

    def test_different_order(self):
        """Sequences in different order still match by input_ids content."""
        index_a = _make_index(
            sequences={0: (10, 20, 30), 1: (40, 50)},
        )
        index_b = _make_index(
            sequences={0: (40, 50), 1: (10, 20, 30)},
        )

        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index_a, y=index_b))
        assert len(plan.match_steps.x) == 5

    def test_prefix_match(self):
        """A-side has shorter sequence (prefix of B-side), still matches."""
        index_a = _make_index(
            sequences={0: (10, 20)},
        )
        index_b = _make_index(
            sequences={0: (10, 20, 30)},
        )

        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index_a, y=index_b))
        assert len(plan.match_steps.x) == 2

    def test_no_match(self):
        """Completely different sequences produce no matches."""
        index_a = _make_index(
            sequences={0: (10, 20)},
        )
        index_b = _make_index(
            sequences={0: (99, 88)},
        )

        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index_a, y=index_b))
        assert len(plan.match_steps.x) == 0

    def test_ambiguous_same_input_ids(self):
        """Two sequences with identical input_ids: greedy match, no error."""
        index_a = _make_index(
            sequences={0: (10, 20), 1: (10, 20)},
        )
        index_b = _make_index(
            sequences={0: (10, 20), 1: (10, 20)},
        )

        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index_a, y=index_b))
        assert len(plan.match_steps.x) == 4


class TestComputeAlignmentPlanCrossLayout:
    """Tests for alignment plan across different step distributions."""

    def test_thd_vs_thd_different_step_splits(self):
        """Two thd sides with same tokens but different step distributions."""
        side_aux_a = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20]),
                    positions=torch.tensor([0, 1]),
                    seq_lens=torch.tensor([2]),
                    seq_ids=(SGLangSeqId(rid="X"),),
                ),
                1: TokenAlignerStepAux(
                    input_ids=torch.tensor([30]),
                    positions=torch.tensor([2]),
                    seq_lens=torch.tensor([1]),
                    seq_ids=(SGLangSeqId(rid="X"),),
                ),
            },
            framework="sglang",
            layout="thd",
        )
        side_aux_b = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30]),
                    positions=torch.tensor([0, 1, 2]),
                    seq_lens=torch.tensor([3]),
                    seq_ids=(SGLangSeqId(rid="X"),),
                ),
            },
            framework="sglang",
            layout="thd",
        )

        index_a = build_seqs_info(side_aux_a)
        index_b = build_seqs_info(side_aux_b)

        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index_a, y=index_b))
        assert len(plan.match_steps.x) == 3

    def test_sglang_vs_megatron_thd(self):
        """SGLang multi-step thd aligned with Megatron single-step thd."""
        side_aux_a = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 40, 50]),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    seq_lens=torch.tensor([3, 2]),
                    seq_ids=(SGLangSeqId(rid="A"), SGLangSeqId(rid="B")),
                ),
                1: TokenAlignerStepAux(
                    input_ids=torch.tensor([31, 51]),
                    positions=torch.tensor([3, 2]),
                    seq_lens=torch.tensor([1, 1]),
                    seq_ids=(SGLangSeqId(rid="A"), SGLangSeqId(rid="B")),
                ),
            },
            framework="sglang",
            layout="thd",
        )
        side_aux_b = TokenAlignerGlobalAux(
            steps={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 31, 40, 50, 51]),
                    positions=torch.tensor([0, 1, 2, 3, 0, 1, 2]),
                    seq_lens=torch.tensor([4, 3]),
                    seq_ids=(MegatronSeqId(step=0, seq_index=0), MegatronSeqId(step=0, seq_index=1)),
                ),
            },
            framework="megatron",
            layout="thd",
        )

        index_a = build_seqs_info(side_aux_a)
        index_b = build_seqs_info(side_aux_b)

        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index_a, y=index_b))

        assert len(plan.match_steps.x) == 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(
    *,
    sequences: dict[int, tuple[int, ...]],
    layout: str = "thd",
) -> TokenAlignerSeqsInfo:
    """Create a TokenAlignerSeqsInfo from simplified input_ids-only specification."""
    records: dict[int, TokenAlignerSeqInfo] = {}
    for seq_id, input_ids in sequences.items():
        num_tokens = len(input_ids)
        records[seq_id] = TokenAlignerSeqInfo(
            input_ids=list(input_ids),
            positions=list(range(num_tokens)),
            steps=[0] * num_tokens,
            indices=list(range(num_tokens)),
        )
    return TokenAlignerSeqsInfo(sequences=records, layout=layout)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
