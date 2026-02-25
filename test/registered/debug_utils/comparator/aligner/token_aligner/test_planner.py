import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.planner import (
    _match_sequences,
    compute_token_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.seq_info_builder import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    MegatronSeqId,
    SGLangSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerSeqInfo,
    TokenAlignerSeqsInfo,
    TokenAlignerStepAux,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


class TestBuildTokenIndexSGLangThd:
    """Tests for SGLang thd token index building."""

    def test_single_step_prefill(self):
        """Single prefill step with two sequences."""
        side_aux = TokenAlignerGlobalAux(
            step_auxs={
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
            step_auxs={
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
            step_auxs={
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
            step_auxs={
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
            step_auxs={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 40, 50]),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    seq_lens=torch.tensor([3, 2]),
                    seq_ids=(
                        MegatronSeqId(step=0, seq_index=0),
                        MegatronSeqId(step=0, seq_index=1),
                    ),
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
            step_auxs={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 40]),
                    positions=torch.tensor([0, 1, 0, 1]),
                    seq_lens=torch.tensor([2, 2]),
                    seq_ids=(
                        MegatronSeqId(step=0, seq_index=0),
                        MegatronSeqId(step=0, seq_index=1),
                    ),
                ),
                1: TokenAlignerStepAux(
                    input_ids=torch.tensor([50, 60, 70, 80]),
                    positions=torch.tensor([0, 1, 0, 1]),
                    seq_lens=torch.tensor([2, 2]),
                    seq_ids=(
                        MegatronSeqId(step=1, seq_index=0),
                        MegatronSeqId(step=1, seq_index=1),
                    ),
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
    """Tests for _match_sequences: for each y, find matching x."""

    def test_exact_match_simple(self):
        """Identical input_ids on both sides → all matched."""
        matched = _match_seqs(
            x={0: (10, 20, 30), 1: (40, 50)},
            y={0: (10, 20, 30), 1: (40, 50)},
        )
        assert _matched_ids(matched) == {(0, 0), (1, 1)}

    def test_exact_match_different_order(self):
        """Sequences in different order still match by content."""
        matched = _match_seqs(
            x={0: (10, 20), 1: (40, 50)},
            y={0: (40, 50), 1: (10, 20)},
        )
        assert _matched_ids(matched) == {(1, 0), (0, 1)}

    def test_exact_match_different_seq_ids(self):
        """Seq IDs don't need to correspond — matching is by content."""
        matched = _match_seqs(
            x={5: (10, 20), 9: (30, 40)},
            y={2: (30, 40), 7: (10, 20)},
        )
        assert _matched_ids(matched) == {(9, 2), (5, 7)}

    def test_no_match(self):
        """Completely different input_ids → no matches."""
        matched = _match_seqs(
            x={0: (10, 20)},
            y={0: (99, 88)},
        )
        assert matched == []

    def test_empty_sides(self):
        """Empty x or y → no matches."""
        assert _match_seqs(x={}, y={0: (10,)}) == []
        assert _match_seqs(x={0: (10,)}, y={}) == []
        assert _match_seqs(x={}, y={}) == []

    def test_x_has_more_sequences(self):
        """Extra x sequences are ignored (no y needs them)."""
        matched = _match_seqs(
            x={0: (10, 20), 1: (30, 40), 2: (50, 60)},
            y={0: (30, 40)},
        )
        assert _matched_ids(matched) == {(1, 0)}

    def test_y_has_more_sequences(self):
        """Extra y sequences remain unmatched."""
        matched = _match_seqs(
            x={0: (10, 20)},
            y={0: (10, 20), 1: (30, 40), 2: (50, 60)},
        )
        assert _matched_ids(matched) == {(0, 0)}

    def test_one_x_not_reused(self):
        """Each x can only be claimed once, even if multiple y want it."""
        matched = _match_seqs(
            x={0: (10, 20)},
            y={0: (10, 20), 1: (10, 20)},
        )
        assert len(matched) == 1

    def test_ambiguous_all_matched(self):
        """Multiple identical sequences on both sides → all paired (greedy 1:1)."""
        matched = _match_seqs(
            x={0: (10, 20), 1: (10, 20), 2: (10, 20)},
            y={0: (10, 20), 1: (10, 20), 2: (10, 20)},
        )
        assert len(matched) == 3
        x_ids = {m[0] for m in matched}
        y_ids = {m[1] for m in matched}
        assert x_ids == {0, 1, 2}
        assert y_ids == {0, 1, 2}

    def test_prefix_x_shorter(self):
        """x has fewer tokens (prefix of y) → prefix match."""
        matched = _match_seqs(
            x={0: (10, 20)},
            y={0: (10, 20, 30)},
        )
        assert _matched_ids(matched) == {(0, 0)}

    def test_prefix_y_shorter(self):
        """y has fewer tokens (prefix of x) → prefix match."""
        matched = _match_seqs(
            x={0: (10, 20, 30)},
            y={0: (10, 20)},
        )
        assert _matched_ids(matched) == {(0, 0)}

    def test_prefix_picks_longest(self):
        """Among multiple prefix candidates, picks the one with longest overlap."""
        matched = _match_seqs(
            x={0: (10,), 1: (10, 20, 30)},
            y={0: (10, 20, 30, 40)},
        )
        assert _matched_ids(matched) == {(1, 0)}

    def test_exact_preferred_over_prefix(self):
        """Exact match is tried first, even if a longer prefix candidate exists."""
        matched = _match_seqs(
            x={0: (10, 20), 1: (10, 20, 30)},
            y={0: (10, 20)},
        )
        assert _matched_ids(matched) == {(0, 0)}

    def test_prefix_fallback_after_exact(self):
        """Exact matches consume sequences, remaining use prefix match."""
        matched = _match_seqs(
            x={0: (10, 20, 30), 1: (40, 50)},
            y={0: (10, 20, 30), 1: (40, 50, 60)},
        )
        assert len(matched) == 2
        matched_set = _matched_ids(matched)
        assert (0, 0) in matched_set
        assert (1, 1) in matched_set

    def test_single_token_sequences(self):
        """Single-token sequences can match."""
        matched = _match_seqs(
            x={0: (42,)},
            y={0: (42,)},
        )
        assert _matched_ids(matched) == {(0, 0)}

    def test_no_partial_overlap_without_prefix(self):
        """Overlapping content that isn't a prefix → no match."""
        matched = _match_seqs(
            x={0: (10, 20, 30)},
            y={0: (20, 30, 40)},
        )
        assert matched == []


class TestComputeAlignmentPlanCrossLayout:
    """Tests for alignment plan across different step distributions."""

    def test_thd_vs_thd_different_step_splits(self):
        """Two thd sides with same tokens but different step distributions."""
        side_aux_a = TokenAlignerGlobalAux(
            step_auxs={
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
            step_auxs={
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
            step_auxs={
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
            step_auxs={
                0: TokenAlignerStepAux(
                    input_ids=torch.tensor([10, 20, 30, 31, 40, 50, 51]),
                    positions=torch.tensor([0, 1, 2, 3, 0, 1, 2]),
                    seq_lens=torch.tensor([4, 3]),
                    seq_ids=(
                        MegatronSeqId(step=0, seq_index=0),
                        MegatronSeqId(step=0, seq_index=1),
                    ),
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


def _make_seq_info_dict(
    sequences: dict[int, tuple[int, ...]],
) -> dict[int, TokenAlignerSeqInfo]:
    """Create a dict of TokenAlignerSeqInfo from {seq_id: input_ids_tuple}."""
    result: dict[int, TokenAlignerSeqInfo] = {}
    for seq_id, input_ids in sequences.items():
        num_tokens = len(input_ids)
        result[seq_id] = TokenAlignerSeqInfo(
            input_ids=list(input_ids),
            positions=list(range(num_tokens)),
            steps=[0] * num_tokens,
            indices=list(range(num_tokens)),
        )
    return result


def _match_seqs(
    *,
    x: dict[int, tuple[int, ...]],
    y: dict[int, tuple[int, ...]],
) -> list[tuple[int, int]]:
    """Shorthand: build SeqInfo dicts and call _match_sequences."""
    return _match_sequences(
        seqs=Pair(x=_make_seq_info_dict(x), y=_make_seq_info_dict(y))
    )


def _matched_ids(matched: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Convert matched pairs list to set for order-independent comparison."""
    return set(matched)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
