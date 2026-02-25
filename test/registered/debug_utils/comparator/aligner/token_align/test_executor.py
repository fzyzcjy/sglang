from __future__ import annotations

import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    AuxTensorsForStep,
    SideAux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.executor import (
    execute_alignment,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.indexer import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.planner import (
    compute_alignment_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    AlignmentPlan,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestExecuteAlignment:
    """Tests for token alignment execution."""

    def test_thd_vs_thd_identity(self):
        """Two identical thd sides produce element-wise equal aligned tensors."""
        torch.manual_seed(42)
        hidden_step0 = torch.randn(5, 8)  # 5 tokens, hidden_dim=8
        hidden_step1 = torch.randn(2, 8)  # 2 tokens

        aux = AuxTensorsForStep(
            input_ids=torch.tensor([10, 20, 30, 40, 50]),
            positions=torch.tensor([0, 1, 2, 0, 1]),
            seq_lens=torch.tensor([3, 2]),
            seq_ids=("A", "B"),
        )
        aux_step1 = AuxTensorsForStep(
            input_ids=torch.tensor([31, 51]),
            positions=torch.tensor([3, 2]),
            seq_lens=torch.tensor([1, 1]),
            seq_ids=("A", "B"),
        )

        side_aux = SideAux(
            steps={0: aux, 1: aux_step1},
            framework="sglang",
            layout="thd",
        )

        index = build_seqs_info(side_aux)
        plan = compute_alignment_plan(indices=Pair(x=index, y=index))

        tensors = {0: hidden_step0, 1: hidden_step1}
        aligned: Pair[torch.Tensor] = execute_alignment(
            plan=plan, tensors=Pair(x=tensors, y=tensors)
        )

        assert torch.equal(aligned.x, aligned.y)
        assert aligned.x.shape[0] == len(plan.match_steps.x)

    def test_zero_matched_tokens(self):
        """Empty AlignmentPlan (no matched tokens) returns shape[0]==0 without crash."""
        torch.manual_seed(42)

        plan = AlignmentPlan(
            match_steps=Pair(x=(), y=()),
            match_indices=Pair(x=(), y=()),
        )

        tensors = {0: torch.randn(5, 8)}
        aligned: Pair[torch.Tensor] = execute_alignment(
            plan=plan, tensors=Pair(x=tensors, y=tensors)
        )

        assert aligned.x.shape[0] == 0
        assert aligned.y.shape[0] == 0
        assert aligned.x.shape[1:] == (8,)
        assert aligned.y.shape[1:] == (8,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
