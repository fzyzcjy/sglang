import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    AuxTensorsForStep,
    SideAux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.executor import (
    execute_alignment,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.planner import (
    build_token_index,
    compute_alignment_plan,
)
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
            req_pool_indices=torch.tensor([7, 3]),
            rids=("A", "B"),
        )
        aux_step1 = AuxTensorsForStep(
            input_ids=torch.tensor([31, 51]),
            positions=torch.tensor([3, 2]),
            seq_lens=torch.tensor([1, 1]),
            req_pool_indices=torch.tensor([7, 3]),
            rids=("A", "B"),
        )

        side_aux = SideAux(
            steps={0: aux, 1: aux_step1},
            framework="sglang",
            layout="thd",
        )

        index = build_token_index(side_aux)
        plan = compute_alignment_plan(index_a=index, index_b=index)

        tensors = {0: hidden_step0, 1: hidden_step1}
        aligned_a, aligned_b = execute_alignment(
            plan=plan, tensors_a=tensors, tensors_b=tensors
        )

        assert torch.equal(aligned_a, aligned_b)
        assert aligned_a.shape[0] == plan.summary.num_matched_tokens

    def test_thd_vs_bshd_alignment(self):
        """SGLang thd and Megatron bshd produce correctly aligned tokens."""
        torch.manual_seed(42)

        side_aux_a = SideAux(
            steps={
                0: AuxTensorsForStep(
                    input_ids=torch.tensor([10, 20, 30]),
                    positions=torch.tensor([0, 1, 2]),
                    seq_lens=torch.tensor([3]),
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
                    input_ids=torch.tensor([[10, 20, 30, 0]]),  # [1, 4] with padding
                    positions=torch.tensor([[0, 1, 2, 3]]),
                    seq_lens=torch.tensor([3]),
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

        hidden_a = torch.randn(3, 8)  # [3, hidden]
        hidden_b_raw = torch.randn(1, 4, 8)  # [B=1, S=4, hidden]
        hidden_b_raw[0, :3] = hidden_a  # same content for matched tokens

        aligned_a, aligned_b = execute_alignment(
            plan=plan,
            tensors_a={0: hidden_a},
            tensors_b={0: hidden_b_raw},
        )

        assert aligned_a.shape == aligned_b.shape
        assert aligned_a.shape[0] == 3
        assert torch.allclose(aligned_a, aligned_b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
