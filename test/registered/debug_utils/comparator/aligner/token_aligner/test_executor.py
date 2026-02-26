from __future__ import annotations

import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.executor import (
    execute_token_aligner,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.planner import (
    compute_token_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.seq_info_builder import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    SGLangSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerPlan,
    TokenAlignerStepAux,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.dims import TokenLayout
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

        aux = TokenAlignerStepAux(
            input_ids=[10, 20, 30, 40, 50],
            positions=[0, 1, 2, 0, 1],
            seq_lens=[3, 2],
            seq_ids=[SGLangSeqId(rid="A"), SGLangSeqId(rid="B")],
        )
        aux_step1 = TokenAlignerStepAux(
            input_ids=[31, 51],
            positions=[3, 2],
            seq_lens=[1, 1],
            seq_ids=[SGLangSeqId(rid="A"), SGLangSeqId(rid="B")],
        )

        side_aux = TokenAlignerGlobalAux(
            step_auxs={0: aux, 1: aux_step1},
            framework="sglang",
            layout=TokenLayout.T,
        )

        index = build_seqs_info(side_aux)
        plan = compute_token_aligner_plan(seqs_info_pair=Pair(x=index, y=index))

        tensors = {0: hidden_step0, 1: hidden_step1}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan, tensor_of_step_pair=Pair(x=tensors, y=tensors)
        )

        assert torch.equal(aligned.x, aligned.y)
        assert aligned.x.shape[0] == len(plan.locators.x.steps)

    def test_zero_matched_tokens(self):
        """Empty TokenAlignerPlan (no matched tokens) returns shape[0]==0 without crash."""
        torch.manual_seed(42)

        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[], token_index_in_step=[]),
                y=TokenLocator(steps=[], token_index_in_step=[]),
            ),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

        tensors = {0: torch.randn(5, 8)}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan, tensor_of_step_pair=Pair(x=tensors, y=tensors)
        )

        assert aligned.x.shape[0] == 0
        assert aligned.y.shape[0] == 0
        assert aligned.x.shape[1:] == (8,)
        assert aligned.y.shape[1:] == (8,)


class TestTokenDim:
    """Tests for non-zero token_dim support."""

    def _make_simple_plan(self, *, num_tokens: int) -> TokenAlignerPlan:
        locator = TokenLocator(
            steps=[0] * num_tokens,
            token_index_in_step=list(range(num_tokens)),
        )
        return TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

    def test_token_dim_nonzero(self) -> None:
        """tensor shape [3, 5, 8], token_dim=1 -> token dim stays at dim 1."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(3, 5, 8)
        plan: TokenAlignerPlan = self._make_simple_plan(num_tokens=5)

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=1, y=1),
        )

        assert aligned.x.shape == (3, 5, 8)
        assert torch.equal(aligned.x, aligned.y)
        for i in range(5):
            assert torch.equal(
                aligned.x.select(dim=1, index=i), tensor.select(dim=1, index=i)
            )

    def test_token_dim_last(self) -> None:
        """tensor shape [3, 8, 5], token_dim=2 -> token dim stays at dim 2."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(3, 8, 5)
        plan: TokenAlignerPlan = self._make_simple_plan(num_tokens=5)

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=2, y=2),
        )

        assert aligned.x.shape == (3, 8, 5)
        for i in range(5):
            assert torch.equal(
                aligned.x.select(dim=2, index=i), tensor.select(dim=2, index=i)
            )

    def test_token_dim_zero(self) -> None:
        """token_dim=0 selects along first dimension (standard t-h-d layout)."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(5, 8)
        plan: TokenAlignerPlan = self._make_simple_plan(num_tokens=5)

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=0, y=0),
        )

        assert aligned.x.shape == (5, 8)
        for i in range(5):
            assert torch.equal(aligned.x[i], tensor.select(dim=0, index=i))

    def test_zero_matched_tokens_nonzero_token_dim(self) -> None:
        """Empty plan with token_dim=1 produces correct empty shape."""
        torch.manual_seed(42)

        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[], token_index_in_step=[]),
                y=TokenLocator(steps=[], token_index_in_step=[]),
            ),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

        # tensor shape [3, 5, 8], token_dim=1
        tensors: dict[int, torch.Tensor] = {0: torch.randn(3, 5, 8)}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=1, y=1),
        )

        # token dim (dim 1) set to 0, other dims preserved -> [3, 0, 8]
        assert aligned.x.shape == (3, 0, 8)
        assert aligned.y.shape == (3, 0, 8)

    def test_high_rank_tensor(self) -> None:
        """tensor shape [2, 3, 5, 4, 8] (a b t c d), token_dim=2 -> stays at dim 2."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(2, 3, 5, 4, 8)
        plan: TokenAlignerPlan = self._make_simple_plan(num_tokens=5)

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=2, y=2),
        )

        assert aligned.x.shape == (2, 3, 5, 4, 8)
        for i in range(5):
            assert torch.equal(
                aligned.x.select(dim=2, index=i), tensor.select(dim=2, index=i)
            )


class TestBSHDExecutor:
    """Tests for BSHD tensor reshape in executor.

    BSHD tensors have separate B (batch) and S (seq) dims that must be collapsed
    into a single flat token dim [B*S] for alignment. The B and S dims are NOT
    always at positions 0 and 1 — they can appear anywhere in the shape.

    After collapsing B*S → flat token dim, the result has one fewer dimension
    than the input. The executor should:
    1. Collapse B and S dims into a single flat dim
    2. Index into the flat dim using token_index_in_step
    3. Return a result with the flat token dim (not the original B, S dims)

    In all cases below, token_dim refers to the position of the B dim in the
    original tensor (since B and S are adjacent and B comes first, the collapsed
    B*S dim ends up at B's position).
    """

    # ── Case 1: standard "b s h d" — B=dim0, S=dim1 ──

    def test_bshd_standard_bs_at_front(self):
        """dims="b s h d": shape [B=2, S=3, H=4, D=5].

        After collapse → [6, 4, 5]. token_dim=0 indexes into flat B*S.
        """
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(2, 3, 4, 5)
        flat: torch.Tensor = tensor.reshape(6, 4, 5)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 3, 5],  # batch0/pos0, batch1/pos0, batch1/pos2
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=0, y=0),
        )

        assert aligned.x.shape == (3, 4, 5)
        assert torch.equal(aligned.x[0], flat[0])
        assert torch.equal(aligned.x[1], flat[3])
        assert torch.equal(aligned.x[2], flat[5])

    # ── Case 2: "b s h" — minimal 3D, B=dim0, S=dim1 ──

    def test_bshd_3d_bs_at_front(self):
        """dims="b s h": shape [B=2, S=3, H=4].

        After collapse → [6, 4]. token_dim=0.
        """
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(2, 3, 4)
        flat: torch.Tensor = tensor.reshape(6, 4)

        locator = TokenLocator(
            steps=[0, 0, 0, 0],
            token_index_in_step=[0, 2, 3, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=0, y=0),
        )

        assert aligned.x.shape == (4, 4)
        assert torch.equal(aligned.x[0], flat[0])
        assert torch.equal(aligned.x[1], flat[2])
        assert torch.equal(aligned.x[2], flat[3])
        assert torch.equal(aligned.x[3], flat[5])

    # ── Case 3: "h b s d" — B=dim1, S=dim2, non-leading ──

    def test_bshd_bs_not_at_front(self):
        """dims="h b s d": shape [H=4, B=2, S=3, D=5].

        B at dim1, S at dim2. After collapse → [H=4, B*S=6, D=5].
        token_dim=1 (position of the collapsed B*S dim).
        """
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 2, 3, 5)
        # collapse dims 1,2 → [4, 6, 5]
        flat: torch.Tensor = tensor.reshape(4, 6, 5)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 3, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=1, y=1),
        )

        assert aligned.x.shape == (4, 3, 5)
        for idx, flat_idx in enumerate([0, 3, 5]):
            assert torch.equal(
                aligned.x.select(dim=1, index=idx),
                flat.select(dim=1, index=flat_idx),
            )

    # ── Case 4: "e b s h d" — expert dim before B, B=dim1, S=dim2 ──

    def test_bshd_expert_before_bs(self):
        """dims="e b s h d": shape [E=2, B=3, S=4, H=5, D=6].

        B at dim1, S at dim2. After collapse → [E=2, B*S=12, H=5, D=6].
        token_dim=1.
        """
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(2, 3, 4, 5, 6)
        flat: torch.Tensor = tensor.reshape(2, 12, 5, 6)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 5, 11],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=1, y=1),
        )

        assert aligned.x.shape == (2, 3, 5, 6)
        for idx, flat_idx in enumerate([0, 5, 11]):
            assert torch.equal(
                aligned.x.select(dim=1, index=idx),
                flat.select(dim=1, index=flat_idx),
            )

    # ── Case 5: "h d b s" — B and S at the end ──

    def test_bshd_bs_at_end(self):
        """dims="h d b s": shape [H=4, D=5, B=2, S=3].

        B at dim2, S at dim3. After collapse → [H=4, D=5, B*S=6].
        token_dim=2.
        """
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 5, 2, 3)
        flat: torch.Tensor = tensor.reshape(4, 5, 6)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[1, 3, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: tensor}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=2, y=2),
        )

        assert aligned.x.shape == (4, 5, 3)
        for idx, flat_idx in enumerate([1, 3, 5]):
            assert torch.equal(
                aligned.x.select(dim=2, index=idx),
                flat.select(dim=2, index=flat_idx),
            )

    # ── Case 6: cross-layout THD vs BSHD ──

    def test_cross_layout_thd_vs_bshd(self):
        """x is THD [T=6, H=8] (token_dim=0), y is BSHD [B=2, S=3, H=8] (token_dim=0).

        After BSHD collapse → y becomes [6, 8], same token count.
        Both sides select 3 matched tokens along their respective token dims.
        """
        torch.manual_seed(42)
        tensor_thd: torch.Tensor = torch.randn(6, 8)
        tensor_bshd: torch.Tensor = torch.randn(2, 3, 8)
        flat_bshd: torch.Tensor = tensor_bshd.reshape(6, 8)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 2, 5],
        )
        plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.BS),
        )

        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x={0: tensor_thd}, y={0: tensor_bshd}),
            token_dims=Pair(x=0, y=0),
        )

        assert aligned.x.shape == (3, 8)
        assert aligned.y.shape == (3, 8)
        assert torch.equal(aligned.x[0], tensor_thd[0])
        assert torch.equal(aligned.y[0], flat_bshd[0])
        assert torch.equal(aligned.y[2], flat_bshd[5])

    # ── Case 7: empty plan with non-leading B,S ──

    def test_bshd_empty_plan_bs_not_at_front(self):
        """Empty plan with dims="h b s d": shape [H=4, B=2, S=3, D=5].

        After collapse → shape [H=4, D=5] with token_dim=1 set to 0 → [H=4, 0, D=5].
        """
        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[], token_index_in_step=[]),
                y=TokenLocator(steps=[], token_index_in_step=[]),
            ),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: torch.randn(4, 2, 3, 5)}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=1, y=1),
        )

        # [4, 2, 3, 5] → collapse B,S at dims 1,2 → [4, 6, 5] → token_dim=1 set to 0 → [4, 0, 5]
        assert aligned.x.shape == (4, 0, 5)
        assert aligned.y.shape == (4, 0, 5)

    # ── Case 8: empty plan standard BSHD ──

    def test_bshd_empty_plan_bs_at_front(self):
        """Empty plan with dims="b s h": shape [B=2, S=3, H=4].

        After collapse → [0, 4].
        """
        plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[], token_index_in_step=[]),
                y=TokenLocator(steps=[], token_index_in_step=[]),
            ),
            layouts=Pair(x=TokenLayout.BS, y=TokenLayout.BS),
        )

        tensors: dict[int, torch.Tensor] = {0: torch.randn(2, 3, 4)}
        aligned: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan,
            tensor_of_step_pair=Pair(x=tensors, y=tensors),
            token_dims=Pair(x=0, y=0),
        )

        assert aligned.x.shape == (0, 4)
        assert aligned.y.shape == (0, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
