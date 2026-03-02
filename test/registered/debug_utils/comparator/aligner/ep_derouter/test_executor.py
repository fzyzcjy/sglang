import sys
from typing import Any

import pydantic
import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.executor import (
    execute_de_router_plan,
)
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.types import DeRouterPlan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class _FakeAuxLoader:
    """Minimal mock for RawAuxLoader that returns pre-set tensors."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def load(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return dict(self._tensors)


_FAKE_META: dict[str, Any] = {"step": 0, "rank": 0, "layer_id": 0}


def _make_ep_meta_tensors(
    dispatch_path: str, *, num_tokens: int, top_k: int
) -> dict[str, torch.Tensor]:
    return {
        f"{dispatch_path}_ep_num_tokens": torch.tensor(num_tokens),
        f"{dispatch_path}_ep_top_k": torch.tensor(top_k),
    }


class TestExecuteDeRouterPlan:
    """Test plugin selection and aux tensor resolution."""

    def test_fused_moe_dispatch(self) -> None:
        """Plan with dispatch_path='fused_moe' selects FusedMoEDeRouter."""
        num_tokens: int = 2
        top_k: int = 1
        hidden_dim: int = 3

        plan: DeRouterPlan = DeRouterPlan(dispatch_path="fused_moe")

        sorted_token_ids: torch.Tensor = torch.tensor([1, 0], dtype=torch.int64)
        routed_tensor: torch.Tensor = torch.tensor(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
        )
        loader = _FakeAuxLoader(
            {
                "fused_moe_sorted_token_ids": sorted_token_ids,
                **_make_ep_meta_tensors(
                    "fused_moe", num_tokens=num_tokens, top_k=top_k
                ),
            }
        )

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_loader=loader, meta=_FAKE_META
        )

        assert result.shape == (num_tokens * top_k, hidden_dim)
        # sorted_token_ids = [1, 0] → pos 0 maps to flatten_idx 1, pos 1 maps to 0
        assert torch.allclose(result[0], torch.tensor([20.0, 21.0, 22.0]))
        assert torch.allclose(result[1], torch.tensor([10.0, 11.0, 12.0]))

    def test_unknown_dispatch_path_raises(self) -> None:
        """Unknown dispatch_path is rejected by Pydantic validation."""
        with pytest.raises(pydantic.ValidationError):
            DeRouterPlan(dispatch_path="unknown_path")

    def test_missing_aux_loader_raises(self) -> None:
        """Missing aux_loader when plugin requires aux tensors raises ValueError."""
        plan: DeRouterPlan = DeRouterPlan(dispatch_path="fused_moe")
        with pytest.raises(ValueError, match="no aux_loader"):
            execute_de_router_plan(
                plan=plan,
                tensor=torch.randn(2, 3),
                aux_loader=None,
                meta=_FAKE_META,
            )

    def test_missing_aux_tensor_raises(self) -> None:
        """Missing required auxiliary tensor in loader output raises ValueError."""
        plan: DeRouterPlan = DeRouterPlan(dispatch_path="fused_moe")
        loader = _FakeAuxLoader({})
        with pytest.raises(ValueError, match="not found"):
            execute_de_router_plan(
                plan=plan,
                tensor=torch.randn(2, 3),
                aux_loader=loader,
                meta=_FAKE_META,
            )

    def test_megatron_a2a_dispatch(self) -> None:
        """Plan with dispatch_path='megatron_a2a' selects MegatronA2ADeRouter."""
        num_tokens: int = 3
        top_k: int = 1
        hidden_dim: int = 2

        plan: DeRouterPlan = DeRouterPlan(dispatch_path="megatron_a2a")

        loader = _FakeAuxLoader(
            {
                "megatron_a2a_reversed_local_input_permutation_mapping": torch.arange(
                    num_tokens, dtype=torch.long
                ),
                **_make_ep_meta_tensors(
                    "megatron_a2a", num_tokens=num_tokens, top_k=top_k
                ),
            }
        )
        routed_tensor: torch.Tensor = torch.randn(num_tokens, hidden_dim)

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_loader=loader, meta=_FAKE_META
        )

        assert torch.allclose(result, routed_tensor)

    def test_deepep_normal_dispatch(self) -> None:
        """End-to-end: deepep_normal with src2dst permutation."""
        num_tokens: int = 4
        top_k: int = 1
        hidden_dim: int = 2

        plan: DeRouterPlan = DeRouterPlan(dispatch_path="deepep_normal")

        # src2dst: canonical 0→dispatch 1, canonical 1→dispatch 0,
        #          canonical 2→dispatch 3, canonical 3→dispatch 2
        src2dst: torch.Tensor = torch.tensor([1, 0, 3, 2], dtype=torch.long)
        loader = _FakeAuxLoader(
            {
                "deepep_normal_src2dst": src2dst,
                **_make_ep_meta_tensors(
                    "deepep_normal", num_tokens=num_tokens, top_k=top_k
                ),
            }
        )
        # routed_tensor is in dispatch order: [dispatch0, dispatch1, dispatch2, dispatch3]
        routed_tensor: torch.Tensor = torch.tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        )

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_loader=loader, meta=_FAKE_META
        )

        assert result.shape == (num_tokens * top_k, hidden_dim)
        # forward_perm[dispatch] = canonical: [1, 0, 3, 2]
        # output[canonical] = routed[dispatch]:
        #   canonical 0 = routed[1] = [20, 21]
        #   canonical 1 = routed[0] = [10, 11]
        #   canonical 2 = routed[3] = [40, 41]
        #   canonical 3 = routed[2] = [30, 31]
        assert torch.allclose(result[0], torch.tensor([20.0, 21.0]))
        assert torch.allclose(result[1], torch.tensor([10.0, 11.0]))
        assert torch.allclose(result[2], torch.tensor([40.0, 41.0]))
        assert torch.allclose(result[3], torch.tensor([30.0, 31.0]))

    def test_deepep_normal_output_index_dispatch(self) -> None:
        """End-to-end: deepep_normal_output_index with output_index permutation."""
        num_tokens: int = 4
        top_k: int = 1
        hidden_dim: int = 2

        plan: DeRouterPlan = DeRouterPlan(dispatch_path="deepep_normal_output_index")

        # output_index[tok][k] = dispatch_pos
        # token 0 → dispatch 1, token 1 → dispatch 0,
        # token 2 → dispatch 3, token 3 → dispatch 2
        output_index: torch.Tensor = torch.tensor(
            [[1], [0], [3], [2]], dtype=torch.long
        )
        loader = _FakeAuxLoader(
            {
                "deepep_normal_output_index": output_index,
                **_make_ep_meta_tensors(
                    "deepep_normal_output_index",
                    num_tokens=num_tokens,
                    top_k=top_k,
                ),
            }
        )
        # routed_tensor is in dispatch order: [dispatch0, dispatch1, dispatch2, dispatch3]
        routed_tensor: torch.Tensor = torch.tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        )

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_loader=loader, meta=_FAKE_META
        )

        assert result.shape == (num_tokens * top_k, hidden_dim)
        # forward_perm[dispatch_pos] = canonical_flat_idx (= tok*top_k + k)
        # dispatch 0 ← token 1 (canonical 1)
        # dispatch 1 ← token 0 (canonical 0)
        # dispatch 2 ← token 3 (canonical 3)
        # dispatch 3 ← token 2 (canonical 2)
        # output[canonical] = routed[dispatch]:
        #   canonical 0 = routed[1] = [20, 21]
        #   canonical 1 = routed[0] = [10, 11]
        #   canonical 2 = routed[3] = [40, 41]
        #   canonical 3 = routed[2] = [30, 31]
        assert torch.allclose(result[0], torch.tensor([20.0, 21.0]))
        assert torch.allclose(result[1], torch.tensor([10.0, 11.0]))
        assert torch.allclose(result[2], torch.tensor([40.0, 41.0]))
        assert torch.allclose(result[3], torch.tensor([30.0, 31.0]))

    def test_deepep_ll_dispatch(self) -> None:
        """End-to-end: deepep_ll with 2-expert 3D tensor, verifies flatten + scatter."""
        num_tokens: int = 2
        top_k: int = 2
        num_experts: int = 2
        expected_m: int = 2
        hidden_dim: int = 3

        plan: DeRouterPlan = DeRouterPlan(dispatch_path="deepep_ll")

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        # Expert 0: token 0 and token 1
        packed_recv_src_info[0, 0] = 0
        packed_recv_src_info[0, 1] = 1
        # Expert 1: token 0 and token 1 (second appearance → k=1)
        packed_recv_src_info[1, 0] = 0
        packed_recv_src_info[1, 1] = 1

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        routed_tensor: torch.Tensor = torch.zeros(num_experts, expected_m, hidden_dim)
        routed_tensor[0, 0] = torch.tensor([10.0, 11.0, 12.0])  # token 0, k=0
        routed_tensor[0, 1] = torch.tensor([20.0, 21.0, 22.0])  # token 1, k=0
        routed_tensor[1, 0] = torch.tensor([30.0, 31.0, 32.0])  # token 0, k=1
        routed_tensor[1, 1] = torch.tensor([40.0, 41.0, 42.0])  # token 1, k=1

        loader = _FakeAuxLoader(
            {
                "deepep_ll_packed_recv_src_info": packed_recv_src_info,
                "deepep_ll_masked_m": masked_m,
                **_make_ep_meta_tensors(
                    "deepep_ll", num_tokens=num_tokens, top_k=top_k
                ),
            }
        )

        result: torch.Tensor = execute_de_router_plan(
            plan=plan, tensor=routed_tensor, aux_loader=loader, meta=_FAKE_META
        )

        total_slots: int = num_tokens * top_k
        assert result.shape == (total_slots, hidden_dim)
        # token 0, k=0 → slot 0
        assert torch.allclose(result[0], torch.tensor([10.0, 11.0, 12.0]))
        # token 0, k=1 → slot 1
        assert torch.allclose(result[1], torch.tensor([30.0, 31.0, 32.0]))
        # token 1, k=0 → slot 2
        assert torch.allclose(result[2], torch.tensor([20.0, 21.0, 22.0]))
        # token 1, k=1 → slot 3
        assert torch.allclose(result[3], torch.tensor([40.0, 41.0, 42.0]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
