import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_ll import (
    DeepEPLLDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _make_aux(
    packed_recv_src_info: torch.Tensor,
    masked_m: torch.Tensor,
    recv_topk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "deepep_ll_packed_recv_src_info": packed_recv_src_info,
        "deepep_ll_masked_m": masked_m,
        "deepep_ll_recv_topk_ids": recv_topk_ids,
    }


class TestDeepEPLLDeRouter:
    """Test flatten_routed_tensor and compute_forward_permutation."""

    def test_flatten_extracts_valid_rows(self) -> None:
        """flatten_routed_tensor extracts valid rows from 3D tensor."""
        num_experts: int = 2
        expected_m: int = 4
        hidden_dim: int = 3

        routed_tensor: torch.Tensor = torch.zeros(num_experts, expected_m, hidden_dim)
        routed_tensor[0, 0] = torch.tensor([10.0, 11.0, 12.0])
        routed_tensor[0, 1] = torch.tensor([20.0, 21.0, 22.0])
        routed_tensor[1, 0] = torch.tensor([30.0, 31.0, 32.0])
        routed_tensor[1, 1] = torch.tensor([40.0, 41.0, 42.0])

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        flat: torch.Tensor = plugin.flatten_routed_tensor(
            routed_tensor=routed_tensor,
            aux_tensors={
                "deepep_ll_packed_recv_src_info": torch.zeros(2, 4),
                "deepep_ll_masked_m": masked_m,
                "deepep_ll_recv_topk_ids": torch.zeros(2, 2, dtype=torch.long),
            },
        )

        assert flat.shape == (4, hidden_dim)
        assert torch.allclose(flat[0], torch.tensor([10.0, 11.0, 12.0]))
        assert torch.allclose(flat[1], torch.tensor([20.0, 21.0, 22.0]))
        assert torch.allclose(flat[2], torch.tensor([30.0, 31.0, 32.0]))
        assert torch.allclose(flat[3], torch.tensor([40.0, 41.0, 42.0]))

    def test_basic_permutation(self) -> None:
        """Each token dispatched to one expert → k=0 for each."""
        num_tokens: int = 4
        top_k: int = 2
        num_experts: int = 2
        expected_m: int = 4

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        # expert 0 gets token 0 and token 1
        packed_recv_src_info[0, 0] = 0
        packed_recv_src_info[0, 1] = 1
        # expert 1 gets token 2 and token 3
        packed_recv_src_info[1, 0] = 2
        packed_recv_src_info[1, 1] = 3

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        # topk_ids: each token's first expert matches where it was dispatched
        # token 0 → experts [0, 5], token 1 → experts [0, 6]
        # token 2 → experts [1, 7], token 3 → experts [1, 8]
        topk_ids: torch.Tensor = torch.tensor(
            [[0, 5], [0, 6], [1, 7], [1, 8]], dtype=torch.long
        )

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(packed_recv_src_info, masked_m, topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=4,
        )

        assert perm.shape == (4,)
        # expert 0 matches k=0 for tokens 0,1; expert 1 matches k=0 for tokens 2,3
        # token0,k=0→0; token1,k=0→2; token2,k=0→4; token3,k=0→6
        assert torch.equal(perm, torch.tensor([0, 2, 4, 6], dtype=torch.long))

    def test_padding_rows_ignored(self) -> None:
        """Only masked_m valid rows contribute to the permutation."""
        num_tokens: int = 2
        top_k: int = 1
        num_experts: int = 2
        expected_m: int = 4

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        packed_recv_src_info[0, 0] = 0
        packed_recv_src_info[0, 1] = 999  # padding
        packed_recv_src_info[1, 0] = 1
        packed_recv_src_info[1, 1] = 999  # padding

        masked_m: torch.Tensor = torch.tensor([1, 1], dtype=torch.long)
        topk_ids: torch.Tensor = torch.tensor([[0], [1]], dtype=torch.long)

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(packed_recv_src_info, masked_m, topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=2,
        )

        assert perm.shape == (2,)
        assert torch.equal(perm, torch.tensor([0, 1], dtype=torch.long))

    def test_top_k_assignment_via_topk_ids(self) -> None:
        """When a token appears in multiple experts, k-index from topk_ids lookup."""
        num_tokens: int = 2
        top_k: int = 2
        num_experts: int = 2
        expected_m: int = 2

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_experts, expected_m, dtype=torch.long
        )
        # expert 0 gets both tokens
        packed_recv_src_info[0, 0] = 0
        packed_recv_src_info[0, 1] = 1
        # expert 1 gets both tokens
        packed_recv_src_info[1, 0] = 0
        packed_recv_src_info[1, 1] = 1

        masked_m: torch.Tensor = torch.tensor([2, 2], dtype=torch.long)

        # token 0 → experts [0, 1] (expert 0 is k=0, expert 1 is k=1)
        # token 1 → experts [1, 0] (expert 1 is k=0, expert 0 is k=1)
        topk_ids: torch.Tensor = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long
        )

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(packed_recv_src_info, masked_m, topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=4,
        )

        assert perm.shape == (4,)
        # Flat order from _extract_valid_rows: expert0[0]=tok0, expert0[1]=tok1,
        #   expert1[0]=tok0, expert1[1]=tok1
        # tok0 at expert0 → k=0 → slot 0*2+0 = 0
        # tok1 at expert0 → k=1 → slot 1*2+1 = 3
        # tok0 at expert1 → k=1 → slot 0*2+1 = 1
        # tok1 at expert1 → k=0 → slot 1*2+0 = 2
        assert torch.equal(perm, torch.tensor([0, 3, 1, 2], dtype=torch.long))

    def test_ep_rank_offset(self) -> None:
        """Local expert IDs are offset by ep_rank * num_local_experts for global lookup."""
        num_tokens: int = 2
        top_k: int = 2
        num_local_experts: int = 2
        expected_m: int = 2

        packed_recv_src_info: torch.Tensor = torch.zeros(
            num_local_experts, expected_m, dtype=torch.long
        )
        # local expert 0 gets token 0, local expert 1 gets token 1
        packed_recv_src_info[0, 0] = 0
        packed_recv_src_info[1, 0] = 1
        masked_m: torch.Tensor = torch.tensor([1, 1], dtype=torch.long)

        # topk_ids uses GLOBAL expert IDs
        # token 0 → global experts [2, 5] (local 0 on ep_rank=1 → global 2)
        # token 1 → global experts [3, 4] (local 1 on ep_rank=1 → global 3)
        topk_ids: torch.Tensor = torch.tensor(
            [[2, 5], [3, 4]], dtype=torch.long
        )

        aux: dict[str, torch.Tensor] = _make_aux(packed_recv_src_info, masked_m, topk_ids)
        aux["_ep_rank"] = torch.tensor(1, dtype=torch.long)

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=aux,
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=2,
        )

        assert perm.shape == (2,)
        # local expert 0 + ep_rank=1 * 2 = global expert 2
        # token 0, expert 2 → k=0 → slot 0*2+0 = 0
        # local expert 1 + ep_rank=1 * 2 = global expert 3
        # token 1, expert 3 → k=0 → slot 1*2+0 = 2
        assert torch.equal(perm, torch.tensor([0, 2], dtype=torch.long))

    def test_resolve_num_tokens(self) -> None:
        """resolve_num_tokens infers correct token count from topk_ids shape."""
        packed_recv_src_info: torch.Tensor = torch.zeros(4, 8, dtype=torch.long)
        masked_m: torch.Tensor = torch.tensor([1, 1, 1, 0], dtype=torch.long)
        topk_ids: torch.Tensor = torch.zeros(5, 2, dtype=torch.long)

        plugin: DeepEPLLDeRouter = DeepEPLLDeRouter()
        result: int = plugin.resolve_num_tokens(
            num_tokens=999,
            aux_tensors=_make_aux(packed_recv_src_info, masked_m, topk_ids),
        )

        assert result == 5  # topk_ids.shape[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
