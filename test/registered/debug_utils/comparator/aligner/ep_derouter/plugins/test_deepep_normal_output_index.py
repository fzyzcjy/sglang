import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_normal_output_index import (
    DeepEPNormalOutputIndexDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _make_aux(
    output_index: torch.Tensor,
    recv_topk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "deepep_normal_output_index": output_index,
        "deepep_normal_recv_topk_ids": recv_topk_ids,
    }


class TestDeepEPNormalOutputIndexDeRouter:
    """Test compute_forward_permutation from output_index + recv_topk_ids."""

    def test_identity(self) -> None:
        """output_index is identity → forward_perm is identity."""
        num_tokens: int = 4
        top_k: int = 1

        output_index: torch.Tensor = torch.arange(
            num_tokens, dtype=torch.long
        ).unsqueeze(1)
        recv_topk_ids: torch.Tensor = torch.arange(
            num_tokens, dtype=torch.long
        ).unsqueeze(1)

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(output_index, recv_topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=num_tokens * top_k,
        )

        assert perm.shape == (num_tokens * top_k,)
        assert torch.equal(perm, torch.arange(num_tokens, dtype=torch.long))

    def test_reversed(self) -> None:
        """output_index reverses dispatch positions → forward_perm is inverse."""
        num_tokens: int = 3
        top_k: int = 1

        output_index: torch.Tensor = torch.tensor([[2], [1], [0]], dtype=torch.long)
        recv_topk_ids: torch.Tensor = torch.tensor([[5], [3], [1]], dtype=torch.long)

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(output_index, recv_topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=num_tokens * top_k,
        )

        # dispatch 0 ← token 2, dispatch 1 ← token 1, dispatch 2 ← token 0
        assert torch.equal(perm, torch.tensor([2, 1, 0], dtype=torch.long))

    def test_top_k_2(self) -> None:
        """output_index with top_k=2: each token has two dispatch positions."""
        num_tokens: int = 2
        top_k: int = 2
        total_slots: int = num_tokens * top_k

        output_index: torch.Tensor = torch.tensor([[3, 1], [0, 2]], dtype=torch.long)
        recv_topk_ids: torch.Tensor = torch.tensor(
            [[10, 20], [30, 40]], dtype=torch.long
        )

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(output_index, recv_topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)
        # dispatch 0 ← token 1, k=0 → canonical 2
        # dispatch 1 ← token 0, k=1 → canonical 1
        # dispatch 2 ← token 1, k=1 → canonical 3
        # dispatch 3 ← token 0, k=0 → canonical 0
        assert torch.equal(perm, torch.tensor([2, 1, 3, 0], dtype=torch.long))

    def test_invalid_experts_filtered(self) -> None:
        """Entries with expert_id=-1 are treated as invalid (garbage output_index)."""
        num_tokens: int = 3
        top_k: int = 2
        buffer_size: int = 256

        # token 0: k=0 invalid (expert=-1, garbage pos=0), k=1 valid (expert=5, pos=128)
        # token 1: k=0 valid (expert=3, pos=0), k=1 invalid
        # token 2: k=0 invalid, k=1 valid (expert=7, pos=129)
        GARBAGE: int = 999999
        output_index: torch.Tensor = torch.tensor(
            [[0, 128], [0, GARBAGE], [GARBAGE, 129]], dtype=torch.long
        )
        recv_topk_ids: torch.Tensor = torch.tensor(
            [[-1, 5], [3, -1], [-1, 7]], dtype=torch.long
        )

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(output_index, recv_topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=buffer_size,
        )

        assert perm.shape == (buffer_size,)
        # Only 3 valid entries:
        # pos 128 ← token 0, k=1 → canonical 0*2+1 = 1
        # pos 0   ← token 1, k=0 → canonical 1*2+0 = 2
        # pos 129 ← token 2, k=1 → canonical 2*2+1 = 5
        assert perm[0].item() == 2
        assert perm[128].item() == 1
        assert perm[129].item() == 5
        # All other positions should be -1
        assert (perm == -1).sum().item() == buffer_size - 3

    def test_large_buffer_sparse(self) -> None:
        """Contiguous buffer much larger than num_tokens*top_k (realistic scenario)."""
        num_tokens: int = 5
        top_k: int = 8
        buffer_size: int = 1536

        output_index: torch.Tensor = torch.full(
            (num_tokens, top_k), -1, dtype=torch.long
        )
        recv_topk_ids: torch.Tensor = torch.full(
            (num_tokens, top_k), -1, dtype=torch.long
        )

        # Set a few valid entries (mimicking EP rank receiving subset of experts)
        valid_entries: list[tuple[int, int, int, int]] = [
            # (tok, k, expert_id, dispatch_pos)
            (0, 1, 50, 1024),
            (0, 6, 6, 128),
            (1, 3, 28, 512),
            (3, 5, 33, 768),
            (4, 7, 0, 0),
        ]
        for tok, k, expert_id, pos in valid_entries:
            output_index[tok, k] = pos
            recv_topk_ids[tok, k] = expert_id

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(output_index, recv_topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=buffer_size,
        )

        assert perm.shape == (buffer_size,)

        for tok, k, _, pos in valid_entries:
            expected_canonical: int = tok * top_k + k
            assert perm[pos].item() == expected_canonical, (
                f"tok={tok}, k={k}: perm[{pos}] = {perm[pos].item()}, "
                f"expected {expected_canonical}"
            )

        assert (perm >= 0).sum().item() == len(valid_entries)

    def test_output_shape(self) -> None:
        """Output always has shape [num_routed]."""
        num_tokens: int = 6
        top_k: int = 2
        total_slots: int = num_tokens * top_k

        output_index: torch.Tensor = torch.arange(
            num_tokens * top_k, dtype=torch.long
        ).view(num_tokens, top_k)
        recv_topk_ids: torch.Tensor = torch.arange(
            num_tokens * top_k, dtype=torch.long
        ).view(num_tokens, top_k)

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors=_make_aux(output_index, recv_topk_ids),
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
