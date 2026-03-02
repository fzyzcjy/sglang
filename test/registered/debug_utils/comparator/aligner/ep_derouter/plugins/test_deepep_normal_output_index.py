import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.deepep_normal_output_index import (
    DeepEPNormalOutputIndexDeRouter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestDeepEPNormalOutputIndexDeRouter:
    """Test compute_forward_permutation from output_index."""

    def test_identity(self) -> None:
        """output_index is identity → forward_perm is identity."""
        num_tokens: int = 4
        top_k: int = 1

        # output_index[tok][k] = tok*top_k + k (identity mapping)
        output_index: torch.Tensor = torch.arange(
            num_tokens, dtype=torch.long
        ).unsqueeze(1)

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_output_index": output_index},
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

        # token 0 → dispatch 2, token 1 → dispatch 1, token 2 → dispatch 0
        output_index: torch.Tensor = torch.tensor([[2], [1], [0]], dtype=torch.long)

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_output_index": output_index},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=num_tokens * top_k,
        )

        # forward_perm[dispatch_pos] = canonical_flat_idx
        # dispatch 0 ← token 2 (canonical 2), dispatch 1 ← token 1, dispatch 2 ← token 0
        assert torch.equal(perm, torch.tensor([2, 1, 0], dtype=torch.long))

    def test_top_k_2(self) -> None:
        """output_index with top_k=2: each token has two dispatch positions."""
        num_tokens: int = 2
        top_k: int = 2
        total_slots: int = num_tokens * top_k

        # output_index shape: [num_tokens, top_k]
        # token 0, k=0 → dispatch 3; token 0, k=1 → dispatch 1
        # token 1, k=0 → dispatch 0; token 1, k=1 → dispatch 2
        output_index: torch.Tensor = torch.tensor([[3, 1], [0, 2]], dtype=torch.long)

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_output_index": output_index},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)
        # forward_perm[dispatch_pos] = canonical_flat_idx
        # canonical_flat: tok*top_k+k
        # dispatch 0 ← token 1, k=0 → canonical 1*2+0 = 2
        # dispatch 1 ← token 0, k=1 → canonical 0*2+1 = 1
        # dispatch 2 ← token 1, k=1 → canonical 1*2+1 = 3
        # dispatch 3 ← token 0, k=0 → canonical 0*2+0 = 0
        assert torch.equal(perm, torch.tensor([2, 1, 3, 0], dtype=torch.long))

    def test_partial_ep_rank(self) -> None:
        """EP rank only handles subset of tokens (num_routed < total_slots)."""
        num_tokens: int = 4
        top_k: int = 1

        # This rank handles tokens 1 and 3, dispatched to positions 0 and 1
        # output_index includes ALL tokens but only 2 are routed to this rank
        # Actually, output_index on this rank only has the tokens dispatched here
        # But we pass num_tokens = original token count before dispatch
        # and output_index only covers the tokens on this rank
        # Wait - output_index shape is [num_recv_tokens, top_k], same as topk_ids
        # num_recv_tokens is the number of tokens received by this EP rank

        # Let's say this rank received 2 tokens (originally token 1 and 3)
        # output_index maps them to contiguous buffer positions
        output_index: torch.Tensor = torch.tensor([[0], [1]], dtype=torch.long)

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_output_index": output_index},
            num_tokens=2,
            top_k=top_k,
            num_routed=2,
        )

        assert perm.shape == (2,)
        # Identity: dispatch 0 ← canonical 0, dispatch 1 ← canonical 1
        assert torch.equal(perm, torch.tensor([0, 1], dtype=torch.long))

    def test_output_shape(self) -> None:
        """Output always has shape [num_routed]."""
        num_tokens: int = 6
        top_k: int = 2
        total_slots: int = num_tokens * top_k

        output_index: torch.Tensor = torch.arange(
            num_tokens * top_k, dtype=torch.long
        ).view(num_tokens, top_k)

        plugin: DeepEPNormalOutputIndexDeRouter = DeepEPNormalOutputIndexDeRouter()
        perm: torch.Tensor = plugin.compute_forward_permutation(
            aux_tensors={"deepep_normal_output_index": output_index},
            num_tokens=num_tokens,
            top_k=top_k,
            num_routed=total_slots,
        )

        assert perm.shape == (total_slots,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
