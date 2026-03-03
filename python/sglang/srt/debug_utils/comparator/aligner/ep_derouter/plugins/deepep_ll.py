from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.utils import (
    compute_within_group_indices,
)


class DeepEPLLDeRouter(DeRouterPlugin):
    """De-router for SGLang DeepEP Low-Latency dispatch path.

    Routed tensor is 3D: ``(num_experts, expected_m, hidden_size)``.
    ``deepep_ll_masked_m[expert_i]`` specifies how many of the ``expected_m``
    rows are valid for expert ``expert_i``.

    ``deepep_ll_packed_recv_src_info`` has shape ``(num_experts, expected_m)``
    and encodes the source identity of each received token.  The encoding is::

        deepep_ll_packed_recv_src_info[e][j] % num_tokens == original_token_index
    """

    @property
    def required_aux_dump_names(self) -> frozenset[str]:
        return frozenset({"deepep_ll_masked_m", "deepep_ll_packed_recv_src_info"})

    def resolve_num_tokens(
        self,
        num_tokens: int,
        aux_tensors: dict[str, torch.Tensor],
    ) -> int:
        """Infer num_tokens from packed_recv_src_info valid entries.

        The dumped ``ep_num_tokens`` for the LL path is unreliable because
        ``hidden_states.shape[0]`` in ``dispatch_b`` equals ``num_experts``
        (post-dispatch shape), not the pre-dispatch token count.

        We recover the true token count as ``max(valid_token_ids) + 1``.
        """
        packed: torch.Tensor = aux_tensors["deepep_ll_packed_recv_src_info"]
        masked_m: torch.Tensor = aux_tensors["deepep_ll_masked_m"]

        flat_src_info: torch.Tensor = _extract_valid_rows(
            packed.unsqueeze(-1), masked_m
        ).squeeze(-1)

        if flat_src_info.numel() == 0:
            return 0

        return int(flat_src_info.long().max().item()) + 1

    def flatten_routed_tensor(
        self,
        routed_tensor: torch.Tensor,
        aux_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        masked_m: torch.Tensor = aux_tensors["deepep_ll_masked_m"]
        return _extract_valid_rows(routed_tensor, masked_m)

    def compute_forward_permutation(
        self,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
        num_routed: int,
    ) -> torch.Tensor:
        packed_recv_src_info: torch.Tensor = aux_tensors[
            "deepep_ll_packed_recv_src_info"
        ]
        masked_m: torch.Tensor = aux_tensors["deepep_ll_masked_m"]

        flat_src_info: torch.Tensor = _extract_valid_rows(
            packed_recv_src_info.unsqueeze(-1), masked_m
        ).squeeze(-1)

        token_ids: torch.Tensor = flat_src_info.long() % num_tokens
        k_indices: torch.Tensor = compute_within_group_indices(token_ids)
        forward_perm: torch.Tensor = token_ids * top_k + k_indices

        total_slots: int = num_tokens * top_k
        forward_perm[forward_perm >= total_slots] = -1
        return forward_perm


def _extract_valid_rows(
    tensor_3d: torch.Tensor, masked_m: torch.Tensor
) -> torch.Tensor:
    """Extract valid rows from a 3D ``[num_experts, expected_m, ...]`` tensor.

    For each expert ``e``, takes the first ``masked_m[e]`` rows and concatenates
    them into a 2D tensor.
    """
    expected_m: int = tensor_3d.shape[1]
    arange: torch.Tensor = torch.arange(expected_m, device=masked_m.device)
    valid_mask: torch.Tensor = arange.unsqueeze(0) < masked_m.rename(None).unsqueeze(1)
    return tensor_3d.rename(None)[valid_mask]
