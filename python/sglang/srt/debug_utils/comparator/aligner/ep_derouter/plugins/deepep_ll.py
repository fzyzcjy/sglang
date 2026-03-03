from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin


class DeepEPLLDeRouter(DeRouterPlugin):
    """De-router for SGLang DeepEP Low-Latency dispatch path.

    Routed tensor is 3D: ``(num_experts, expected_m, hidden_size)``.
    ``deepep_ll_masked_m[expert_i]`` specifies how many of the ``expected_m``
    rows are valid for expert ``expert_i``.

    ``deepep_ll_packed_recv_src_info`` has shape ``(num_experts, expected_m)``
    and encodes the source identity of each received token.  The encoding is::

        deepep_ll_packed_recv_src_info[e][j] % num_tokens == original_token_index

    ``deepep_ll_recv_topk_ids`` has shape ``(num_tokens, top_k)`` and contains
    the original top-k expert assignments for each token, used to determine the
    correct k-slot for each dispatched token-expert pair.
    """

    @property
    def required_aux_dump_names(self) -> frozenset[str]:
        return frozenset(
            {
                "deepep_ll_masked_m",
                "deepep_ll_packed_recv_src_info",
                "deepep_ll_recv_topk_ids",
            }
        )

    @property
    def cross_rank_aux_names(self) -> frozenset[str]:
        return frozenset({"deepep_ll_recv_topk_ids"})

    def resolve_num_tokens(
        self,
        num_tokens: int,
        aux_tensors: dict[str, torch.Tensor],
    ) -> int:
        """Infer num_tokens from recv_topk_ids shape.

        The dumped ``ep_num_tokens`` for the LL path is unreliable because
        ``hidden_states.shape[0]`` in ``dispatch_b`` equals ``num_experts``
        (post-dispatch shape), not the pre-dispatch token count.

        We recover the true token count from ``recv_topk_ids.shape[0]``.
        """
        topk_ids: torch.Tensor = aux_tensors["deepep_ll_recv_topk_ids"]
        return topk_ids.shape[0]

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
        topk_ids: torch.Tensor = aux_tensors["deepep_ll_recv_topk_ids"].long()

        flat_src_info: torch.Tensor = _extract_valid_rows(
            packed_recv_src_info.unsqueeze(-1), masked_m
        ).squeeze(-1)

        expert_ids: torch.Tensor = _extract_expert_ids(masked_m)

        ep_rank: int = int(aux_tensors.get("_ep_rank", torch.tensor(0)).item())
        num_local_experts: int = masked_m.shape[0]
        expert_ids = expert_ids + ep_rank * num_local_experts

        token_ids: torch.Tensor = flat_src_info.long() % num_tokens

        k_indices: torch.Tensor = _lookup_k_indices(
            token_ids=token_ids,
            expert_ids=expert_ids,
            topk_ids=topk_ids,
        )

        forward_perm: torch.Tensor = token_ids * top_k + k_indices

        total_slots: int = num_tokens * top_k
        forward_perm[forward_perm >= total_slots] = -1
        forward_perm[k_indices < 0] = -1
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


def _extract_expert_ids(masked_m: torch.Tensor) -> torch.Tensor:
    """Build flat expert_id for each valid entry, matching _extract_valid_rows ordering."""
    num_experts: int = masked_m.shape[0]
    expected_m: int = int(masked_m.max().item()) if masked_m.numel() > 0 else 0
    if expected_m == 0:
        return torch.empty(0, dtype=torch.long, device=masked_m.device)

    arange_m: torch.Tensor = torch.arange(expected_m, device=masked_m.device)
    valid_mask: torch.Tensor = arange_m.unsqueeze(0) < masked_m.rename(None).unsqueeze(1)

    expert_arange: torch.Tensor = torch.arange(
        num_experts, dtype=torch.long, device=masked_m.device
    )
    expert_grid: torch.Tensor = expert_arange.unsqueeze(1).expand(
        num_experts, expected_m
    )
    return expert_grid[valid_mask]


def _lookup_k_indices(
    *,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """For each (token_id, expert_id) pair, find the k-position in topk_ids.

    Returns k_index for each entry, or -1 if the expert_id is not found.
    """
    k_indices: torch.Tensor = torch.full_like(token_ids, fill_value=-1)

    for i in range(token_ids.shape[0]):
        tok: int = token_ids[i].item()
        exp: int = expert_ids[i].item()
        row: torch.Tensor = topk_ids[tok]
        matches: torch.Tensor = (row == exp).nonzero(as_tuple=False)
        if matches.numel() > 0:
            k_indices[i] = matches[0, 0]

    return k_indices
