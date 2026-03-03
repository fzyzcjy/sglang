from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin


class DeepEPNormalOutputIndexDeRouter(DeRouterPlugin):
    """De-router for SGLang DeepEP Normal dispatch path (deep_gemm contiguous).

    The deep_gemm contiguous path uses ``ep_scatter()`` + ``output_index``
    instead of ``src2dst``.  ``output_index`` has shape ``[num_tokens, top_k]``
    where ``output_index[tok][k] = dispatch_pos`` (row index in the contiguous
    buffer after scatter).  Only entries where ``recv_topk_ids[tok][k] >= 0``
    are valid; the rest contain uninitialized memory from ``torch.empty_like``.

    We invert valid entries to get ``forward_perm[dispatch_pos] = tok * top_k + k``.
    """

    @property
    def required_aux_dump_names(self) -> frozenset[str]:
        return frozenset({"deepep_normal_output_index", "deepep_normal_recv_topk_ids"})

    @property
    def cross_rank_aux_names(self) -> frozenset[str]:
        return frozenset({"deepep_normal_recv_topk_ids"})

    def compute_forward_permutation(
        self,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
        num_routed: int,
    ) -> torch.Tensor:
        output_index: torch.Tensor = aux_tensors["deepep_normal_output_index"].long()
        recv_topk_ids: torch.Tensor = aux_tensors["deepep_normal_recv_topk_ids"].long()

        forward_perm: torch.Tensor = torch.full(
            (num_routed,), -1, dtype=torch.long, device=output_index.device
        )

        tok_indices: torch.Tensor = torch.arange(num_tokens, device=output_index.device)
        for k in range(top_k):
            canonical_flat: torch.Tensor = tok_indices * top_k + k
            dispatch_pos: torch.Tensor = output_index[:, k]
            expert_ids: torch.Tensor = recv_topk_ids[:, k]
            valid: torch.Tensor = (
                (expert_ids >= 0) & (dispatch_pos >= 0) & (dispatch_pos < num_routed)
            )
            forward_perm[dispatch_pos[valid]] = canonical_flat[valid]

        return forward_perm
