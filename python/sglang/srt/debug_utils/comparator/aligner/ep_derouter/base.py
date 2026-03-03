from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DeRouterPlugin(ABC):
    """Base class for de-routing routed MoE tensors back to global (token, top_k) order.

    Subclasses implement:

    - ``required_aux_dump_names``: the dump tensor names this plugin needs.
    - ``flatten_routed_tensor``: reshape non-2D tensors into flat 2D form
      (default is identity — only DeepEP LL overrides this).
    - ``compute_forward_permutation``: return an index mapping from the flat
      routed tensor to the canonical ``[num_tokens * top_k, ...]`` output.

    The entrypoint applies the permutation uniformly for all plugins.
    """

    @property
    @abstractmethod
    def required_aux_dump_names(self) -> frozenset[str]:
        """Dump tensor names that must be present in aux_tensors."""
        ...

    @property
    def cross_rank_aux_names(self) -> frozenset[str]:
        """Aux names that may be loaded from any rank (not just the current one).

        Override this when an auxiliary tensor is only dumped by a subset of
        ranks (e.g. ``topk_ids`` in DP-attention, where only the token-owning
        rank dumps it).  The executor will try all ranks for these names.
        """
        return frozenset()

    def flatten_routed_tensor(
        self,
        routed_tensor: torch.Tensor,
        aux_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reshape the routed tensor into 2D ``[num_routed, ...]`` form.

        Override for non-2D routed tensors (e.g. DeepEP LL 3D).
        Default: return the tensor unchanged.
        """
        return routed_tensor

    def resolve_num_tokens(
        self,
        num_tokens: int,
        aux_tensors: dict[str, torch.Tensor],
    ) -> int:
        """Optionally override *num_tokens* using auxiliary data.

        The default returns the value as-is.  Plugins may override this when
        the dumped ``ep_num_tokens`` is unreliable (e.g. the DeepEP LL path
        where ``hidden_states.shape[0]`` is ``num_experts`` rather than the
        pre-dispatch token count).
        """
        return num_tokens

    @abstractmethod
    def compute_forward_permutation(
        self,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
        num_routed: int,
    ) -> torch.Tensor:
        """Compute forward permutation from flat routed positions to output slots.

        Args:
            aux_tensors: Auxiliary tensors dumped alongside (e.g. sorted_token_ids).
            num_tokens: Total number of tokens before dispatch.
            top_k: Number of experts per token.
            num_routed: Number of rows in the flattened routed tensor.

        Returns:
            1-D ``torch.long`` tensor of shape ``[num_routed]``, where
            ``result[i]`` is the output slot index for ``flat_routed[i]``.
            Use ``-1`` to mark padding / discard positions.
            Output slot values are in ``[0, num_tokens * top_k)``.
        """
