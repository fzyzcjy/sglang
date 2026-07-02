from typing import Optional

import msgspec
import torch


class VerifyWindow(msgspec.Struct, frozen=True):
    positions_2d: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_cache_loc_2d: torch.Tensor


class RaggedVerifyWindow(msgspec.Struct, frozen=True):
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_ids: torch.Tensor


class TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor


class DraftForwardResult(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    can_run_graph: bool


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor] = None
    # dsv4 post-hc_head PRE-norm tap from the eager compute_base_logits, fed
    # explicitly to compute_confidence (None on the fold path -- the captured
    # sampler threads its own tap in-graph -- and for dense drafts).
    confidence_tap: Optional[torch.Tensor] = None
    # True iff this proposal came from the captured greedy draft graph, i.e.
    # the sampler's draft-token buffer holds THIS step's drafts (gate for the
    # verify-graph accept fold, which reads that buffer in-graph).
    folded: bool = False
