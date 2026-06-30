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
    seq_lens_cpu: torch.Tensor


class TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    # None on the captured greedy fast path (only _accept_sampling reads it).
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor


class DraftForwardResult(msgspec.Struct, frozen=True):
    # Output of the unified draft block forward. ``raw_hidden`` is the model's
    # un-reshaped backbone hidden (dense: 2-D ``[bs*gamma, d]``; dsv4: 3-D
    # ``[bs*gamma, hc, d]``); it is fed straight to ``compute_base_logits`` (the model
    # owns the matmul / hc-collapse). ``draft_hidden_3d`` is ``raw_hidden.view(bs,
    # gamma, -1)``, the dense markov / dense confidence input. dsv4's markov takes no
    # hidden and its confidence reads the model-stashed ``_x_post_hc``, so dsv4 ignores
    # ``draft_hidden_3d``.
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    # True iff the draft forward replayed a cuda graph (in-graph sampler wrote out).
    can_run_graph: bool


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: Optional[torch.Tensor]
