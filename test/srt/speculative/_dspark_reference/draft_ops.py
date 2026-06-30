# Upstream source: deepspec/eval/dspark/draft_ops.py
# Copied verbatim except:
# - DeepSpec imports replaced with local reference imports
# - `DraftProposal` base class from deepspec.eval.base_evaluator replaced with
#   a minimal local shim (we only need the dataclass fields for parity tests)
# - `logits_to_probs` imported from local sampling shim
from __future__ import annotations

from dataclasses import dataclass
from test.srt.speculative._dspark_reference.sampling import logits_to_probs
from typing import Optional

import torch


@dataclass
class DraftProposal:
    """Minimal shim for deepspec.eval.base_evaluator.DraftProposal."""

    draft_token_count: int
    verify_input_ids: torch.Tensor
    draft_probs: Optional[torch.Tensor]


@dataclass
class DSparkDraftProposal(DraftProposal):
    confidence_logits: Optional[torch.Tensor] = None


def build_dspark_proposal(
    model,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    temperature: float,
    confidence_threshold: float,
) -> DSparkDraftProposal:
    """Build a DSpark draft proposal from backbone hidden states.

    Args:
        model: Qwen3DSparkModel or Gemma4DSparkModel reference instance.
        draft_input_ids: [1, seq] input token ids (batch_size=1 required).
        block_hidden: [1, block_size, hidden_size] draft backbone output.
        block_size: number of draft positions (gamma).
        temperature: draft sampling temperature.
        confidence_threshold: not used in static-verify MVP (always 0).

    Returns:
        DSparkDraftProposal with verify_input_ids and draft_probs.
    """
    assert draft_input_ids.size(0) == 1, "build_dspark_proposal requires batch_size=1"
    proposal_hidden_states = block_hidden[:, :block_size, :]
    base_draft_logits = model.compute_logits(proposal_hidden_states)
    sampled_tokens, draft_logits = model.sample_draft_tokens(
        base_draft_logits,
        first_prev_token_ids=draft_input_ids[:, 0],
        temperature=temperature,
        hidden_states=proposal_hidden_states,
    )

    proposal_draft_tokens = int(block_size)

    verify_input_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :proposal_draft_tokens]],
        dim=1,
    )
    draft_probs = logits_to_probs(
        draft_logits[:, :proposal_draft_tokens, :],
        temperature,
    )
    return DSparkDraftProposal(
        draft_token_count=proposal_draft_tokens,
        verify_input_ids=verify_input_ids,
        draft_probs=draft_probs,
        confidence_logits=None,
    )
