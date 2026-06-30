from __future__ import annotations

import torch

from sglang.srt.utils.async_probe import maybe_detect_in_closed_range

# Confidence compute body (paper §5.2). The two-steps-prior causal barrier that
# fixes the verify budget K now lives in the host relay + carry (FutureMap
# confidence channel + HostConfidenceBudgetPlanner); this module only computes the
# per-step confidence the worker publishes and sorts on. Losslessness never depends
# on the confidence or the lag -- it is guaranteed by the accept-cap in
# _cap_correct_len.


def build_markov_embed_stack(
    *,
    anchor_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    markov_head,
    gamma: int,
) -> torch.Tensor:
    # Per-step prev tokens fed to the Markov head during the serial loop:
    # step 0 sees the anchor, step i (>0) sees the previously sampled token,
    # i.e. prev_seq = [anchor, s_0, ..., s_{gamma-2}] (the chapter's
    # off-by-one). markov_embed[:, i] = markov_w1(prev_seq[:, i]).
    prev_seq = torch.cat(
        [anchor_tokens.view(-1, 1), draft_tokens[:, : gamma - 1]], dim=1
    )
    return markov_head.get_prev_embeddings(prev_seq)


def compute_confidence(
    *,
    draft_hidden: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    confidence_head,
    markov_head,
    gamma: int,
) -> torch.Tensor:
    # Dense DSpark: the confidence head consumes the same post-norm draft
    # hidden that feeds base_logits (DeepSpec qwen3 modeling feeds the
    # post-norm output_hidden to both lm_head and the confidence head). For
    # with_markov heads it also takes the per-step markov_embed stack.
    assert confidence_head is not None
    if confidence_head.with_markov:
        markov_embed_stack = build_markov_embed_stack(
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            markov_head=markov_head,
            gamma=gamma,
        )
    else:
        markov_embed_stack = None
    confidence_raw = confidence_head(draft_hidden, markov_embed_stack)
    # apply_sts applies the per-position STS temperature (identity when no table
    # is loaded) then sigmoid, mapping the head logit to (0, 1). Losslessness does
    # not depend on the calibration, only on the scheduler being non-anticipating.
    confidence = confidence_head.apply_sts(confidence_raw)
    # Closed interval: sigmoid saturates to exactly 0.0/1.0 at fp32 for large
    # logits. Advisory only; async + gated, no per-step sync.
    maybe_detect_in_closed_range(confidence, 0.0, 1.0, "DSpark confidence")
    return confidence
