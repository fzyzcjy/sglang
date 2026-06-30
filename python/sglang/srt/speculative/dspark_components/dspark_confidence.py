from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.utils.async_probe import maybe_detect_in_closed_range

# Confidence relay ring (paper §5.2 two-steps-prior causal barrier). The K-source
# reads the confidence stashed _CONFIDENCE_RELAY_LAG_STEPS decode steps earlier so
# the verify budget K is causally independent of the current step's just-sampled
# draft tokens. The ring depth must exceed the lag so the slot being read this step
# is never the one just overwritten (slot s is rewritten at step s + depth).
#
# Lag-depth basis (honest, see report): this worker computes verify_lens INLINE
# within each decode step (_forward_decode: propose -> _stash_confidence ->
# _schedule_verify_lens, all stream-ordered on the forward stream), so there is no
# multi-step ZOS/overlap pipeline in THIS path forcing a natural lag -- without the
# ring the K-source would read the confidence stashed THIS step (lag 0). The ring's
# lag is therefore a DELIBERATELY imposed causal barrier reproducing the paper's
# two-steps-prior design, NOT an alignment to an emergent pipeline depth. Any lag
# >= 1 already yields the barrier (K independent of the current step's tokens);
# 2 reproduces the paper literally. The optimal value depends on the deployed ZOS /
# overlap configuration, which cannot be measured on CPU -- it is a single tunable
# constant and should be validated on GPU. Losslessness never depends on the lag:
# it is guaranteed by the accept-cap in _cap_correct_len.
_CONFIDENCE_RELAY_LAG_STEPS: int = 2
_CONFIDENCE_RELAY_RING_DEPTH: int = _CONFIDENCE_RELAY_LAG_STEPS + 1
# Sentinel for a ring-slot row that was never written (or written for a different
# request), used by the per-row identity guard to mask stale rows before forming K.
_CONFIDENCE_RELAY_UNSET_SEQ_LEN: int = -1


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


class ConfidenceRelay:
    def __init__(self, *, device, gamma: int, model_runner) -> None:
        self.device = device
        self.gamma = int(gamma)
        self.model_runner = model_runner
        self.ring: Optional[torch.Tensor] = None
        self.ring_seq_lens: Optional[torch.Tensor] = None
        self.step_ct: int = 0

    def ensure_buffers(self, *, confidence: torch.Tensor) -> None:
        if self.ring is not None:
            return
        req_pool_size = int(self.model_runner.req_to_token_pool.req_to_token.shape[0])
        # Step-indexed ring (paper §5.2): depth slots, each [req_pool_size, gamma],
        # scatter-written by req_pool_indices on the forward stream. Reads of a slot
        # written _CONFIDENCE_RELAY_LAG_STEPS steps ago are stream-ordered after that
        # write, so the relay stays no-synchronize on the critical path (no event /
        # D2H stream needed -- the prior single-buffer relay's gated host copy is
        # folded away with pull_confidence_history).
        self.ring = torch.empty(
            (_CONFIDENCE_RELAY_RING_DEPTH, req_pool_size, self.gamma),
            dtype=confidence.dtype,
            device=self.device,
        )
        # Per-slot per-row request identity (H1): the prefix_len stamped when the
        # row was written, used to mask ring rows whose lag-steps-prior occupant was
        # a different request before they enter K. Init to the unset sentinel.
        self.ring_seq_lens = torch.full(
            (_CONFIDENCE_RELAY_RING_DEPTH, req_pool_size),
            _CONFIDENCE_RELAY_UNSET_SEQ_LEN,
            dtype=torch.int64,
            device=self.device,
        )

    def stash(
        self,
        *,
        req_pool_indices: torch.Tensor,
        confidence: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> None:
        # Write this step's confidence into the current ring slot
        # (step_ct % depth) on the forward stream and stamp the per-row prefix_len
        # so a later two-steps-prior read can tell whether the slot's occupant is
        # still the same request (H1). step_ct is advanced once per decode step in
        # _forward_decode, after both the write and the lagged read.
        self.ensure_buffers(confidence=confidence)
        write_slot = self.step_ct % _CONFIDENCE_RELAY_RING_DEPTH
        self.ring[write_slot, req_pool_indices] = confidence
        self.ring_seq_lens[write_slot].fill_(_CONFIDENCE_RELAY_UNSET_SEQ_LEN)
        self.ring_seq_lens[write_slot, req_pool_indices] = prefix_lens.to(torch.int64)

    def two_steps_prior_k_survival(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # K-source survival for the verify budget: cumprod of the confidence stashed
        # _CONFIDENCE_RELAY_LAG_STEPS decode steps ago (paper §5.2 two-steps-prior
        # causal barrier). Reads the lagged ring slot device->device on the forward
        # stream (stream-ordered after that slot's write, so no synchronize), so the
        # budget K cannot depend on the current step's just-sampled draft tokens.
        #
        # H1 identity guard + cold-start fallback (paper-unspecified engineering
        # choice): a ring slot is indexed by req-pool row only, and the row's
        # occupant lag steps ago may be a DIFFERENT request (or none, at cold
        # start). A row is the SAME request iff its stamped prefix_len is a valid
        # ancestor of the current prefix_len -- present (>= 0), strictly smaller (a
        # live request commits >= 1 token/step), and within lag steps of growth
        # (<= lag * (gamma + 1)). Stale / new / just-switched rows fall back to
        # verify-all (survival = 1.0 at every position) so they are admitted into
        # the full window rather than carrying a stranger's confidence into K.
        if self.ring is None or self.ring_seq_lens is None:
            return None
        read_slot = (
            self.step_ct - _CONFIDENCE_RELAY_LAG_STEPS
        ) % _CONFIDENCE_RELAY_RING_DEPTH
        lagged_confidence = self.ring[read_slot, req_pool_indices]
        k_survival = torch.cumprod(lagged_confidence.to(torch.float32), dim=1)

        stamped_seq_lens = self.ring_seq_lens[read_slot, req_pool_indices]
        growth = prefix_lens.to(torch.int64) - stamped_seq_lens
        max_growth = _CONFIDENCE_RELAY_LAG_STEPS * (self.gamma + 1)
        fresh = ((stamped_seq_lens >= 0) & (growth >= 1) & (growth <= max_growth)).view(
            -1, 1
        )
        return torch.where(fresh, k_survival, torch.ones_like(k_survival))

    def current_live_sort_survival(
        self, *, req_pool_indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        # Sort-source survival for the rank/truncate (paper §5.2: "sorted by the
        # actual up-to-date confidence"): cumprod of THIS step's just-stashed
        # confidence, read from the current ring slot (step_ct % depth) device->device
        # on the forward stream. Always present for the current batch (it was stashed
        # this step), so no identity guard / fallback is needed here. Distinct from
        # the K-source, which is the two-steps-prior survival; admission is ordered by
        # the current confidence while the budget K is set by the lagged confidence.
        if self.ring is None:
            return None
        write_slot = self.step_ct % _CONFIDENCE_RELAY_RING_DEPTH
        current_confidence = self.ring[write_slot, req_pool_indices]
        return torch.cumprod(current_confidence.to(torch.float32), dim=1)

    def advance_step(self) -> None:
        self.step_ct += 1
