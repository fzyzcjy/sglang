from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult


def greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(step_logits, dim=-1)


class DsparkDraftSampler:
    """Capture-safe greedy proposal (model.compute_base_logits + Markov argmax) folded
    into the draft cuda graph, like DFlash #29395. Greedy-only/no-RNG: the worker reads
    ``out`` only for all-greedy batches, else eager. tp>1 is fine: capture runs inside
    the runner's graph-capture context, which makes compute_base_logits' vocab
    all-gather graph-safe (same as the target lm-head gather in every decode graph).

    When ``confidence_fn`` is set (the planner carries a confidence head), confidence
    is also computed in-graph right after the Markov block and written to
    ``confidence_out``. This is a correctness requirement, not an optimization: the
    dsv4 hook reads the ``_x_post_hc`` tap stashed by compute_base_logits, and only a
    same-graph read is guaranteed fresh -- an eager post-replay read would see the LAST
    captured bs-tier's capture-time buffer.
    """

    def __init__(self, *, model, gamma, max_bs, device, confidence_fn=None):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        # Proposed draft tokens [bs*gamma]: written in-graph, read after replay.
        self.out = torch.empty(
            (int(max_bs) * self.gamma,), dtype=torch.int64, device=device
        )
        # planner.compute_confidence_tensor-shaped callable, or None (no head).
        self.confidence_fn = confidence_fn
        # Confidence [max_bs, gamma]: written in-graph, read after replay.
        self.confidence_out = (
            torch.empty((int(max_bs), self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.gamma
        # Same path the worker runs eagerly: model base logits -> serial Markov argmax.
        base_logits = self.model.compute_base_logits(hidden_states).view(
            bs, self.gamma, -1
        )
        anchor = input_ids.view(bs, self.gamma)[:, 0]
        draft_tokens, _ = self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=hidden_states.view(bs, self.gamma, -1),
            sampler=greedy_step_sampler,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=hidden_states.view(bs, self.gamma, -1),
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
            )
            self.confidence_out[:bs].copy_(confidence)


def make_next_draft_input(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    # The next step's draft state is just the bonus tokens + committed seq lens.
    # The next anchor's target hidden is read back from the draft KV pool (it was
    # written there by the commit injection), not relayed through the spec input,
    # so the legacy Eagle-shaped ``hidden_states`` slot stays the empty placeholder.
    return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
    # Per-request greedy mask (review M4). A row is greedy iff top_k <= 1,
    # which mirrors SamplingBatchInfo.is_all_greedy (= all rows greedy). In a
    # mixed batch the previous batch-level branch forced greedy rows onto the
    # rejection-sampling accept path; the per-row mask lets greedy rows use
    # argmax-match accept and sampling rows use the chain kernel.
    if sampling_info is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    return (sampling_info.top_ks <= 1).view(-1)


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    sampling_info,
    markov_head,
    device: torch.device,
) -> DraftBlockResult:
    bs = base_logits.shape[0]
    greedy_mask = resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
    # any_sampling == not is_all_greedy, read host-side (is_all_greedy is a
    # Python bool on sampling_info) so this branch draws no GPU sync. No
    # sampling_info -> all-greedy fast path (argmax only, no RNG draw).
    any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()

    if sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    if not any_sampling:
        # All-greedy batch: argmax only. Crucially this must NOT draw random
        # numbers (no torch.multinomial), otherwise the RNG stream diverges
        # from the off/cutoff/a+b path and breaks byte-identical losslessness.
        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            # Per-row mixed sampling: greedy rows take argmax, sampling rows draw
            # from the temperature-scaled softmax, so a mixed batch keeps each
            # request's own draft distribution. With at least one sampling row this
            # matches the all-sampling RNG draw count (one draw per step), so the
            # all-sampling path stays byte-identical.
            probs = torch.softmax(step_logits.float() / temperatures[:, None], dim=-1)
            if fast_sampling:
                # Reference Gumbel-max trick: argmax(probs / Exp(1)) ~ Categorical(probs),
                # one fused pass with no full-vocab CDF and no D2H sync, unlike
                # torch.multinomial. Setting greedy rows' noise to 1 makes their
                # argmax(probs / 1) == argmax(probs) == argmax(logits) (softmax is
                # monotone), so this single argmax also yields the greedy token and
                # the separate greedy torch.argmax(step_logits) drops out (the P1
                # argmax in the profile). exponential_ still fills every row, so the
                # per-step draw count is unchanged; greedy rows discard their noise.
                noise = torch.empty_like(probs).exponential_(1)
                noise = torch.where(greedy_mask[:, None], 1.0, noise)
                return probs.div_(noise).argmax(dim=-1)
            else:
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

    draft_tokens, corrected_logits = markov_head.sample_block(
        base_logits,
        first_prev_tokens=anchor_tokens,
        hidden_states=draft_hidden,
        sampler=sampler,
    )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
    )
