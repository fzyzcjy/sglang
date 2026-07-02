from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.dspark_components.kernels.sample_step_tokens import (
    SampleStepTokens,
)

# ADHOC-SHAPE-PRINT (workflow ii, revert after capture)
from sglang.srt.debug_utils.dumper import get_tensor_info as _gti

_ADHOC_SAMPLE_PRINTS = [0]


def greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(step_logits, dim=-1)


class DsparkDraftSampler:

    def __init__(self, *, model, gamma, max_bs, device, confidence_fn=None, out=None):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        # An external ``out`` (the verify epilogue's draft_tokens_buf) makes the
        # verify graph read the same stable memory the draft graph writes.
        if out is not None:
            assert out.shape == (int(max_bs) * self.gamma,) and out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (int(max_bs) * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((int(max_bs), self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.gamma
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
    return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
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
    any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()

    if sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    if not any_sampling:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            # Per-row mixed sampling: greedy rows take argmax, sampling rows draw
            # from the temperature-scaled softmax, so a mixed batch keeps each
            # request's own draft distribution. With at least one sampling row this
            # matches the all-sampling RNG draw count (one draw per step), so the
            # all-sampling path stays byte-identical.
            if fast_sampling:
                # Reference Gumbel-max trick: argmax(probs / Exp(1)) ~ Categorical(probs),
                # one fused pass with no full-vocab CDF and no D2H sync, unlike
                # torch.multinomial. Setting greedy rows' noise to 1 makes their
                # argmax(probs / 1) == argmax(probs) == argmax(logits) (softmax is
                # monotone), so this single argmax also yields the greedy token and
                # the separate greedy torch.argmax(step_logits) drops out (the P1
                # argmax in the profile). The exponential_ draw stays here (elementwise,
                # cheap) so the per-step RNG count is unchanged: empty_like(step_logits,
                # fp32) has the same shape/dtype/count as the old empty_like(probs) draw,
                # keeping downstream accept coins byte-identical. The kernel then does the
                # softmax + argmax reductions (deleted from this path), consuming the noise.
                # Contiguous [bs, vocab] draw (not empty_like: step_logits is now the
                # strided full[..., :vocab] view). numel is unchanged, so the RNG stream --
                # and the downstream accept coins -- stay byte-identical.
                exp_noise = torch.empty(
                    step_logits.shape, dtype=torch.float32, device=step_logits.device
                ).exponential_(1)
                if _ADHOC_SAMPLE_PRINTS[0] < 24:  # ADHOC-SHAPE-PRINT
                    _ADHOC_SAMPLE_PRINTS[0] += 1
                    print(
                        f"[SHAPE][sample #{_ADHOC_SAMPLE_PRINTS[0]}] "
                        f"step_logits stride={tuple(step_logits.stride())} contig={step_logits.is_contiguous()} {_gti(step_logits)} || "
                        f"exp_noise {_gti(exp_noise)} || temps {_gti(temperatures)} || greedy {_gti(greedy_mask)}",
                        flush=True,
                    )
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    greedy_mask=greedy_mask,
                    exp_noise=exp_noise,
                )
            else:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
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
