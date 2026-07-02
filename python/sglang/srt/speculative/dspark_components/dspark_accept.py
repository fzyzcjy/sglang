from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.dspark_components.kernels.accept_greedy import AcceptGreedy
from sglang.srt.speculative.dspark_components.kernels.accept_sampling import (
    AcceptSampling,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


def accept_draft_tokens(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_block: DraftBlockResult,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_layout: Optional[RaggedVerifyLayout] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Per-request accept (greedy argmax-match vs rejection sampling), dispatched
    # by batch composition. Both rules are lossless. The third return is the
    # per-request cap_trim_lens (correct drafts the confidence cap dropped).
    greedy_mask = draft_block.greedy_mask
    # All-greedy fast path. is_all_greedy is host-side, so the branch is sync-free.
    all_greedy = sampling_info is None or sampling_info.is_all_greedy
    if all_greedy:
        return AcceptGreedy.execute(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_layout=cutoff_layout,
        )
    draft_probs = torch.softmax(
        draft_block.corrected_logits.float() / draft_block.temperatures[:, None, None],
        dim=-1,
    )
    # All-sampling fast path: no greedy rows -> only the chain kernel (host-side, sync-free).
    if not sampling_info.is_any_greedy:
        return AcceptSampling.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_layout=cutoff_layout,
        )
    # Mixed: run both rules and select per row by greedy_mask.
    greedy_len, greedy_bonus, greedy_trim = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_layout=cutoff_layout,
    )
    sampling_len, sampling_bonus, sampling_trim = AcceptSampling.execute(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_layout=cutoff_layout,
    )
    correct_len = torch.where(
        greedy_mask, greedy_len.to(sampling_len.dtype), sampling_len
    )
    bonus = torch.where(greedy_mask, greedy_bonus, sampling_bonus)
    cap_trim_lens = torch.where(
        greedy_mask, greedy_trim.to(sampling_trim.dtype), sampling_trim
    )
    return correct_len, bonus, cap_trim_lens
