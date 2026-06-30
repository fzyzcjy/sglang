from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_dflash_verify_target_probs,
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton


def cap_correct_len(
    *,
    correct_len: torch.Tensor,
    layout: RaggedVerifyLayout,
) -> torch.Tensor:
    # Cutoff-only cap: commit at most ell_r = verify_len - 1 correct drafts per
    # request. Capping accept is lossless -- fewer correctly-verified drafts are
    # committed and the bonus (recomputed by callers at the capped index) is
    # still the target's true next token at the cap.
    ell_r = (layout.verify_lens.to(device=correct_len.device) - 1).to(correct_len.dtype)
    return torch.minimum(correct_len, ell_r)


def accept_greedy(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_layout: Optional[RaggedVerifyLayout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    if cutoff_layout is not None:
        correct_len = cap_correct_len(correct_len=correct_len, layout=cutoff_layout)
        row_ids = torch.arange(bs, device=target_predict.device)
        bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
    return correct_len, bonus


def accept_sampling(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_layout: Optional[RaggedVerifyLayout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    target_probs = build_dflash_verify_target_probs(
        next_token_logits=target_logits,
        sampling_info=sampling_info,
        draft_token_num=verify_num_draft_tokens,
        bs=bs,
        max_top_k=draft_input.max_top_k,
        uniform_top_k_value=draft_input.uniform_top_k_value,
    )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    uniform_samples = torch.rand((bs, gamma), dtype=torch.float32, device=device)
    uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=device)
    candidates_i64 = candidates.to(torch.int64)
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    correct_len = accept_token_num
    if cutoff_layout is not None:
        correct_len = cap_correct_len(correct_len=correct_len, layout=cutoff_layout)
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus


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
) -> tuple[torch.Tensor, torch.Tensor]:
    # Per-request accept (greedy argmax-match vs rejection sampling), dispatched
    # by batch composition. Both rules are lossless.
    greedy_mask = draft_block.greedy_mask
    # All-greedy fast path. is_all_greedy is host-side, so the branch is sync-free.
    all_greedy = sampling_info is None or sampling_info.is_all_greedy
    if all_greedy:
        return accept_greedy(
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
        return accept_sampling(
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
    greedy_len, greedy_bonus = accept_greedy(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_layout=cutoff_layout,
    )
    sampling_len, sampling_bonus = accept_sampling(
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
    return correct_len, bonus


def build_out_tokens(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    bs = draft_tokens.shape[0]
    out_tokens = torch.empty(
        (bs, verify_num_draft_tokens),
        dtype=torch.int64,
        device=draft_tokens.device,
    )
    out_tokens[:, :gamma].copy_(draft_tokens)
    out_tokens[:, gamma].fill_(0)
    out_tokens.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
    return out_tokens
