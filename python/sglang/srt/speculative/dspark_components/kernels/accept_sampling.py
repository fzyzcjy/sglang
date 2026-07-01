from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_dflash_verify_target_probs,
)
from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    CapCorrectLen,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_ACCEPT_SAMPLING.get()


class AcceptSampling:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        gamma: int,
        verify_num_draft_tokens: int,
        cutoff_layout: Optional[RaggedVerifyLayout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        gamma: int,
        verify_num_draft_tokens: int,
        cutoff_layout: Optional[RaggedVerifyLayout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "AcceptSampling.triton is not implemented yet; run with "
            "SGLANG_DSPARK_KERNEL_ACCEPT_SAMPLING=torch until the triton kernel lands."
        )


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_layout is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, layout=cutoff_layout
        )
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus, cap_trim_lens
