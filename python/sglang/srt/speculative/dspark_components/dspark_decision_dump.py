from __future__ import annotations

import json
import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class DsparkDecisionDumper:
    # Runtime observability probe for the confidence + verify-length schedule
    # (read-only; never touches the accept path). Off unless
    # SGLANG_DSPARK_DEBUG_MAIN_OUTPUT is set; rank-0 only. Emits one grep-friendly
    # ``DSPARK_DEBUG_MAIN_OUTPUT=<compact json>`` line per decode step carrying the whole
    # decision (global budget + per-request raw confidence, cumprod survival, chosen
    # verify_len) alongside the realized outcome (accept length, correct drafts, the
    # confidence-cap trim). A post-processing script greps these lines to check the
    # scheduler "looks normal": budget shrinks per-request as the batch grows, longer
    # windows land on the high-survival requests, and the two-steps-prior budget lag.
    #
    # Each per-request entry also carries ``rid`` (the external request id, row-aligned
    # with ``req_pool_indices``). ``req`` is the engine pool slot, which is recycled and
    # cannot be joined back to a question; ``rid`` is stable, so post-processing can group
    # dump rows by a dataset label encoded in the rid (e.g. ``gsm8k::0007::r0``).
    #
    # Design mirrors ConfidenceMetricsProbe: decoupled from the planner (fed only
    # primitives) so the worker stays a one-field, one-call wiring. The whole dump is
    # a debug tap gated behind an env flag, so the batch of D2H copies it does is
    # acceptable (production keeps it off).

    def __init__(
        self,
        *,
        gamma: int,
        verify_num_draft_tokens: int,
        tp_rank: int,
    ) -> None:
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = int(verify_num_draft_tokens)
        self.tp_rank = int(tp_rank)

    def maybe_dump(
        self,
        *,
        forward_ct: Optional[int],
        bs: int,
        mode: str,
        budget: Optional[int],
        lag_steps: Optional[int],
        verify_lens: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        rids: Optional[list[str]],
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        if not envs.SGLANG_DSPARK_DEBUG_MAIN_OUTPUT.get():
            return
        if self.tp_rank != 0:
            return
        if bs <= 0:
            return

        record = self._build_record(
            forward_ct=forward_ct,
            bs=bs,
            mode=mode,
            budget=budget,
            lag_steps=lag_steps,
            verify_lens=verify_lens,
            confidence=confidence,
            req_pool_indices=req_pool_indices,
            rids=rids,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            bonus_tokens=bonus_tokens,
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            commit_lens=commit_lens,
        )
        logger.info(
            "DSPARK_DEBUG_MAIN_OUTPUT=%s", json.dumps(record, separators=(",", ":"))
        )

    def _build_record(
        self,
        *,
        forward_ct: Optional[int],
        bs: int,
        mode: str,
        budget: Optional[int],
        lag_steps: Optional[int],
        verify_lens: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        rids: Optional[list[str]],
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> dict:
        # verify_lens is the layout's DEVICE tensor (real per-request windows). The
        # sync-free device path leaves layout.verify_lens_cpu None to stay off the
        # forward stream, so the dump must D2H the device tensor here (one debug-only
        # copy) -- reading verify_lens_cpu would miss the real ragged schedule and
        # falsely report a uniform block. None layout (static / cold-start ragged)
        # genuinely verifies the uniform full block: treat every request as gamma+1.
        if verify_lens is None:
            verify_len_per_req = [self.verify_num_draft_tokens] * bs
        else:
            verify_len_per_req = [int(v) for v in verify_lens.detach().to("cpu").tolist()]

        req_ids = req_pool_indices.detach().to("cpu").tolist()
        prefixes = prefix_lens.detach().to("cpu").tolist()
        draft_rows = draft_tokens.detach().to("cpu").tolist()
        bonus = bonus_tokens.detach().to("cpu").tolist()
        correct = correct_len.detach().to("cpu").tolist()
        cap_trim = cap_trim_lens.detach().to("cpu").tolist()
        commit = commit_lens.detach().to("cpu").tolist()

        # survival = cumprod of the per-position conditional confidence: the head emits
        # P(draft_k correct | drafts 0..k-1 correct), so the prefix product is the true
        # P(first k+1 drafts all accepted). Dump BOTH so analysis can confirm the
        # cumprod is applied and that longer windows track high survival.
        if confidence is not None:
            conf_rows = confidence.detach().float().to("cpu")
            survival_rows = torch.cumprod(conf_rows, dim=1).tolist()
            conf_rows = conf_rows.tolist()
        else:
            conf_rows = None
            survival_rows = None

        reqs: list[dict] = []
        for row in range(bs):
            entry = {
                "rid": None if rids is None else rids[row],
                "req": int(req_ids[row]),
                "prefix": int(prefixes[row]),
                "verify_len": int(verify_lens[row]),
                # acc_len (incl. bonus) = correct drafts committed + 1 bonus token.
                "acc_len": int(commit[row]),
                "correct_drafts": int(correct[row]),
                # cap_trim = target-correct drafts the confidence cap dropped this step
                # (>0 means the schedule cut this request shorter than it could accept;
                # only observable in cap-accept, always 0 in static/compact).
                "cap_trim": int(cap_trim[row]),
                "draft_tokens": [int(t) for t in draft_rows[row]],
                "bonus_token": int(bonus[row]),
            }
            if conf_rows is not None:
                entry["confidence"] = [round(float(p), 4) for p in conf_rows[row]]
                entry["survival"] = [round(float(p), 4) for p in survival_rows[row]]
            reqs.append(entry)

        num_verify_tokens = sum(verify_lens)
        return {
            "forward_ct": None if forward_ct is None else int(forward_ct),
            "bs": int(bs),
            "gamma": self.gamma,
            "mode": mode,
            "budget": None if budget is None else int(budget),
            "lag_steps": None if lag_steps is None else int(lag_steps),
            "num_verify_tokens": int(num_verify_tokens),
            "avg_verify_len": round(num_verify_tokens / bs, 4),
            "reqs": reqs,
        }
