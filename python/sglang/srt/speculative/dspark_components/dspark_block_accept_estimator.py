from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import msgspec
import torch

from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

logger = logging.getLogger(__name__)

_GATHER_ROW_CHUNK = 128
_STATE_SWEEP_INTERVAL = 1024
_STATE_EXPIRE_STEPS = 4096
_FLUSH_EVERY_STEPS = 16


class _PendingBlock(msgspec.Struct):
    forward_ct: int
    anchor_pos: int
    window: int
    trimmed_tokens: List[int]
    next_offset: int


class _RequestState(msgspec.Struct):
    expected_seq_len: int = -1
    last_seen_ct: int = 0
    pending: List[_PendingBlock] = []


class _GatherPlan(msgspec.Struct):
    row_indices: List[int] = []
    token_indices: List[int] = []


class BlockAcceptEstimateRecorder:
    def __init__(self, *, path: str, gamma: int) -> None:
        self._gamma = gamma
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w")
        self._states: dict[str, _RequestState] = {}
        self._steps_since_flush = 0
        self._observed_step_ct = 0
        self._discontinuity_drop_ct = 0
        self._skipped_step_ct = 0
        self._warned_skip_reasons: set[str] = set()
        logger.info(
            "DSPARK block accept estimate recorder enabled: path=%s gamma=%d",
            path,
            gamma,
        )

    def observe_verify_step(
        self,
        *,
        forward_ct: int,
        rids: List[str],
        draft_tokens: torch.Tensor,
        corrected_logits: Optional[torch.Tensor],
        draft_temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        target_logits: torch.Tensor,
        target_temperatures: torch.Tensor,
        truncated_sampling_mask: Optional[torch.Tensor],
        logits_adjustments_are_noop: bool,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        bonus: torch.Tensor,
        prefix_lens: torch.Tensor,
        layout: Optional[RaggedVerifyLayout],
    ) -> None:
        if not logits_adjustments_are_noop:
            self._skip_step(
                reason="non-noop logits adjustments (penalizer/logit_bias/grammar) "
                "in batch; cross-step conditioning of the gathered target "
                "probabilities would be state-dependent"
            )
            return
        if corrected_logits is None:
            self._skip_step(reason="corrected_logits unavailable (folded draft path)")
            return

        gamma = self._gamma
        rows_per_request = gamma + 1
        bs = len(rids)
        assert draft_tokens.shape == (bs, gamma)
        assert corrected_logits.shape[0] == bs and corrected_logits.shape[1] == gamma
        assert target_logits.shape[0] == bs * rows_per_request

        correct_lens = correct_len.tolist()
        cap_trims = cap_trim_lens.tolist()
        bonus_tokens = bonus.tolist()
        drafts = draft_tokens.tolist()
        greedy_rows = greedy_mask.tolist()
        if truncated_sampling_mask is not None:
            truncated_rows = truncated_sampling_mask.tolist()
        else:
            truncated_rows = [False] * bs
        seq_lens = prefix_lens.tolist()
        if layout is not None:
            verify_lens = layout.verify_lens.tolist()
        else:
            verify_lens = [rows_per_request] * bs

        target_plan = _GatherPlan()
        draft_plan = _GatherPlan()
        row_records: List[dict[str, Any]] = []
        row_pending_gathers: List[List[tuple[_PendingBlock, int, int]]] = []

        for b in range(bs):
            rid = rids[b]
            state = self._states.setdefault(rid, _RequestState())
            state.last_seen_ct = forward_ct

            cl = int(correct_lens[b])
            window = int(verify_lens[b]) - 1
            seq_len = int(seq_lens[b])
            assert 0 <= cl <= window <= gamma

            if state.expected_seq_len >= 0 and seq_len != state.expected_seq_len:
                if state.pending:
                    self._discontinuity_drop_ct += len(state.pending)
                    state.pending = []
            state.expected_seq_len = seq_len + cl + 1

            if greedy_rows[b] or truncated_rows[b]:
                if truncated_rows[b] and not greedy_rows[b]:
                    self._warn_once(
                        reason="requests with top-k/top-p/min-p sampling are "
                        "excluded per-row; the estimator only supports "
                        "pure-temperature sampling (processed target distribution "
                        "would differ from plain softmax(logits/T))"
                    )
                state.pending = []
                row_pending_gathers.append([])
                continue

            realized = drafts[b][:cl] + [bonus_tokens[b]]

            record: dict[str, Any] = {
                "rid": rid,
                "fct": forward_ct,
                "w": window,
                "cl": cl,
                "ct": int(cap_trims[b]),
            }
            censored = cl == window and window < gamma
            if censored:
                trimmed_tokens = drafts[b][window:gamma]
                state.pending.append(
                    _PendingBlock(
                        forward_ct=forward_ct,
                        anchor_pos=seq_len - 1,
                        window=window,
                        trimmed_tokens=trimmed_tokens,
                        next_offset=window + 1,
                    )
                )
                record["q_lp_slot"] = len(draft_plan.row_indices)
                record["trimmed_tokens"] = trimmed_tokens
                for offset in range(window + 1, gamma + 1):
                    draft_plan.row_indices.append(b * gamma + offset - 1)
                    draft_plan.token_indices.append(drafts[b][offset - 1])

            pending_gathers: List[tuple[_PendingBlock, int, int]] = []
            kept_pending: List[_PendingBlock] = []
            for block in state.pending:
                diverged = False
                while block.next_offset <= gamma:
                    position = block.anchor_pos + block.next_offset
                    row = position - seq_len
                    assert row >= 0
                    if row > cl:
                        break
                    token = block.trimmed_tokens[block.next_offset - block.window - 1]
                    target_plan.row_indices.append(b * rows_per_request + row)
                    target_plan.token_indices.append(token)
                    pending_gathers.append((block, block.next_offset, realized[row]))
                    block.next_offset += 1
                    if realized[row] != token:
                        diverged = True
                        break
                if not diverged and block.next_offset <= gamma:
                    kept_pending.append(block)
            state.pending = kept_pending

            row_records.append(record)
            row_pending_gathers.append(pending_gathers)

        draft_logprobs = _gather_token_logprobs(
            logits=corrected_logits.reshape(bs * gamma, -1),
            temperatures=draft_temperatures.reshape(bs)
            .to(torch.float32)
            .repeat_interleave(gamma),
            plan=draft_plan,
        )
        target_logprobs = _gather_token_logprobs(
            logits=target_logits,
            temperatures=target_temperatures.reshape(bs)
            .to(torch.float32)
            .repeat_interleave(rows_per_request),
            plan=target_plan,
        )

        target_cursor = 0
        record_cursor = 0
        for b in range(bs):
            pending_gathers = row_pending_gathers[b]
            if greedy_rows[b]:
                assert not pending_gathers
                continue
            record = row_records[record_cursor]
            record_cursor += 1
            q_slot = record.pop("q_lp_slot", None)
            if q_slot is not None:
                num_trimmed = gamma - record["w"]
                record["q_lp"] = draft_logprobs[q_slot : q_slot + num_trimmed]
            if pending_gathers:
                record["pg"] = [
                    [
                        block.forward_ct,
                        offset,
                        target_logprobs[target_cursor + i],
                        block.trimmed_tokens[offset - block.window - 1],
                        realized_token,
                    ]
                    for i, (block, offset, realized_token) in enumerate(pending_gathers)
                ]
                target_cursor += len(pending_gathers)
            self._file.write(json.dumps(record) + "\n")
        assert record_cursor == len(row_records)
        assert target_cursor == len(target_plan.row_indices)

        self._observed_step_ct += 1
        self._steps_since_flush += 1
        if self._steps_since_flush >= _FLUSH_EVERY_STEPS:
            self._file.flush()
            self._steps_since_flush = 0
        if self._observed_step_ct % _STATE_SWEEP_INTERVAL == 0:
            self._sweep_states(forward_ct=forward_ct)

    def _sweep_states(self, *, forward_ct: int) -> None:
        expired = [
            rid
            for rid, state in self._states.items()
            if forward_ct - state.last_seen_ct > _STATE_EXPIRE_STEPS
        ]
        for rid in expired:
            del self._states[rid]

    def _skip_step(self, *, reason: str) -> None:
        self._skipped_step_ct += 1
        self._warn_once(
            reason=f"skipping step: {reason} (pending blocks of affected requests "
            "are dropped by the seq-len continuity check)"
        )

    def _warn_once(self, *, reason: str) -> None:
        if reason not in self._warned_skip_reasons:
            self._warned_skip_reasons.add(reason)
            logger.warning(
                "DSPARK block accept estimate recorder: %s (warned once)", reason
            )


def _gather_token_logprobs(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    plan: _GatherPlan,
) -> List[float]:
    if not plan.row_indices:
        return []
    device = logits.device
    row_indices = torch.tensor(plan.row_indices, dtype=torch.long, device=device)
    token_indices = torch.tensor(plan.token_indices, dtype=torch.long, device=device)
    temps = temperatures.to(device=device)[row_indices].clamp_min(1e-5)

    results: List[torch.Tensor] = []
    for start in range(0, row_indices.shape[0], _GATHER_ROW_CHUNK):
        end = start + _GATHER_ROW_CHUNK
        rows = logits[row_indices[start:end]].to(torch.float32)
        rows = rows / temps[start:end, None]
        log_norm = torch.logsumexp(rows, dim=-1)
        token_logits = rows.gather(dim=1, index=token_indices[start:end, None]).squeeze(
            1
        )
        results.append(token_logits - log_norm)
    return torch.cat(results).cpu().tolist()
