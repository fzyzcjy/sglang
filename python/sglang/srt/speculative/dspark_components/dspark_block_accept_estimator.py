from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import msgspec
import torch

from sglang.srt.kv_canary.runner.future_tensor import DelayedDeviceHostHandler
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


class BlockAcceptEstimateRecorder:
    def __init__(self, *, path: str, gamma: int, device: torch.device) -> None:
        self._gamma = gamma
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w")
        self._device = device
        self._states: dict[str, _RequestState] = {}
        self._steps_since_flush = 0
        self._observed_step_ct = 0
        self._discontinuity_drop_ct = 0
        self._skipped_step_ct = 0
        self._warned_skip_reasons: set[str] = set()

        self._delayed: Optional[DelayedDeviceHostHandler] = None
        if device.type == "cuda":
            self._delayed = DelayedDeviceHostHandler(
                d2h_stream=torch.cuda.Stream(device=device)
            )

        logger.info(
            "DSPARK block accept estimate recorder enabled: path=%s gamma=%d async=%s",
            path,
            gamma,
            self._delayed is not None,
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
        skip_reason = self._skip_reason(
            logits_adjustments_are_noop=logits_adjustments_are_noop,
            corrected_logits=corrected_logits,
        )
        if skip_reason is not None:
            self._skip_step(reason=skip_reason)

        def compute_on_device() -> Optional[dict[str, Any]]:
            if skip_reason is not None:
                return None
            return self._build_device_bundle(
                forward_ct=forward_ct,
                rids=rids,
                draft_tokens=draft_tokens,
                corrected_logits=corrected_logits,
                draft_temperatures=draft_temperatures,
                greedy_mask=greedy_mask,
                target_logits=target_logits,
                target_temperatures=target_temperatures,
                truncated_sampling_mask=truncated_sampling_mask,
                correct_len=correct_len,
                cap_trim_lens=cap_trim_lens,
                bonus=bonus,
                prefix_lens=prefix_lens,
                layout=layout,
            )

        if self._delayed is not None:
            self._delayed.step(
                compute_on_device=compute_on_device,
                postprocess_on_host=self._settle_and_write,
            )
        else:
            bundle = compute_on_device()
            if bundle is not None:
                self._settle_and_write(bundle)

    def flush(self) -> None:
        if self._delayed is not None:
            self._delayed.step(
                compute_on_device=lambda: None,
                postprocess_on_host=self._settle_and_write,
            )
        self._file.flush()
        self._steps_since_flush = 0

    def _build_device_bundle(
        self,
        *,
        forward_ct: int,
        rids: List[str],
        draft_tokens: torch.Tensor,
        corrected_logits: torch.Tensor,
        draft_temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        target_logits: torch.Tensor,
        target_temperatures: torch.Tensor,
        truncated_sampling_mask: Optional[torch.Tensor],
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        bonus: torch.Tensor,
        prefix_lens: torch.Tensor,
        layout: Optional[RaggedVerifyLayout],
    ) -> dict[str, Any]:
        gamma = self._gamma
        rows_per_request = gamma + 1
        bs = len(rids)
        device = target_logits.device
        assert draft_tokens.shape == (bs, gamma)
        assert corrected_logits.shape[0] == bs and corrected_logits.shape[1] == gamma
        assert target_logits.shape[0] == bs * rows_per_request

        if truncated_sampling_mask is not None:
            truncated_mask = truncated_sampling_mask
        else:
            truncated_mask = torch.zeros(bs, dtype=torch.bool, device=device)
        if layout is not None:
            verify_lens = layout.verify_lens
        else:
            verify_lens = torch.full(
                (bs,), rows_per_request, dtype=torch.int32, device=device
            )

        draft_temps_full = (
            draft_temperatures.reshape(bs).to(torch.float32).repeat_interleave(gamma)
        )
        target_temps_full = (
            target_temperatures.reshape(bs)
            .to(torch.float32)
            .repeat_interleave(rows_per_request)
        )

        draft_flat = draft_tokens.reshape(-1)
        q_all = self._gather_logprobs(
            logits=corrected_logits.reshape(bs * gamma, -1),
            row_indices=torch.arange(bs * gamma, device=device),
            token_indices=draft_flat,
            temps=draft_temps_full,
        ).reshape(bs, gamma)

        diag_rows = (
            (torch.arange(bs, device=device) * rows_per_request)[:, None]
            + torch.arange(gamma, device=device)[None, :]
        ).reshape(-1)
        target_diag_logprobs = self._gather_logprobs(
            logits=target_logits,
            row_indices=diag_rows,
            token_indices=draft_flat,
            temps=target_temps_full,
        ).reshape(bs, gamma)

        pending_rows: List[int] = []
        pending_tokens: List[int] = []
        pending_slot_lookup: dict[tuple[int, int, int], int] = {}
        for b in range(bs):
            state = self._states.get(rids[b])
            if state is None or not state.pending or state.expected_seq_len < 0:
                continue
            expected_seq_len = state.expected_seq_len
            for block_idx, block in enumerate(state.pending):
                offset = block.next_offset
                while offset <= gamma:
                    row = block.anchor_pos + offset - expected_seq_len
                    if row < 0 or row >= rows_per_request:
                        break
                    token = block.trimmed_tokens[offset - block.window - 1]
                    pending_slot_lookup[(b, block_idx, offset)] = len(pending_rows)
                    pending_rows.append(b * rows_per_request + row)
                    pending_tokens.append(token)
                    offset += 1

        if pending_rows:
            pending_logprobs = self._gather_logprobs(
                logits=target_logits,
                row_indices=torch.tensor(pending_rows, dtype=torch.long, device=device),
                token_indices=torch.tensor(
                    pending_tokens, dtype=torch.long, device=device
                ),
                temps=target_temps_full,
            )
        else:
            pending_logprobs = torch.zeros(0, dtype=torch.float32, device=device)

        return {
            "forward_ct": int(forward_ct),
            "rids": list(rids),
            "draft_tokens": draft_tokens,
            "correct_len": correct_len,
            "cap_trim_lens": cap_trim_lens,
            "bonus": bonus,
            "prefix_lens": prefix_lens,
            "greedy_mask": greedy_mask,
            "truncated_mask": truncated_mask,
            "verify_lens": verify_lens,
            "q_all": q_all,
            "target_diag_logprobs": target_diag_logprobs,
            "pending_logprobs": pending_logprobs,
            "pending_slot_lookup": pending_slot_lookup,
        }

    def _settle_and_write(self, bundle: dict[str, Any]) -> None:
        gamma = self._gamma
        forward_ct = bundle["forward_ct"]
        rids = bundle["rids"]
        bs = len(rids)

        correct_lens = bundle["correct_len"].tolist()
        cap_trims = bundle["cap_trim_lens"].tolist()
        bonus_tokens = bundle["bonus"].tolist()
        drafts = bundle["draft_tokens"].tolist()
        greedy_rows = bundle["greedy_mask"].tolist()
        truncated_rows = bundle["truncated_mask"].tolist()
        seq_lens = bundle["prefix_lens"].tolist()
        verify_lens = bundle["verify_lens"].tolist()
        q_all = bundle["q_all"].tolist()
        target_diag_logprobs = bundle["target_diag_logprobs"].tolist()
        pending_logprobs = bundle["pending_logprobs"].tolist()
        pending_slot_lookup = bundle["pending_slot_lookup"]

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
            num_old_pending = len(state.pending)
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
                record["trimmed_tokens"] = trimmed_tokens
                record["q_lp"] = q_all[b][window:gamma]

            pending_gathers: List[list] = []
            kept_pending: List[_PendingBlock] = []
            for block_idx, block in enumerate(state.pending):
                diverged = False
                while block.next_offset <= gamma:
                    position = block.anchor_pos + block.next_offset
                    row = position - seq_len
                    assert row >= 0
                    if row > cl:
                        break
                    token = block.trimmed_tokens[block.next_offset - block.window - 1]
                    if block_idx < num_old_pending:
                        p_lp = pending_logprobs[
                            pending_slot_lookup[(b, block_idx, block.next_offset)]
                        ]
                    else:
                        p_lp = target_diag_logprobs[b][row]
                    pending_gathers.append(
                        [
                            block.forward_ct,
                            block.next_offset,
                            p_lp,
                            token,
                            realized[row],
                        ]
                    )
                    block.next_offset += 1
                    if realized[row] != token:
                        diverged = True
                        break
                if not diverged and block.next_offset <= gamma:
                    kept_pending.append(block)
            state.pending = kept_pending

            if pending_gathers:
                record["pg"] = pending_gathers
            self._file.write(json.dumps(record) + "\n")

        self._observed_step_ct += 1
        self._steps_since_flush += 1
        if self._steps_since_flush >= _FLUSH_EVERY_STEPS:
            self._file.flush()
            self._steps_since_flush = 0
        if self._observed_step_ct % _STATE_SWEEP_INTERVAL == 0:
            self._sweep_states(forward_ct=forward_ct)

    def _gather_logprobs(
        self,
        *,
        logits: torch.Tensor,
        row_indices: torch.Tensor,
        token_indices: torch.Tensor,
        temps: torch.Tensor,
    ) -> torch.Tensor:
        if row_indices.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=logits.device)
        per_row_temps = temps[row_indices].clamp_min(1e-5)
        results: List[torch.Tensor] = []
        for start in range(0, row_indices.shape[0], _GATHER_ROW_CHUNK):
            end = start + _GATHER_ROW_CHUNK
            rows = logits[row_indices[start:end]].to(torch.float32)
            rows = rows / per_row_temps[start:end, None]
            log_norm = torch.logsumexp(rows, dim=-1)
            token_logits = rows.gather(
                dim=1, index=token_indices[start:end, None]
            ).squeeze(1)
            results.append(token_logits - log_norm)
        return torch.cat(results)

    def _sweep_states(self, *, forward_ct: int) -> None:
        expired = [
            rid
            for rid, state in self._states.items()
            if forward_ct - state.last_seen_ct > _STATE_EXPIRE_STEPS
        ]
        for rid in expired:
            del self._states[rid]

    def _skip_reason(
        self,
        *,
        logits_adjustments_are_noop: bool,
        corrected_logits: Optional[torch.Tensor],
    ) -> Optional[str]:
        if not logits_adjustments_are_noop:
            return (
                "non-noop logits adjustments (penalizer/logit_bias/grammar) "
                "in batch; cross-step conditioning of the gathered target "
                "probabilities would be state-dependent"
            )
        if corrected_logits is None:
            return "corrected_logits unavailable (folded draft path)"
        return None

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
