from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager, nullcontext
from typing import Callable, ContextManager, Iterator, Optional

import msgspec
import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensors

_NULL_SEGMENT = nullcontext()

COMPONENT_CORE = "core"
COMPONENT_STEP_CPU_TIME = "step_cpu_time"
COMPONENT_STEP_GPU_TIME = "step_gpu_time"
COMPONENT_DRAFT_GPU_TIME = "draft_gpu_time"
COMPONENT_TARGET_VERIFY_GPU_TIME = "target_verify_gpu_time"
COMPONENT_REQS = "reqs"

ALL_COMPONENTS = (
    COMPONENT_CORE,
    COMPONENT_STEP_CPU_TIME,
    COMPONENT_STEP_GPU_TIME,
    COMPONENT_DRAFT_GPU_TIME,
    COMPONENT_TARGET_VERIFY_GPU_TIME,
    COMPONENT_REQS,
)

SEGMENT_STEP = "step"
SEGMENT_DRAFT = "draft"
SEGMENT_TARGET_VERIFY = "target_verify"

INFO_DUMP_MAX_RECORDS = 200_000
INFO_DUMP_MAX_STEP_CPU_SECONDS = 1.0


def resolve_components(raw: tuple[str, ...]) -> set[str]:
    selected = {token.strip() for token in raw if token.strip()}
    if not selected:
        return set()
    if "all" in selected:
        return set(ALL_COMPONENTS)
    unknown = selected - set(ALL_COMPONENTS)
    if unknown:
        raise ValueError(
            f"Unknown SGLANG_DSPARK_DEBUG_DUMP component(s) {sorted(unknown)}; "
            f"valid: {sorted(ALL_COMPONENTS)} or 'all'."
        )
    return selected


class ReqDetail(msgspec.Struct, omit_defaults=True):
    req_pool_index: int
    prefix_len: int
    verify_len: int
    acc_len: int
    correct_drafts: int
    cap_trim: int
    bonus_token: int
    draft_tokens: list[int]
    rid: Optional[str] = None
    confidence: Optional[list[float]] = None
    survival: Optional[list[float]] = None


class DecodeStepRecord(msgspec.Struct, omit_defaults=True):
    forward_ct: int
    bs: int = -1
    mode: str = ""
    budget: Optional[int] = None
    lag_steps: Optional[int] = None
    num_running_reqs: int = -1
    num_verify_tokens: int = -1
    step_cpu_ms: Optional[float] = None
    step_gpu_ms: Optional[float] = None
    draft_gpu_ms: Optional[float] = None
    target_verify_gpu_ms: Optional[float] = None
    reqs: Optional[list[ReqDetail]] = None


class DecodeStepObservation(msgspec.Struct):
    forward_ct: int
    bs: int
    mode: str
    budget: Optional[int]
    lag_steps: Optional[int]
    num_verify_tokens: int
    verify_lens: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor]
    req_pool_indices: torch.Tensor
    prefix_lens: torch.Tensor
    draft_tokens: torch.Tensor
    bonus_tokens: torch.Tensor
    correct_len: torch.Tensor
    cap_trim_lens: torch.Tensor
    commit_lens: torch.Tensor
    rids: Optional[list[str]]


class _PendingStep(msgspec.Struct):
    forward_ct: int
    bs: int
    mode: str
    budget: Optional[int]
    lag_steps: Optional[int]
    num_verify_tokens: int
    step_cpu_ms: Optional[float]
    future: Optional[FutureTensors]
    segment_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]


class DsparkInfoDumper:
    def __init__(
        self,
        *,
        components: set[str],
        gamma: int,
        verify_num_draft_tokens: int,
        tp_rank: int,
        device: torch.device,
        mode_value: str,
        max_records: int = INFO_DUMP_MAX_RECORDS,
        max_step_cpu_seconds: float = INFO_DUMP_MAX_STEP_CPU_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = int(verify_num_draft_tokens)
        self.tp_rank = int(tp_rank)
        self.device = device
        self.mode_value = mode_value
        self._clock = clock
        self._max_step_cpu_seconds = max_step_cpu_seconds

        self._core = COMPONENT_CORE in components
        self._step_cpu = COMPONENT_STEP_CPU_TIME in components
        self._step_gpu = COMPONENT_STEP_GPU_TIME in components
        self._draft_gpu = COMPONENT_DRAFT_GPU_TIME in components
        self._target_verify_gpu = COMPONENT_TARGET_VERIFY_GPU_TIME in components
        self._reqs = COMPONENT_REQS in components
        self._components = sorted(components)
        self.enabled = bool(components) and self.tp_rank == 0

        self._records: deque[DecodeStepRecord] = deque(maxlen=max_records)
        self._pending: Optional[_PendingStep] = None
        self._prev_stamp: Optional[float] = None

        self._d2h_stream: Optional[torch.cuda.Stream] = None
        if self.enabled and self._reqs:
            self._d2h_stream = torch.cuda.Stream(device=device)

        self._current_segments: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = (
            {}
        )
        self._open_segments: dict[str, torch.cuda.Event] = {}

    def begin_step(self) -> None:
        if not self.enabled:
            return
        self._current_segments = {}
        self._open_segments = {}
        if self._step_gpu:
            self._open_segment(SEGMENT_STEP)

    def segment(self, name: str) -> ContextManager[None]:
        if not self.enabled or not self._segment_enabled(name):
            return _NULL_SEGMENT
        return self._active_segment(name)

    @contextmanager
    def _active_segment(self, name: str) -> Iterator[None]:
        self._open_segment(name)
        try:
            yield
        finally:
            self._close_segment(name)

    def observe_decode_step(self, obs: DecodeStepObservation) -> None:
        if not self.enabled:
            return
        if self._step_gpu:
            self._close_segment(SEGMENT_STEP)

        now = self._clock()
        step_cpu_ms = self._step_cpu_ms(now=now)
        self._drain_pending()

        future = self._stage_reqs(obs) if self._reqs else None
        self._pending = _PendingStep(
            forward_ct=int(obs.forward_ct),
            bs=int(obs.bs),
            mode=obs.mode,
            budget=None if obs.budget is None else int(obs.budget),
            lag_steps=None if obs.lag_steps is None else int(obs.lag_steps),
            num_verify_tokens=int(obs.num_verify_tokens),
            step_cpu_ms=step_cpu_ms,
            future=future,
            segment_events=self._current_segments,
        )
        self._current_segments = {}
        self._prev_stamp = now

    def note_non_decode_step(self) -> None:
        if not self.enabled:
            return
        self._drain_pending()
        self._prev_stamp = None
        self._current_segments = {}
        self._open_segments = {}

    def flush(self) -> None:
        if not self.enabled:
            return
        self._drain_pending()

    def dump(self) -> Optional[dict]:
        if not self.enabled:
            return None
        self.flush()
        return {
            "mode": self.mode_value,
            "gamma": self.gamma,
            "verify_num_draft_tokens": self.verify_num_draft_tokens,
            "components": self._components,
            "records": [msgspec.to_builtins(record) for record in self._records],
        }

    def _segment_enabled(self, name: str) -> bool:
        if name == SEGMENT_STEP:
            return self._step_gpu
        if name == SEGMENT_DRAFT:
            return self._draft_gpu
        if name == SEGMENT_TARGET_VERIFY:
            return self._target_verify_gpu
        return False

    def _open_segment(self, name: str) -> None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._open_segments[name] = start

    def _close_segment(self, name: str) -> None:
        start = self._open_segments.pop(name, None)
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._current_segments[name] = (start, end)

    def _stage_reqs(self, obs: DecodeStepObservation) -> Optional[FutureTensors]:
        tensors: dict[str, torch.Tensor] = {
            "req_pool_indices": obs.req_pool_indices,
            "prefix_lens": obs.prefix_lens,
            "draft_tokens": obs.draft_tokens,
            "bonus_tokens": obs.bonus_tokens,
            "correct_len": obs.correct_len,
            "cap_trim_lens": obs.cap_trim_lens,
            "commit_lens": obs.commit_lens,
        }
        if obs.verify_lens is not None:
            tensors["verify_lens"] = obs.verify_lens
        if obs.confidence is not None:
            tensors["confidence"] = obs.confidence
        return FutureTensors.device_to_host(tensors, d2h_stream=self._d2h_stream)

    def _drain_pending(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return

        record = DecodeStepRecord(forward_ct=pending.forward_ct)
        if self._core:
            record.bs = pending.bs
            record.mode = pending.mode
            record.budget = pending.budget
            record.lag_steps = pending.lag_steps
            record.num_running_reqs = pending.bs
            record.num_verify_tokens = pending.num_verify_tokens
        if self._step_cpu:
            record.step_cpu_ms = pending.step_cpu_ms
        if self._step_gpu:
            record.step_gpu_ms = self._segment_ms(pending, SEGMENT_STEP)
        if self._draft_gpu:
            record.draft_gpu_ms = self._segment_ms(pending, SEGMENT_DRAFT)
        if self._target_verify_gpu:
            record.target_verify_gpu_ms = self._segment_ms(
                pending, SEGMENT_TARGET_VERIFY
            )
        if self._reqs and pending.future is not None:
            record.reqs = self._build_reqs(host=pending.future.wait(), bs=pending.bs)
        elif pending.future is not None:
            pending.future.wait()

        self._records.append(record)

    def _step_cpu_ms(self, *, now: float) -> Optional[float]:
        prev = self._prev_stamp
        if prev is None:
            return None
        step_cpu = now - prev
        if not (0.0 < step_cpu <= self._max_step_cpu_seconds):
            return None
        return round(step_cpu * 1000.0, 4)

    def _segment_ms(self, pending: _PendingStep, name: str) -> Optional[float]:
        events = pending.segment_events.get(name)
        if events is None:
            return None
        start, end = events
        end.synchronize()
        return round(start.elapsed_time(end), 4)

    def _build_reqs(self, *, host: dict, bs: int) -> list[ReqDetail]:
        req_ids = host["req_pool_indices"].tolist()
        prefixes = host["prefix_lens"].tolist()
        draft_rows = host["draft_tokens"].tolist()
        bonus = host["bonus_tokens"].tolist()
        correct = host["correct_len"].tolist()
        cap_trim = host["cap_trim_lens"].tolist()
        commit = host["commit_lens"].tolist()
        verify_lens = host["verify_lens"].tolist() if "verify_lens" in host else None
        if "confidence" in host:
            conf_host = host["confidence"].float()
            conf_rows = conf_host.tolist()
            survival_rows = torch.cumprod(conf_host, dim=1).tolist()
        else:
            conf_rows = None
            survival_rows = None

        reqs: list[ReqDetail] = []
        for row in range(bs):
            verify_len = (
                self.verify_num_draft_tokens
                if verify_lens is None
                else int(verify_lens[row])
            )
            reqs.append(
                ReqDetail(
                    req_pool_index=int(req_ids[row]),
                    prefix_len=int(prefixes[row]),
                    verify_len=verify_len,
                    acc_len=int(commit[row]),
                    correct_drafts=int(correct[row]),
                    cap_trim=int(cap_trim[row]),
                    bonus_token=int(bonus[row]),
                    draft_tokens=[int(t) for t in draft_rows[row]],
                    confidence=(
                        None
                        if conf_rows is None
                        else [round(float(p), 4) for p in conf_rows[row]]
                    ),
                    survival=(
                        None
                        if survival_rows is None
                        else [round(float(p), 4) for p in survival_rows[row]]
                    ),
                )
            )
        return reqs
