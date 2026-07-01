from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.triton_ops.gather_spec_extras import gather_spec_extras
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def decide_needs_cpu_seq_lens(
    server_args: ServerArgs,
    attn_backends: Sequence[AttentionBackend],
) -> bool:
    """Whether FutureMap must publish seq_lens_cpu / sum.

    OR over per-backend needs_cpu_seq_lens; force True under TBO (it reads the
    CPU mirror outside the backend layer to split the batch) or ngram (its
    USE_FULL_MASK verify path reads the host mirror regardless of backend).
    """
    # Local import: keep overlap_utils' module-level deps leaf-only so it stays
    # importable everywhere; spec_info pulls in the spec/schedule_batch graph.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if server_args.enable_two_batch_overlap:
        # FIXME: support TBO without seq lens cpu value
        return True
    algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if algo.is_ngram():
        # ngram's USE_FULL_MASK verify path reads seq_lens_cpu per req to size
        # the tree mask, regardless of the attn backend (e.g. Triton opts out).
        return True
    # Skip unset slots (e.g. draft_extend_attn_backend on some spec configs);
    # missing flag -> True so undeclared backends stay on the legacy path.
    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


def decide_needs_confidence_relay(server_args: ServerArgs) -> bool:
    """Whether FutureMap must publish the per-step DSpark confidence to host.

    Capability gate (mirror decide_needs_cpu_seq_lens): only DSpark in a
    ragged-non-static verify mode reads the relayed confidence to schedule
    per-request verify lengths. Every other algorithm (plain / EAGLE / DFlash /
    ngram / static-DSpark) is a no-op -- the channel allocates nothing and
    publish/resolve early-return on the False flag, so there is zero added
    overhead. The is_dspark / mode decision lives HERE (not in the hot
    publish/resolve path) so shared code never branches on spec identity.
    """
    # Local imports: keep overlap_utils' module-level deps leaf-only.
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if not algo.is_dspark():
        return False
    return read_ragged_verify_mode() is not RaggedVerifyMode.STATIC


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

# Token-buf consume tracking: init to -1, assert non-negative on gather,
# write -1 back. Catches "gather without intermediate stash" bugs. CI enables
# via the existing SGLANG_IS_IN_CI; off in production.
_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()


@torch.compile(dynamic=True, disable=_is_npu)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:
    """Fused: assert all `values >= 0` and scatter -1 into `buf[indices]`.
    Compiled so the reduction + assert + scatter run as one kernel launch."""
    torch._assert_async((values >= 0).all())
    buf[indices] = -1


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:
    """Materialize input_ids at forward entry. Two sources:

    - Prefill: H2D copy from pinned CPU staging (prefill_input_ids_cpu).
    - Decode/spec_v2: gather from FutureMap (last iter's sampled token).
    """
    if batch.prefill_input_ids_cpu is not None:
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(
                    decode_gpu,
                    future_map.output_tokens_buf,
                    batch.mix_running_indices,
                )
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                batch.input_ids, future_map.output_tokens_buf, batch.req_pool_indices
            )

    # Only the overlap path relays spec extras through the future_map; the
    # synchronous (non-overlap) V2 path installs next_draft_input directly.
    if batch.enable_overlap and not batch.spec_algorithm.is_none():
        future_map._resolve_spec_extras(batch)


# Sentinel for a confidence-relay row that was never written (or written for a
# different request). The DSpark host budget planner's per-row identity guard
# masks stale rows (stamped seq_len == this) before they enter the verify budget.
CONFIDENCE_RELAY_UNSET_SEQ_LEN: int = -1

# Sync-free confidence relay (deferred pinned ring). The D2H copy is issued at
# publish time (async, gated on publish_ready), not at resolve time, so
# resolve_confidence_cpu reads an already-landed pinned slot with NO
# cudaStreamSynchronize. RING_LAG is how many publishes back resolve reads: under
# the 1-deep overlap pipeline the immediately-prior forward (lag 1) is still in
# flight at resolve, so the newest GUARANTEED-landed publish is lag 2 (that
# forward's result was already popped/processed by event_loop_overlap). Reading
# lag 2 needs no wait. This RING_LAG replaces the old "wait_event + synchronize"
# lag-1 pull; the DSpark planner's overlap relay_lag_steps MUST equal RING_LAG so
# the host carry supplies the correct remaining lag (carry_steps = total_lag -
# RING_LAG). RING_DEPTH = RING_LAG + 1 keeps the slot being read alive across the
# one intervening publish before it is overwritten.
CONFIDENCE_RELAY_RING_LAG: int = 2
CONFIDENCE_RELAY_RING_DEPTH: int = CONFIDENCE_RELAY_RING_LAG + 1


class ResolvedConfidence(msgspec.Struct):
    """Host snapshot of one batch's relayed DSpark confidence, read from the
    already-landed pinned ring slot (host tensors, indexed to the batch's rows).

    - confidence: [bs, gamma] the lag-RING_LAG per-step confidence; with overlap
      carry_steps == 0 the budget planner consumes it directly as two-steps-prior.
    - seq_lens_stamp: [bs] prefix_len stamped when that confidence was computed
      (same ring slot), the freshness-guard operand.
    - prefix_lens: [bs] post-commit seq_lens from the same ring slot, the guard's
      other operand. Ring-read (not a fresh D2H) to stay sync-free -> growth window
      one step shorter, still inside [1, lag*(gamma+1)]; losslessness never depends
      on the guard.
    """

    confidence: torch.Tensor
    seq_lens_stamp: torch.Tensor
    prefix_lens: torch.Tensor


@dataclass
class RelayPayload:
    """Per-iteration stash payload for the FutureMap bufs. Non-spec fills only
    `bonus_tokens`; which spec extras get relayed is decided by
    `FutureMap.spec_algo`, not by this payload's shape."""

    bonus_tokens: torch.Tensor
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    draft_probs: Optional[torch.Tensor] = None

    @classmethod
    def from_draft_input(cls, draft_input: EagleDraftInput) -> RelayPayload:
        return cls(
            bonus_tokens=draft_input.bonus_tokens,
            topk_p=draft_input.topk_p,
            topk_index=draft_input.topk_index,
            hidden_states=draft_input.hidden_states,
            draft_probs=getattr(draft_input, "draft_probs", None),
        )


class ConfidenceRelay(msgspec.Struct):
    """DSpark confidence channel state + sync-free pinned-ring relay (see the
    CONFIDENCE_RELAY_RING_LAG comment for the lag / no-sync rationale).

    scatter() writes this step's confidence + prefix stamp into the device bufs;
    issue_ring_copy() async-D2Hs the whole buf into the next ring slot (gated on
    publish_ready); resolve() reads the already-landed lag-RING_LAG slot. Buffers
    lazy-init on the first publish (gamma peeked from the confidence shape);
    device/req_pool_size set at construction, else None/0 (inert when
    needs_confidence_relay is False).
    """

    device: torch.device
    req_pool_size: int
    # device scatter targets (accumulate across steps; only this batch's rows update)
    confidence_buf: Optional[torch.Tensor] = None
    stamp_buf: Optional[torch.Tensor] = None  # prefix_len when confidence computed
    # pinned rings (depth CONFIDENCE_RELAY_RING_DEPTH), paired so a slot's three
    # tensors always come from the same forward
    conf_ring: Optional[torch.Tensor] = None
    stamp_ring: Optional[torch.Tensor] = None
    prefix_ring: Optional[torch.Tensor] = None
    # one device.Event per slot; query() at resolve is the fuse that the slot landed
    copy_done: Optional[list] = None
    # monotone publish counter; slot = pos % depth, resolve reads pos - RING_LAG
    ring_pos: int = 0
    initialized: bool = False

    def _lazy_init(self, confidence: torch.Tensor) -> None:
        # Peek gamma from the first published confidence ([bs, gamma]); allocate the
        # device bufs + (CUDA) pinned rings. The stamp buf + every ring stamp slot
        # seed the unset sentinel so a never-written row is detectably stale to the
        # budget guard (also covers ring slots not yet reached at cold start).
        self.initialized = True
        gamma = confidence.shape[-1]
        self.confidence_buf = torch.empty(
            (self.req_pool_size, gamma), dtype=torch.float32, device=self.device
        )
        self.stamp_buf = torch.full(
            (self.req_pool_size,),
            CONFIDENCE_RELAY_UNSET_SEQ_LEN,
            dtype=torch.int64,
            device=self.device,
        )
        if _is_cuda:
            depth = CONFIDENCE_RELAY_RING_DEPTH
            self.conf_ring = torch.empty(
                (depth, self.req_pool_size, gamma),
                dtype=torch.float32,
                pin_memory=True,
            )
            self.stamp_ring = torch.full(
                (depth, self.req_pool_size),
                CONFIDENCE_RELAY_UNSET_SEQ_LEN,
                dtype=torch.int64,
                pin_memory=True,
            )
            self.prefix_ring = torch.empty(
                (depth, self.req_pool_size), dtype=torch.int64, pin_memory=True
            )
            self.copy_done = [
                torch.get_device_module(self.device).Event() for _ in range(depth)
            ]

    def scatter(
        self,
        indices: torch.Tensor,
        confidence: torch.Tensor,
        confidence_seq_lens: torch.Tensor,
    ) -> None:
        # Scatter this step's confidence + its prefix stamp into the device bufs,
        # BEFORE publish records publish_ready (so the gated ring D2H sees them).
        if not self.initialized:
            self._lazy_init(confidence)
        self.confidence_buf[indices] = confidence.to(self.confidence_buf.dtype)
        self.stamp_buf[indices] = confidence_seq_lens.to(self.stamp_buf.dtype)

    def issue_ring_copy(
        self, prefix_buf: torch.Tensor, *, stream, publish_ready
    ) -> None:
        # Async-D2H the whole buf + stamp + current prefix into the next ring slot,
        # gated on the just-recorded publish_ready. Whole-buffer copy matches the
        # accumulate-then-gather semantics; ring_pos advances once per confidence
        # publish (decode-verify step), in lockstep with resolve's read.
        if not self.initialized or stream is None or publish_ready is None:
            return
        slot = self.ring_pos % CONFIDENCE_RELAY_RING_DEPTH
        stream.wait_event(publish_ready)
        with torch.get_device_module(self.device).stream(stream):
            self.conf_ring[slot].copy_(self.confidence_buf, non_blocking=True)
            self.stamp_ring[slot].copy_(self.stamp_buf, non_blocking=True)
            self.prefix_ring[slot].copy_(prefix_buf, non_blocking=True)
            self.copy_done[slot].record()
        self.ring_pos += 1

    def resolve(
        self,
        batch: ScheduleBatch,
        prefix_buf: torch.Tensor,
        *,
        stream,
        publish_ready,
    ) -> Optional[ResolvedConfidence]:
        # Sync-free pull: read the pinned ring slot published RING_LAG steps ago (its
        # async D2H is already landed) -> NO cudaStreamSynchronize. Returns None (cold
        # start / DP idle) so the caller falls back to verify-all.
        if not self.initialized:
            return None
        draft_input = batch.spec_info
        if draft_input is None:
            return None
        fi = draft_input.future_indices
        if fi is None or fi.shape[0] == 0:
            return None

        if stream is None or publish_ready is None:
            # bootstrap / non-CUDA: synchronous pull (rare; same correctness).
            idx = batch.req_pool_indices
            return ResolvedConfidence(
                confidence=self.confidence_buf[idx].cpu(),
                seq_lens_stamp=self.stamp_buf[idx].cpu(),
                prefix_lens=prefix_buf[idx].cpu(),
            )

        # Not enough publishes yet for a lag-RING_LAG slot -> verify-all fallback.
        if self.ring_pos < CONFIDENCE_RELAY_RING_LAG:
            return None
        slot = (self.ring_pos - CONFIDENCE_RELAY_RING_LAG) % CONFIDENCE_RELAY_RING_DEPTH
        # Fuse: the lag-RING_LAG copy is expected landed (that forward was already
        # processed by the overlap loop). If not (deeper pipeline than assumed), fall
        # back to verify-all rather than reading a stale/unlanded slot -- safe, never
        # wrong (losslessness is the accept-cap's job, not the budget's).
        if not self.copy_done[slot].query():
            return None

        idx_cpu = batch.req_pool_indices_cpu
        return ResolvedConfidence(
            confidence=self.conf_ring[slot][idx_cpu],
            seq_lens_stamp=self.stamp_ring[slot][idx_cpu],
            prefix_lens=self.prefix_ring[slot][idx_cpu],
        )


class FutureMap:
    """Always-on pool-indexed relay for cross-iter values. Forward writes via
    publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu.
    """

    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        # Bufs indexed by req_pool_idx; slot 0 mirrors KV padding row so
        # CUDA-graph padded batches (req_pool_idx == 0) are harmless.
        self.device = device
        self.spec_algo = spec_algo
        # Computed by decide_needs_cpu_seq_lens(); see that helper for the
        # full decision (per-backend flag + TBO / piecewise CG overrides).
        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        # Computed by decide_needs_confidence_relay(); gates the DSpark confidence
        # channel below. False for every non-DSpark / static-mode config -> the
        # channel allocates nothing and publish/resolve early-return.
        self.needs_confidence_relay = needs_confidence_relay
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]

        self.output_tokens_buf = (
            torch.full((self.req_pool_size,), -1, dtype=torch.int64, device=self.device)
            if _DEBUG_ASSERT
            else torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
        )
        self.new_seq_lens_buf = torch.empty(
            (self.req_pool_size,), dtype=torch.int64, device=self.device
        )
        # Pinned host copy of new_seq_lens_buf + private stream for fwd-prepare
        # D2H pulls (gated only on publish, off the schedule stream). CUDA-only:
        # recovers occupancy lost to the WAR barrier (also CUDA-only); other
        # platforms have no barrier and use the plain .cpu() bootstrap path.
        if _is_cuda:
            self.new_seq_lens_cpu_pinned = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
            self.fwd_prepare_d2h_stream = torch.get_device_module(self.device).Stream()
        else:
            self.new_seq_lens_cpu_pinned = None
            self.fwd_prepare_d2h_stream = None
        # Lazy-inited on the first non-empty stash (peeks tensor shapes); non-spec's is a no-op.
        self._forward_buf_initialized = False

        self.publish_ready = None  # lazy device.Event(); only spec_v2 needs it

        # DSpark confidence relay: all per-step state + the sync-free ring logic live
        # in ConfidenceRelay; publish() / resolve_confidence_cpu() are thin forwarders.
        # Gated by needs_confidence_relay (False for non-DSpark -> never touched).
        self.confidence_relay = ConfidenceRelay(
            device=self.device, req_pool_size=self.req_pool_size
        )

    def _lazy_init_forward_buf(self, payload: RelayPayload):
        # Local import (see decide_needs_cpu_seq_lens): keep module-level deps leaf.
        from sglang.srt.speculative.spec_utils import spec_need_hidden_states

        self._forward_buf_initialized = True

        # Spec extras are gated by spec_algo, not by the payload's shape, so a
        # non-spec stash allocates no extra bufs (only output_tokens_buf).
        self.need_topk = self.spec_algo.is_some() and self.spec_algo.need_topk()
        self.need_hidden_states = (
            self.spec_algo.is_some()
            and spec_need_hidden_states()
            and payload.hidden_states is not None
        )

        if self.need_topk:
            topk_p0 = payload.topk_p[0]
            topk_index0 = payload.topk_index[0]
            self.topk_p_buf = torch.empty(
                (self.req_pool_size, *topk_p0.shape),
                dtype=topk_p0.dtype,
                device=self.device,
            )
            self.topk_index_buf = torch.empty(
                (self.req_pool_size, *topk_index0.shape),
                dtype=topk_index0.dtype,
                device=self.device,
            )
        if self.need_hidden_states:
            hidden_states0 = payload.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

        self.draft_probs_buf = None
        if payload.draft_probs is not None:
            draft_probs0 = payload.draft_probs[0]
            self.draft_probs_buf = torch.empty(
                (self.req_pool_size, *draft_probs0.shape),
                dtype=draft_probs0.dtype,
                device=self.device,
            )

    def resolve_confidence_cpu(
        self, batch: ScheduleBatch
    ) -> Optional[ResolvedConfidence]:
        # Thin forwarder to the DSpark confidence relay's sync-free ring read. The
        # current prefix (seq_lens relay buf) is passed for the guard's growth check.
        if not self.needs_confidence_relay:
            return None
        return self.confidence_relay.resolve(
            batch,
            self.new_seq_lens_buf,
            stream=self.fwd_prepare_d2h_stream,
            publish_ready=self.publish_ready,
        )

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        if self.spec_algo.is_ngram():
            # FIXME: remove once precomputed draft is supported.
            return
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            return
        indices = draft_input.future_indices
        if indices.shape[0] == 0:
            return
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        if self.need_topk:
            hidden_states_buf = (
                self.hidden_states_buf if self.need_hidden_states else None
            )
            (
                draft_input.topk_p,
                draft_input.topk_index,
                bonus_tokens,
                hidden_states,
            ) = gather_spec_extras(
                indices,
                self.topk_p_buf,
                self.topk_index_buf,
                self.output_tokens_buf,
                hidden_states_buf,
            )
            draft_input.bonus_tokens = bonus_tokens
            if hidden_states is not None:
                draft_input.hidden_states = hidden_states
            if self.draft_probs_buf is not None and draft_input.draft_probs is not None:
                draft_input.draft_probs = self.draft_probs_buf[indices]
        else:
            draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if self.need_hidden_states and not self.need_topk:
            draft_input.hidden_states = self.hidden_states_buf[indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                draft_input.bonus_tokens, self.output_tokens_buf, indices
            )

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        # Lazy pull from new_seq_lens_buf for spec_v2 (accept_lens not known to
        # schedule). The CPU mirror is gated by needs_cpu_seq_lens; backends that
        # opt out take the GPU-only path below. A private D2H stream overlaps the copy.
        draft_input = batch.spec_info
        if draft_input is None:
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        if self.publish_ready is not None:
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        batch.seq_lens = self.new_seq_lens_buf[fi]

        if not self.needs_cpu_seq_lens:
            # GPU gather above is kept (SB.seq_lens must advance each verify);
            # skip the .cpu() D2H. Downstream takes the GPU-only path.
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            return

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            batch.seq_lens_cpu = batch.seq_lens.cpu()  # bootstrap / non-CUDA
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            return

        # Mechanism: don't sync the schedule stream; gate a private stream on the
        # publish event and copy into the static pinned buffer.
        self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
        with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self.fwd_prepare_d2h_stream.synchronize()

        # FIXME: fi == batch.req_pool_indices; unify future_indices and req_pool_indices.
        batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        confidence_seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        indices = future_indices
        if indices.shape[0] == 0:
            return  # DP idle
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        # DSpark confidence channel (gated): scatter this step's confidence + prefix
        # stamp on the forward stream BEFORE the publish_ready record, so the ring D2H
        # (issued after the record, gated on the same event) sees them. Only DSpark
        # callers pass confidence; every other publisher leaves it None -> no-op.
        publish_confidence = self.needs_confidence_relay and confidence is not None
        if publish_confidence:
            self.confidence_relay.scatter(indices, confidence, confidence_seq_lens)
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()
        # Issue the ring D2H AFTER the record (so it is gated on the just-recorded
        # publish_ready); see ConfidenceRelay.issue_ring_copy.
        if publish_confidence:
            self.confidence_relay.issue_ring_copy(
                self.new_seq_lens_buf,
                stream=self.fwd_prepare_d2h_stream,
                publish_ready=self.publish_ready,
            )

    def stash(self, future_indices: torch.Tensor, payload: RelayPayload) -> None:
        if self.spec_algo.is_ngram():
            # FIXME: remove once precomputed draft is supported.
            return
        indices = future_indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            return
        if not self._forward_buf_initialized:
            self._lazy_init_forward_buf(payload)
        self.output_tokens_buf[indices] = payload.bonus_tokens.to(
            self.output_tokens_buf.dtype
        )

        if self.need_topk:
            self.topk_p_buf[indices] = payload.topk_p.to(self.topk_p_buf.dtype)
            self.topk_index_buf[indices] = payload.topk_index.to(
                self.topk_index_buf.dtype
            )
        if self.need_hidden_states:
            self.hidden_states_buf[indices] = payload.hidden_states.to(
                self.hidden_states_buf.dtype
            )
        if self.draft_probs_buf is not None and payload.draft_probs is not None:
            self.draft_probs_buf[indices] = payload.draft_probs
