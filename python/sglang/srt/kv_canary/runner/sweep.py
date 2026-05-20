from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.verify import RealKvSource, VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_canary.plan_input import build_plan_input_radix_sweep
from sglang.srt.kv_canary.runner.launch import (
    invoke_plan,
    launch_endpoints_sweep,
)
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce
from sglang.srt.kv_canary.state import CanaryDeviceState

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class SweepOrchestrator:
    """Only walks the radix tree. Per-forward HEAD/TAIL covers running req KV slots every step;
    sweep is purely for the radix-cached-but-not-in-running-batch slot set.

    Runs host-side eager (post-replay), kernels are NOT captured into the cuda graph - sweep
    cadence is host-side state and radix walker output size varies per cycle.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        device: torch.device,
        device_state: CanaryDeviceState,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        endpoints: tuple[CanaryEndpoint, ...],
        req_to_token_pool: "ReqToTokenPool",
        swa_window_size: int,
        sweep_verify_capacity: int,
        pump_and_allreduce: PumpAndAllreduce,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._buffer_groups = buffer_groups
        self._endpoints = endpoints
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size
        self._pump_and_allreduce = pump_and_allreduce
        self._radix_cache: Optional["BasePrefixCache"] = None

        self._verify_plan_sweep_radix = VerifyPlan.allocate(
            verify_capacity=max(1, sweep_verify_capacity), device=device
        )
        self._write_plan_sweep = WritePlan.allocate(write_req_capacity=1, device=device)

        self._last_sweep_step: int = -1
        self._sweep_passes: int = 0

    @property
    def sweep_passes(self) -> int:
        return self._sweep_passes

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._radix_cache = radix_cache

    def maybe_run_sweep(self) -> None:
        if self._config.sweep_interval == 0:
            return
        step_counter = self._pump_and_allreduce.step_counter
        if (
            self._last_sweep_step >= 0
            and step_counter - self._last_sweep_step < self._config.sweep_interval
        ):
            return
        self._last_sweep_step = step_counter

        if self._radix_cache is None:
            return

        violation_log = self._device_state.violation_log
        sweep_capacity = int(self._verify_plan_sweep_radix.verify_slot_indices.shape[0])
        for group in self._buffer_groups:
            window = self._swa_window_size if group.kind is PoolKind.SWA else 0
            radix_input = build_plan_input_radix_sweep(
                radix_cache=self._radix_cache,
                swa_window_size=window,
                full_to_swa_index_mapping=group.swa_index_lut,
            )
            walker_output_size = int(radix_input.extra_verify_slot_indices.shape[0])
            if walker_output_size > sweep_capacity:
                # canary_plan_step's cap_mask would otherwise silently drop entries past
                # sweep_capacity while the verify kernel grid still launches against the larger
                # extras_count, OOB-loading verify_slot_indices. Throw instead.
                raise RuntimeError(
                    f"kv-canary: radix-walker emitted {walker_output_size} sweep verify entries, "
                    f"exceeding pre-allocated sweep_verify_capacity={sweep_capacity}; raise the "
                    f"sweep capacity in CanaryLaunchCapacities.from_args (or "
                    f"_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY)"
                )
            _maybe_perturb_sweep_source(
                group=group,
                slot_indices=radix_input.extra_verify_slot_indices,
            )
            invoke_plan(
                plan_input=radix_input,
                verify_plan=self._verify_plan_sweep_radix,
                write_plan=self._write_plan_sweep,
                group=group,
                req_to_token=self._req_to_token_pool.req_to_token,
                swa_window_size=self._swa_window_size,
            )
            launch_endpoints_sweep(
                endpoints=self._endpoints,
                group=group,
                verify_plan=self._verify_plan_sweep_radix,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
            )

        self._sweep_passes += 1
        logger.info(
            "[canary] sweep succeeded %d times (last_step=%d)",
            self._sweep_passes,
            step_counter,
        )


def _maybe_perturb_sweep_source(
    *,
    group: CanaryBufferGroup,
    slot_indices: torch.Tensor,
) -> None:
    if not envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN.get():
        return
    probability = envs.SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB.get()
    if probability <= 0.0:
        return
    if torch.rand((), device="cpu").item() >= probability:
        return
    if not group.real_kv_sources_k:
        logger.info(
            "kv_canary perturb sweep real_kv: no real-kv sources for group_kind=%s entries=%d",
            group.kind.name,
            int(slot_indices.shape[0]),
        )
        return

    logical_slots = [int(slot) for slot in slot_indices.detach().to("cpu").tolist()]
    inspected_targets = 0
    for source_index, source in enumerate(group.real_kv_sources_k):
        for logical_slot_idx in logical_slots:
            slot_idx = _translate_source_only_slot(
                source=source, logical_slot_idx=logical_slot_idx
            )
            if slot_idx == consts.CANARY_RESERVED_SLOT:
                continue
            row = slot_idx // max(1, source.page_size)
            col = (slot_idx % max(1, source.page_size)) * source.num_bytes_per_token
            if row < 0 or row >= int(source.tensor.shape[0]):
                continue
            if col < 0 or col >= int(source.tensor.shape[1]):
                continue
            inspected_targets += 1

            original_byte = int(source.tensor[row, col].item())
            source.tensor[row, col] = original_byte ^ 0xFF
            logger.info(
                "kv_canary perturb sweep real_kv: group_kind=%s source_idx=%d slot=%d row=%d col=%d "
                "original_byte=0x%02X new_byte=0x%02X",
                group.kind.name,
                source_index,
                logical_slot_idx,
                row,
                col,
                original_byte,
                original_byte ^ 0xFF,
            )
            return
    logger.info(
        "kv_canary perturb sweep real_kv: no valid target for group_kind=%s entries=%d sources=%d inspected=%d",
        group.kind.name,
        len(logical_slots),
        len(group.real_kv_sources_k),
        inspected_targets,
    )


def _translate_source_only_slot(
    *,
    source: RealKvSource,
    logical_slot_idx: int,
) -> int:
    slot = int(logical_slot_idx)
    if source.compress_ratio > 1:
        if slot % source.compress_ratio != source.compress_residue:
            return consts.CANARY_RESERVED_SLOT
        slot = (slot - source.compress_residue) // source.compress_ratio
    if source.slot_mapping is not None:
        mapping = source.slot_mapping
        if slot < 0 or slot >= int(mapping.shape[0]):
            return consts.CANARY_RESERVED_SLOT
        mapped = int(mapping[slot].detach().to("cpu").item())
        if mapped < 0:
            return consts.CANARY_RESERVED_SLOT
        slot = mapped
    return slot
