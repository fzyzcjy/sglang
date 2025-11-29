import logging
import os
import time
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Dict

import torch

from sglang.srt.managers.io_struct import ProfileReqOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_npu

_is_npu = is_npu()
if _is_npu:
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.ProfilerActivity.CUDA", torch_npu.profiler.ProfilerActivity.NPU],
        ["profiler.ProfilerActivity.CPU", torch_npu.profiler.ProfilerActivity.CPU],
    ]
    torch_npu._apply_patches(patches)

logger = logging.getLogger(__name__)


class ProfileManager:
    def __init__(self, tp_rank: int, cpu_group):
        self.stage_based_trigger = _StageBasedTrigger(
            all_stages=["prefill", "decode"],
            on_start=self._do_start,
            on_stop=self._do_stop,
        )
        self.tp_rank = tp_rank
        self.cpu_group = cpu_group
        self.profiler_kwargs = None
        self.profiler = None

    def step(self, forward_mode: ForwardMode):
        stage = _get_stage_from_forward_mode(forward_mode)
        if stage is None:
            return

        self.stage_based_trigger.step(stage=stage)

    def configure(
            self,
            output_dir: Optional[str],
            start_step: Optional[int],
            num_steps: Optional[int],
            activities: Optional[List[str]],
            with_stack: Optional[bool],
            record_shapes: Optional[bool],
            profile_by_stage: bool,
            profile_id: str,
            merge_profiles: bool,
            profile_prefix: str,
    ):
        # not supported yet
        assert start_step is None
        assert profile_by_stage, "only support profile_by_stage=true now"  # `false` can be easily supported
        assert not merge_profiles

        self.profiler_kwargs = dict(
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            output_dir=output_dir,
            output_prefix=profile_prefix,
            profile_id=profile_id,
        )

        self.stage_based_trigger.configure(
            num_steps=num_steps,
        )

        return ProfileReqOutput(success=True, message="Succeeded")

    def manual_start(self):
        raise NotImplementedError("manually start is only supported yet")

    def manual_stop(self):
        raise NotImplementedError("manually stop is only supported yet")

    def _do_start(self, stage: Optional[str] = None):
        logger.info(
            f"Profiling starts{f' for {stage}' if stage else ''}. "
            f"Traces will be saved to: {self.profiler_kwargs['output_dir']} "
            f"(with profile id: {self.profiler_kwargs['profile_id']})",
        )

        assert self.profiler is None
        self.profiler = _ProfilerBase.create(
            **self.profiler_kwargs,
            tp_rank=self.tp_rank,
            cpu_group=self.cpu_group,
            output_suffix=f"-{stage}" if stage else "",
        )
        self.profiler.start()

    def _do_stop(self):
        logger.info("Stop profiling...")
        self.profiler.stop()
        logger.info(f"Profiling done. Traces are saved to: {self.profiler_kwargs['output_dir']}")
        self.profiler_kwargs = None
        self.profiler = None


def _get_stage_from_forward_mode(forward_mode: ForwardMode):
    if forward_mode.is_prefill():
        return "prefill"
    elif forward_mode.is_decode():
        return "decode"
    elif forward_mode.is_idle():
        return None
    else:
        raise RuntimeError(f"unsupported profile stage: {forward_mode=}")


# ======================================== Stage related ==========================================


class _StageBasedTrigger:
    @dataclass
    class _StateOfStage:
        target_count: int

    def __init__(self, all_stages, on_start: Callable, on_stop: Callable):
        self.all_stages = all_stages
        self.on_start = on_start
        self.on_stop = on_stop

        self.running_stage: Optional[str] = None
        self.state_of_stage: Optional[Dict[str, _StageBasedTrigger._StateOfStage]] = None

    def configure(self, num_steps: int):
        self.running_stage = None
        self.state_of_stage = {stage: _StageBasedTrigger._StateOfStage(target_count=num_steps) for stage in self.all_stages}

    def step(self, stage: str):
        if self.state_of_stage is None:
            return

        TODO


# ======================================== Concrete profilers ==========================================


class _ProfilerBase(ABC):
    @staticmethod
    def create(activities, with_stack, record_shapes, **kwargs):
        inners = []
        if ("CPU" in activities) or ("GPU" in activities):
            inners.append(
                _ProfilerTorch(
                    **kwargs,
                    activities=activities,
                    with_stack=with_stack,
                    record_shapes=record_shapes,
                )
            )
        if "MEM" in activities:
            inners.append(_ProfilerMemory(**kwargs))
        if "CUDA_PROFILER" in activities:
            inners.append(_ProfilerCudart(**kwargs))
        if "RPD" in activities:  # for ROCM
            inners.append(_ProfilerRPD(**kwargs))

        return _ProfilerList(inners)

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _ProfilerList(_ProfilerBase):
    def __init__(self, inners: List[_ProfilerBase]):
        self.inners = inners

    def start(self):
        for inner in self.inners:
            inner.start()

    def stop(self):
        for inner in self.inners:
            inner.stop()


class _ProfilerConcreteBase(_ProfilerBase):
    def __init__(
            self,
            output_dir: str,
            output_prefix: str,
            output_suffix: str,
            profile_id: str,
            tp_rank: int,
            cpu_group,
    ):
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.output_suffix = output_suffix
        self.profile_id = profile_id
        self.tp_rank = tp_rank
        self.cpu_group = cpu_group


class _ProfilerTorch(_ProfilerConcreteBase):
    def __init__(self, with_stack: bool, record_shapes: bool, activities, **kwargs):
        super().__init__(**kwargs)
        self.with_stack = with_stack
        self.record_shapes = record_shapes
        self.activities = activities

    def start(self):
        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [
            activity_map[a] for a in self.activities if a in activity_map
        ]

        self.torch_profiler = torch.profiler.profile(
            activities=torchprof_activities,
            with_stack=self.with_stack if self.with_stack is not None else True,
            record_shapes=(
                self.record_shapes if self.record_shapes is not None else False
            ),
            on_trace_ready=(
                None
                if not _is_npu
                else torch_npu.profiler.tensorboard_trace_handler(self.output_dir)
            ),
        )
        self.torch_profiler.start()

    def stop(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.torch_profiler.stop()
        if not _is_npu:
            # Build filename with only non-zero ranks to maintain backward compatibility
            filename_parts = [self.profile_id, f"TP-{self.tp_rank}"]

            # Only add other ranks if parallelism is enabled (size > 1)
            if getattr(self, "dp_size", 1) > 1:
                filename_parts.append(f"DP-{getattr(self, 'dp_rank', 0)}")
            if getattr(self, "pp_size", 1) > 1:
                filename_parts.append(f"PP-{getattr(self, 'pp_rank', 0)}")
            if getattr(self, "moe_ep_size", 1) > 1:
                filename_parts.append(f"EP-{getattr(self, 'moe_ep_rank', 0)}")

            filename = (
                    (self.output_prefix + "-" if self.output_prefix else "")
                    + "-".join(filename_parts)
                    + self.output_suffix
                    + ".trace.json.gz"
            )

            self.torch_profiler.export_chrome_trace(
                os.path.join(self.output_dir, filename)
            )
        torch.distributed.barrier(self.cpu_group)

        # TODO: migrate `_merge_profile_traces`


class _ProfilerMemory(_ProfilerConcreteBase):
    def start(self):
        torch.cuda.memory._record_memory_history(max_entries=100000)

    def stop(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        memory_profile_path = os.path.join(
            self.output_dir,
            str(time.time())
            + f"-TP-{self.tp_rank}-memory"
            + self.output_suffix
            + ".pickle",
        )
        torch.cuda.memory._dump_snapshot(memory_profile_path)
        torch.cuda.memory._record_memory_history(enabled=None)


class _ProfilerCudart(_ProfilerConcreteBase):
    def start(self):
        torch.cuda.cudart().cudaProfilerStart()

    def stop(self):
        torch.cuda.cudart().cudaProfilerStop()


class _ProfilerRPD(_ProfilerConcreteBase):
    def start(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        from rpdTracerControl import rpdTracerControl

        rpdTracerControl.skipCreate()

        self.rpd_profile_path = os.path.join(
            self.output_dir,
            "rpd-" + str(time.time()) + f"-TP-{self.tp_rank}" + ".trace.json.gz",
        )

        if self.tp_rank == 0:
            import sqlite3

            from rocpd.schema import RocpdSchema

            if os.path.exists("trace.rpd"):
                os.unlink("trace.rpd")
            schema = RocpdSchema()
            connection = sqlite3.connect("trace.rpd")
            schema.writeSchema(connection)
            connection.commit()
            del connection
        torch.distributed.barrier(self.cpu_group)

        self.rpd_profiler = rpdTracerControl()
        self.rpd_profiler.setPythonTrace(True)
        self.rpd_profiler.start()
        self.rpd_profiler.rangePush("", "rpd profile range", "")

    def stop(self):
        self.rpd_profiler.rangePop()
        self.rpd_profiler.stop()
        self.rpd_profiler.flush()

        torch.distributed.barrier(self.cpu_group)
        if self.tp_rank == 0:
            from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace

            rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
