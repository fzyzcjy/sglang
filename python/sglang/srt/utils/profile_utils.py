import logging
import os
import time
from abc import ABC
from typing import List

import torch

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
    def __init__(self):
        TODO

    def step(self):
        TODO

    def start(self):
        stage_str = f" for {stage.name}" if stage else ""
        logger.info(
            f"Profiling starts{stage_str}. Traces will be saved to: {self.output_dir} (with profile id: {self.profile_id})",
        )

        TODO

        return ProfileReqOutput(success=True, message="Succeeded")

    def stop(self):
        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_suffix = f"-{stage.name}" if stage else ""
        logger.info("Stop profiling" + output_suffix + "...")

        TODO

        logger.info(
            "Profiling done. Traces are saved to: %s%s",
            self.output_dir,
            merge_message,
        )
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None

        return ProfileReqOutput(success=True, message=f"Succeeded.{merge_message}")


class _StageBasedDispatcher:
    def __init__(self):
        TODO

    def step(self, stage: str):
        TODO


class _ProfilerBase(ABC):
    @staticmethod
    def create(activities, with_stack, record_shapes, **kwargs):
        inners = []
        if ("CPU" in activities) or ("GPU" in activities):
            inners.append(
                _ProfilerTorch(**kwargs, activities=activities, with_stack=with_stack, record_shapes=record_shapes))
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
    def __init__(self, output_dir: str, output_suffix: str, profile_id: str, tp_rank: int, cpu_group):
        self.output_dir = output_dir
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
            record_shapes=self.record_shapes if self.record_shapes is not None else False,
            on_trace_ready=(
                None
                if not _is_npu
                else torch_npu.profiler.tensorboard_trace_handler(
                    self.output_dir
                )
            ),
        )
        self.torch_profiler.start()

    def stop(self):
        if profile_prefix:
            stage_prefix = profile_prefix + "-"
        else:
            stage_prefix = ""

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
                    stage_prefix
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
