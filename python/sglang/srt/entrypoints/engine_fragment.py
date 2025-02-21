from typing import Optional

from sglang.srt.distributed import ParallelProcessGroups
from sglang.srt.entrypoints.engine_base import EngineBase
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.orchestration.spmd.orchestrator import SpmdOrchestrator
from sglang.srt.server_args import ServerArgs


class EngineFragment(EngineBase):
    def __init__(
        self,
        gpu_id: int,
        nccl_port: Optional[int] = None,
        tp_rank: Optional[int] = None,
        parallel_process_groups: Optional[ParallelProcessGroups] = None,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        tp_size = kwargs.get('tp_size') or parallel_process_groups.tp.device_mesh_device.get_local_size()
        server_args = ServerArgs(*args, log_level=log_level, tp_size=tp_size, **kwargs)
        self._entrypoint = SpmdOrchestrator(
            server_args=server_args,
            nccl_port=nccl_port,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            parallel_process_groups=parallel_process_groups,
        )

    def _generate_impl(self, obj: GenerateReqInput):
        return self._entrypoint.generate(obj)

    def shutdown(self):
        self._entrypoint.shutdown()


def _compute_tp_size():
