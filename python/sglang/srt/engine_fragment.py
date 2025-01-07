import multiprocessing as mp
from dataclasses import dataclass
from typing import List

from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.utils import create_zmq_ipc_name


class EngineFragment:
    def __init__(
        self,
        tp_rank: int,
        gpu_id: int,
        fragment_args: 'EngineFragmentArgs',
    ):
        self._proc = mp.Process(
            target=run_scheduler_process,
            kwargs=dict(
                server_args=fragment_args.server_args,
                port_args=fragment_args.port_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=None,
                ready_ipc_name=fragment_args.scheduler_ready_ipc_names[tp_rank],
            ),
        )
        self._proc.start()


@dataclass
class EngineFragmentArgs:
    server_args: ServerArgs
    port_args: PortArgs
    scheduler_ready_ipc_names: List[str]

    @staticmethod
    def init_new(log_level: str = "error", *args, **kwargs) -> 'EngineFragmentArgs':
        server_args = ServerArgs(*args, log_level=log_level, **kwargs)
        return EngineFragmentArgs(
            server_args=server_args,
            port_args=PortArgs.init_new(server_args),
            scheduler_ready_ipc_names=[create_zmq_ipc_name() for _ in range(server_args.tp_size)],
        )
