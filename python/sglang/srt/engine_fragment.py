import multiprocessing as mp

from sglang.srt.managers.scheduler import run_scheduler_process


class EngineFragment:
    def __init__(
        self,
        tp_rank: int,
        gpu_id: int,
    ):
        self._proc = mp.Process(
            target=run_scheduler_process,
            kwargs=dict(
                server_args=TODO,
                port_args=TODO,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=None,
                ready_ipc_name=TODO,
            ),
        )
        self._proc.start()
