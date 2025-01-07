import multiprocessing as mp

from sglang.srt.managers.scheduler import run_scheduler_process


class EngineFragment:
    def __init__(self):
        self._proc = mp.Process(
            target=run_scheduler_process,
            kwargs=dict(
                server_args=TODO,
                port_args=TODO,
                gpu_id=TODO,
                tp_rank=TODO,
                dp_rank=TODO,
                ready_ipc_name=TODO,
            ),
        )
        self._proc.start()
