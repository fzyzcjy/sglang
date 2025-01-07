import multiprocessing as mp
from typing import List, Tuple

import torch
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server.subprocess_launcher import _set_envs_and_config
from sglang.srt.server_args import EngineFragmentArgs
from sglang.srt.utils import create_zmq_ipc_name


class EngineFragment:
    def __init__(
            self,
            tp_rank: int,
            gpu_id: int,
            fragment_args: "EngineFragmentArgs",
    ):
        # Not a good idea to change *current* process's env; but it seems Engine currently does so,
        # so here EngineFragment does the same.
        _set_envs_and_config(fragment_args.server_args)

        pipe_parent, pipe_child = mp.Pipe()
        self._fragment_scheduler_pipe = pipe_parent

        self._proc = mp.Process(
            target=run_scheduler_process,
            kwargs=dict(
                server_args=fragment_args.server_args,
                port_args=fragment_args.port_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=None,
                ready_ipc_name=fragment_args.scheduler_ready_ipc_names[tp_rank],
                fragment_scheduler_pipe=pipe_child,
            ),
        )
        self._proc.start()

    def update_weights_from_tensor(self, named_tensors: List[Tuple[str, torch.Tensor]]):
        """Update weights from tensor directly."""
        obj = UpdateWeightsFromTensorReqInput.init_new(named_tensors, 'fragment')
        self._fragment_scheduler_pipe.send(obj)
        return TODO
