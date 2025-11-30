from contextlib import contextmanager

from sglang import ServerArgs
from sglang.srt.environ import envs


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int):
    if (numa_nodes := server_args.numa_node) is not None and envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = numa_nodes[gpu_id]
        numactl_args = f"--cpunodebind={numa_node} --membind={numa_node}"
        with _configure_subprocess_numactl(numactl_args=numactl_args):
            pass
    else:
        yield


@contextmanager
def _configure_subprocess_numactl(numactl_args: str):
    TODO
    try:
        yield
    finally:
        TODO
