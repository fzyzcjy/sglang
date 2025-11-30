from contextlib import contextmanager

from sglang import ServerArgs
from sglang.srt.environ import envs


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int):
    if (numa_node := server_args.numa_node) is not None and envs.SGLANG_NUMA_BIND_V2.get():
        with _configure_subprocess_impl(numa_node=numa_node[gpu_id]):
            pass
    else:
        yield


@contextmanager
def _configure_subprocess_impl(numa_node: int):
    TODO
    try:
        yield
    finally:
        TODO
