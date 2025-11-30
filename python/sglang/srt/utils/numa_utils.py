import multiprocessing
from contextlib import contextmanager

from sglang import ServerArgs
from sglang.srt.environ import envs


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int):
    if (numa_nodes := server_args.numa_node) is not None and envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = numa_nodes[gpu_id]
        numactl_args = f"--cpunodebind={numa_node} --membind={numa_node}"
        executable = _create_numactl_executable(numactl_args=numactl_args)
        with _mp_set_executable(executable=executable):
            pass
    else:
        yield


def _create_numactl_executable(numactl_args: str):
    return TODO


@contextmanager
def _mp_set_executable(executable: str):
    start_method = multiprocessing.get_start_method()
    assert start_method == "spawn", f"{start_method=}"

    old_executable = multiprocessing.spawn.get_executable()
    multiprocessing.spawn.set_executable(executable)
    try:
        yield
    finally:
        multiprocessing.spawn.set_executable(old_executable)
