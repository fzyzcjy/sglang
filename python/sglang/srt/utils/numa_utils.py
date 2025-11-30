from contextlib import contextmanager

from sglang import ServerArgs


@contextmanager
def configure_subprocess(server_args: ServerArgs, gpu_id: int):
    TODO
    try:
        yield
    finally:
        TODO
