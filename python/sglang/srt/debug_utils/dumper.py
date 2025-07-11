import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.utils import get_bool_env_var


class _Dumper:
    """Utility to dump tensors, which can be useful when comparison checking models.

    Example usage:
    debug_utils.dumper.dump("layer_start_hidden_states", hidden_states, layer_id=self.layer_id)
    """

    def __init__(self):
        self.enable = get_bool_env_var("SGLANG_DUMPER_ENABLE")
        self._base_dir = Path(os.environ.get("SGLANG_DUMPER_DIR", "/tmp"))
        self._enable_write_file = get_bool_env_var("SGLANG_DUMPER_WRITE_FILE", "1")
        self._partial_name: Optional[str] = None
        self.forward_pass_id = None

    def dump(self, name, value, **kwargs):
        if not self.enable:
            return

        self._ensure_partial_name()

        rank = dist.get_rank()
        full_kwargs = dict(
            forward_pass_id=self.forward_pass_id,
            rank=rank,
            name=name,
            **kwargs,
        )
        full_filename = "___".join(f"{k}={v}" for k, v in full_kwargs.items()) + ".pt"
        path = (
            self._base_dir / f"sglang_dump_{self._partial_name}_{rank}" / full_filename
        )

        sample_value = _get_sample_value(name, value)

        print(
            f"[{rank}, {time.time()}] {path} "
            f"type={type(value)} "
            f"shape={value.shape if isinstance(value, torch.Tensor) else None} "
            f"dtype={value.dtype if isinstance(value, torch.Tensor) else None} "
            f"sample_value={sample_value}"
        )

        if self._enable_write_file:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, str(path))

    def _ensure_partial_name(self):
        if self._partial_name is not None:
            return
        self._partial_name = _get_partial_name()

def _get_partial_name():
    rank = dist.get_rank()
    if rank == 0:
        partial_name = str(time.time())
    else:
        partial_name = None
    

    return TODO

def _get_sample_value(name, value):
    if value is None:
        return None

    if isinstance(value, tuple):
        return [_get_sample_value(name, x) for x in value]

    if not isinstance(value, torch.Tensor):
        return None

    if value.numel() < 200:
        return value

    slices = [
        slice(0, 5) if dim_size > 200 else slice(None) for dim_size in value.shape
    ]
    return value[tuple(slices)]


dumper = _Dumper()
