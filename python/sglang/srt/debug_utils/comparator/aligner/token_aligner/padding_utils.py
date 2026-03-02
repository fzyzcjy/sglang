from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import torch

from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows


def truncate_seq_lens(seq_lens: list[int], *, num_tokens: int) -> list[int]:
    """Truncate seq_lens to cover exactly *num_tokens* tokens (padding at tail).

    Walks seq_lens front-to-back, keeping sequences whole as long as the
    running total stays below *num_tokens*.  The last kept sequence may be
    shortened so the total equals *num_tokens* exactly.  Trailing sequences
    that fall entirely into the padding region are dropped.
    """
    result: list[int] = []
    remaining: int = num_tokens

    for slen in seq_lens:
        if remaining <= 0:
            break
        if slen <= remaining:
            result.append(slen)
            remaining -= slen
        else:
            result.append(remaining)
            remaining = 0

    return result


def load_num_token_non_padded(
    dump_path: Path, df: pl.DataFrame
) -> Optional[dict[int, int]]:
    """Load ``num_token_non_padded`` per step from dump files.

    Returns ``{step: count}`` or ``None`` if the field is absent from the dump.
    """
    rows = filter_rows(df, conditions={"name": "num_token_non_padded"})
    if not rows:
        return None

    result: dict[int, int] = {}
    for row in rows:
        step: int = int(row["step"])
        if step in result:
            continue
        item: ValueWithMeta = ValueWithMeta.load(dump_path / row["filename"])
        value = item.value
        if isinstance(value, int):
            result[step] = value
        elif isinstance(value, torch.Tensor):
            result[step] = int(value.item())
        else:
            result[step] = int(value)

    return result or None


def strip_padding_from_step_tensors(
    tensor_of_step: dict[int, torch.Tensor],
    num_token_non_padded_by_step: dict[int, int],
) -> dict[int, torch.Tensor]:
    """Narrow each step's data tensor to ``[:num_token_non_padded]`` on the token dim (dim 0)."""
    result: dict[int, torch.Tensor] = {}

    for step, tensor in tensor_of_step.items():
        count: Optional[int] = num_token_non_padded_by_step.get(step)
        if count is not None and count < tensor.shape[0]:
            result[step] = tensor[:count]
        else:
            result[step] = tensor

    return result
