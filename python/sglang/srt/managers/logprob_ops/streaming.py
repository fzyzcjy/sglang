from __future__ import annotations

from typing import Any, Dict

INCREMENTAL_STREAMING_META_INFO_KEYS = (
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
)


def slice_streaming_output_meta_info(
    meta_info: Dict[Any, Any],
    last_output_offset: int,
) -> None:
    """Align output-side metadata with the current incremental streaming chunk."""
    for key in meta_info.keys() & set(INCREMENTAL_STREAMING_META_INFO_KEYS):
        meta_info[key] = meta_info[key][last_output_offset:]
