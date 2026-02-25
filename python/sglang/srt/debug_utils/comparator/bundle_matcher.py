from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import polars as pl

from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import filter_rows


@dataclass(frozen=True)
class TensorInfo:
    filename: str
    name: str
    step: int


TensorBundleInfo = list[TensorInfo]


def match_bundles(
    *,
    df_baseline: pl.DataFrame,
    df_target: pl.DataFrame,
    skip_keys: set[str],
) -> list[Pair[TensorBundleInfo]]:
    match_key_cols: list[str] = [c for c in df_target.columns if c not in skip_keys]
    unique_keys: pl.DataFrame = df_target.select(match_key_cols).unique(
        maintain_order=True
    )

    results: list[Pair[TensorBundleInfo]] = []
    for key_values in unique_keys.iter_rows(named=True):
        rows_baseline: TensorBundleInfo = _rows_to_tensor_infos(
            filter_rows(df_baseline, conditions=key_values)
        )
        rows_target: TensorBundleInfo = _rows_to_tensor_infos(
            filter_rows(df_target, conditions=key_values)
        )
        results.append(Pair(x=rows_baseline, y=rows_target))

    return results


def _rows_to_tensor_infos(rows: list[dict[str, Any]]) -> list[TensorInfo]:
    tensor_info_fields: set[str] = {f.name for f in dataclasses.fields(TensorInfo)}
    return [
        TensorInfo(**{k: v for k, v in row.items() if k in tensor_info_fields})
        for row in rows
    ]
