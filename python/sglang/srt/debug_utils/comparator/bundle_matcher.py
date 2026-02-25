from __future__ import annotations

from typing import Any

import polars as pl

from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import filter_rows

TensorInfoBundle = list[dict[str, Any]]


def match_bundles(
    *,
    df_baseline: pl.DataFrame,
    df_target: pl.DataFrame,
    skip_keys: set[str],
) -> list[Pair[TensorInfoBundle]]:
    match_key_cols: list[str] = [c for c in df_target.columns if c not in skip_keys]
    unique_keys: pl.DataFrame = df_target.select(match_key_cols).unique(
        maintain_order=True
    )

    results: list[Pair[TensorInfoBundle]] = []
    for key_values in unique_keys.iter_rows(named=True):
        rows_baseline: TensorInfoBundle = filter_rows(df_baseline, conditions=key_values)
        rows_target: TensorInfoBundle = filter_rows(df_target, conditions=key_values)
        results.append(Pair(x=rows_baseline, y=rows_target))

    return results
