from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from sglang.srt.debug_utils.dump_loader import filter_rows


@dataclass
class MatchResult:
    match_key: dict[str, Any]
    rows_baseline: list[dict[str, Any]]
    rows_target: list[dict[str, Any]]


def match_rows(
    *,
    df_baseline: pl.DataFrame,
    df_target: pl.DataFrame,
    skip_keys: set[str],
) -> list[MatchResult]:
    """Match rows between baseline and target DataFrames by key columns.

    Key columns are all columns in df_target except those in skip_keys.
    For each unique key combination in df_target, find matching rows in both sides.
    """
    match_key_cols: list[str] = [
        c for c in df_target.columns if c not in skip_keys
    ]
    unique_keys: pl.DataFrame = df_target.select(match_key_cols).unique(
        maintain_order=True
    )

    results: list[MatchResult] = []
    for key_values in unique_keys.iter_rows(named=True):
        rows_target: list[dict[str, Any]] = filter_rows(
            df_target, conditions=key_values
        )
        rows_baseline: list[dict[str, Any]] = filter_rows(
            df_baseline, conditions=key_values
        )
        results.append(
            MatchResult(
                match_key=key_values,
                rows_baseline=rows_baseline,
                rows_target=rows_target,
            )
        )

    return results
