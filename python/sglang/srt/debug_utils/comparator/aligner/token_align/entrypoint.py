from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    has_aux_tensors,
    load_and_normalize_aux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.indexer import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.planner import (
    compute_token_align_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    SeqsInfo,
    TokenAlignGlobalAux,
    TokenAlignPlan,
)
from sglang.srt.debug_utils.comparator.utils import Pair


def compute_maybe_token_align_plan(args, df_baseline, df_target):
    if args.grouping == "logical":
        if not (has_aux_tensors(df_baseline) and has_aux_tensors(df_target)):
            print(
                "Warning: aux tensors missing, skipping token alignment",
                file=sys.stderr,
            )
            return None

        return _build_token_align_plan(
            args=args, df_baseline=df_baseline, df_target=df_target
        )

    return None


def _build_token_align_plan(
    *,
    args: argparse.Namespace,
    df_baseline: pl.DataFrame,
    df_target: pl.DataFrame,
) -> TokenAlignPlan:
    """Load aux tensors, build token indices, and compute the alignment plan."""
    dump_paths: Pair[Path] = Pair(
        x=Path(args.baseline_path), y=Path(args.target_path)
    )
    dfs: Pair[pl.DataFrame] = Pair(x=df_baseline, y=df_target)

    global_aux: Pair[TokenAlignGlobalAux] = Pair(
        x=load_and_normalize_aux(dump_path=dump_paths.x, df=dfs.x),
        y=load_and_normalize_aux(dump_path=dump_paths.y, df=dfs.y),
    )

    seqs_info: Pair[SeqsInfo] = Pair(
        x=build_seqs_info(global_aux.x),
        y=build_seqs_info(global_aux.y),
    )

    return compute_token_align_plan(seqs_info_pair=seqs_info)
