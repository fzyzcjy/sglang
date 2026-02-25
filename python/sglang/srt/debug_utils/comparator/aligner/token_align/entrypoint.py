from __future__ import annotations

import sys
from pathlib import Path

from sglang.srt.debug_utils.comparator.aligner.token_align.aux_loader import (
    has_aux_tensors,
    load_and_normalize_aux,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.indexer import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.planner import (
    compute_alignment_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_align.types import (
    SeqsInfo,
    TokenAlignGlobalAux,
    TokenAlignPlan,
)
from sglang.srt.debug_utils.comparator.utils import Pair


def _compute_maybe_alignment_plan(args, df_baseline, df_target):
    if args.grouping == "logical":
        if not (has_aux_tensors(df_baseline) and has_aux_tensors(df_target)):
            print(
                "Warning: aux tensors missing, skipping token alignment",
                file=sys.stderr,
            )
            return None

        return _build_alignment_plan(
            args=args, df_baseline=df_baseline, df_target=df_target
        )

    return None


def _build_alignment_plan(
    *,
    args: argparse.Namespace,
    df_baseline: pl.DataFrame,
    df_target: pl.DataFrame,
) -> TokenAlignPlan:
    """Load aux tensors, build token indices, and compute the alignment plan."""
    baseline_path: Path = Path(args.baseline_path)
    target_path: Path = Path(args.target_path)

    global_aux_baseline: TokenAlignGlobalAux = load_and_normalize_aux(
        dump_path=baseline_path, df=df_baseline
    )
    global_aux_target: TokenAlignGlobalAux = load_and_normalize_aux(
        dump_path=target_path, df=df_target
    )

    seqs_info: Pair[SeqsInfo] = Pair(
        x=build_seqs_info(global_aux_baseline),
        y=build_seqs_info(global_aux_target),
    )

    return compute_alignment_plan(seqs_info_pair=seqs_info)
