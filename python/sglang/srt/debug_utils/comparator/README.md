# comparator — Internal Architecture

> **Note**: This document describes internal implementation details for developers.
> It is NOT user-facing documentation.

## Glossary

| Term | Definition |
|------|-----------|
| **Tensor group** | The set of all dump files sharing the same logical tensor name across all ranks and steps. E.g. `hidden_states` with TP=2 and 2 steps produces 4 files, which form one tensor group. |
| **Match / MatchResult** | A pairing of one tensor group from the baseline dump with the corresponding group from the target dump, matched by key columns (typically `name`). Contains `rows_baseline` and `rows_target`. |
| **Step** | A logical time step within a dump, delimited by the dumper's `step()` call. Files within a tensor group are first grouped by step; each step is unsharded independently, then steps are concatenated. |
| **Unshard** | Reassembling a tensor that was sharded across parallel ranks back into a single logical tensor. E.g. with TP=4, four shards are concatenated into one complete tensor. |
| **Reorder** | A permutation applied after unshard, e.g. converting CP zigzag ordering to natural ordering. |
| **DimSpec** | Per-dimension metadata annotation describing how that dimension is partitioned across parallel axes (TP/CP/EP/SP), its ordering (zigzag/natural), and reduction (partial). Written by the dumper in the `dims` metadata field, e.g. `"s:cp:zigzag,h"`. |
| **Alignment plan** | When the two dumps have different batch compositions (different sequence ordering, token counts), aux tensors are used to build a token-level mapping. The plan records `(step, index)` correspondences between the two sides. |
| **Aux tensors** | Auxiliary metadata tensors dumped alongside model tensors: `input_ids`, `positions`, `seq_lens`, `req_pool_indices`, `rids`, etc. Not compared numerically; only used to build the alignment plan. |
| **Grouping mode** | `logical`: unshard across ranks then compare the complete tensor. `raw`: compare per-rank shards individually. |
| **Pair** | Generic container `Pair[T]` holding `.x` (baseline) and `.y` (target). Used throughout to keep the two sides together. |

## Module Responsibilities

```
comparator/
├── entrypoint.py              — CLI entry point + top-level orchestration (arg parsing, alignment plan, match iteration, summary output)
├── tensor_group_comparator.py — End-to-end processing of a single tensor group (load → unshard → align → compare)
├── row_matcher.py             — Match baseline/target metadata rows into MatchResult list by key columns
├── dims.py                    — Parse dims annotation strings into DimSpec lists
├── output_types.py            — Output record types (ConfigRecord, ComparisonRecord, SkipRecord, SummaryRecord) and formatting
├── utils.py                   — Shared utilities (Pair, _StrictBase, shape unification, rel_diff)
│
├── tensor_comparison/         — Pure tensor numerical comparison (no dump file loading or unshard)
│   ├── compare.py             — compare_tensors(): compute diff stats, shape unify, downcast
│   ├── types.py               — TensorComparisonInfo, DiffInfo, TensorStats data types
│   ├── formatter.py           — Comparison result → human-readable text
│   └── printer.py             — Print formatted text
│
└── aligner/                   — Reassemble multi-rank shards into complete tensors + token-level alignment
    ├── unshard/               — Concat/pick operations along parallel axes
    │   ├── planner.py         — Generate UnshardPlan from DimSpec + parallel_info
    │   ├── executor.py        — Execute UnshardPlan (concat shards, verify replicated consistency)
    │   ├── parallel_info.py   — Extract and normalize parallel info from dump metadata
    │   └── types.py           — UnshardPlan, ConcatParams, PickParams, AxisInfo
    ├── reorder.py             — Permutation transforms (e.g. zigzag → natural)
    └── token_align/           — Cross-batch token-level alignment
        ├── aux_loader.py      — Load aux tensors and normalize into framework-agnostic StepAux
        ├── indexer.py         — Build SeqsInfo (per-sequence token position indices) from StepAux
        ├── planner.py         — Compute AlignmentPlan from two-sided SeqsInfo
        ├── executor.py        — Execute AlignmentPlan: extract aligned tensors from step→tensor maps
        └── types.py           — StepAux, TokenAlignGlobalAux, SeqsInfo, AlignmentPlan
```

## Data Flow

```
dump files (baseline + target)
        │
        ▼
   read_meta()  →  df_baseline, df_target        ← entrypoint.py
        │
        ▼
   match_rows()  →  list[MatchResult]             ← row_matcher.py
        │
        ▼                    ┌──────────────────────────────────────────┐
   for each match:           │  tensor_group_comparator.py              │
        │                    │                                          │
        ├─ load files        │  _load_and_unshard_by_step()             │
        ├─ unshard per step  │    └─ _load_and_unshard_files()          │
        ├─ align / concat    │  _compare_tensor()                      │
        ├─ compare           │    ├─ execute_alignment() or concat     │
        │                    │    └─ compare_tensors()                  │
        └─ yield record      │                                          │
                             └──────────────────────────────────────────┘
        │
        ▼
   print + summary                                ← entrypoint.py
```
