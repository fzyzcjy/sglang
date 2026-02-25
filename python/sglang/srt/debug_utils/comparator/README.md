# comparator — Internal Architecture

> **This document describes internal implementation details for developers.**
> It is NOT user-facing documentation.
> For user-facing usage, see the project's main docs.

## Glossary

| Term | Definition |
|------|-----------|
| **dump** | A directory of tensor files produced by `dumper`. Each file contains one tensor value plus metadata (name, step, rank, dims, parallel info, etc.). |
| **step** | A logical time step within a dump. The dumper calls `step()` to advance; each tensor belongs to exactly one step. |
| **tensor group** | All tensor files that share the same match key (typically `name`). In `logical` grouping mode a group spans all ranks and steps; in `raw` mode it is per-rank. |
| **match / MatchResult** | A pairing of one tensor group from the baseline dump with the corresponding group from the target dump, produced by `row_matcher`. |
| **unshard** | Reassembling a tensor that was sharded across parallel ranks (TP, CP, EP) back into a single logical tensor, guided by `dims` annotations. |
| **dims / DimSpec** | A per-tensor annotation string (e.g. `"seq:cp:zigzag, hidden:tp"`) that describes how each dimension is partitioned across parallel axes, including ordering and reduction. |
| **alignment plan** | A plan computed from aux tensors (input_ids, positions, seq_lens, …) that describes how to reorder/slice tokens so that two dumps with different batch compositions become element-wise comparable. |
| **aux tensors** | Auxiliary metadata tensors dumped alongside model tensors (input_ids, positions, seq_lens, req_pool_indices, rids). Used to build the alignment plan. |
| **Pair** | A generic container holding an `x` (baseline) and `y` (target) value. Used throughout to keep the two sides together. |
| **ComparisonRecord / SkipRecord** | Output records: `ComparisonRecord` holds diff metrics for a successfully compared tensor group; `SkipRecord` records why a group was skipped (e.g. load failure). |

## Module Responsibilities

```
entrypoint.py                  Top-level CLI + orchestration
  │                            Parses args, loads metadata, builds alignment plan,
  │                            iterates over matches, emits records.
  │
  ├─ row_matcher.py            Pairs tensor groups between baseline and target
  │
  ├─ tensor_group_comparator.py   Processes one MatchResult end-to-end:
  │   │                           load → unshard → align → compare → record
  │   │
  │   ├─ aligner/unshard/      Computes and executes unshard plans
  │   ├─ aligner/reorder.py    Reorders tensors (e.g. zigzag → natural)
  │   └─ tensor_comparison/    Element-wise comparison, stats, formatting
  │
  ├─ aligner/token_align/      Token-level alignment (cross-step, cross-framework)
  │   ├─ aux_loader.py         Loads and normalizes aux tensors
  │   ├─ indexer.py            Builds per-sequence token indices
  │   ├─ planner.py            Computes alignment plan from indices
  │   └─ executor.py           Executes alignment plan on step→tensor dicts
  │
  ├─ dims.py                   Parses dims annotation strings into DimSpec
  ├─ output_types.py           Pydantic record types (Config, Comparison, Skip, Summary)
  └─ utils.py                  Shared utilities (Pair, StrictBase, shape unification, rel_diff)
```

## Data Flow

```
  baseline dump          target dump
       │                      │
       └──── row_matcher ─────┘
                  │
          list[MatchResult]
                  │
       ┌──── for each match ────┐
       │                        │
       │  tensor_group_comparator.compare_tensor_group()
       │    1. Group rows by step
       │    2. For each step: load files → unshard → single tensor
       │    3. If alignment plan exists: execute_alignment()
       │       else: concat steps
       │    4. compare_tensors() → ComparisonRecord
       │                        │
       └────────────────────────┘
                  │
       ComparisonRecord / SkipRecord  →  print_record()
```
