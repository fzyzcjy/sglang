# comparator — Internal Architecture

> This document describes internal implementation details for developers. It is NOT user-facing documentation.

## Key Terms

| Term                     | Definition                                                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tensor bundle**  | All dump files sharing the same logical tensor name. In `logical` mode this spans all ranks and steps; in `raw` mode it is per-rank.              |
| **Unshard**        | Reassembling a sharded tensor from multiple ranks into one complete tensor, guided by `dims` annotations.                                           |
| **Token align plan** | A token-level mapping between two dumps with different batch compositions, built from aux tensors (`input_ids`, `positions`, `seq_lens`, etc.). |

## Data Flow

```
dump files (baseline + target)
        │
        ▼
   read_meta()  →  df_baseline, df_target
        │
        ▼
   match_bundles()  →  list[Pair[TensorBundle]]
        │
        ▼
   for each bundle pair:
        │
        │  ┌───────────────────────────────────────┐
        └─▶│  compare_bundles()                      │
           │                                        │
           │  1. load files & unshard (per step)    │
           │  2. align tokens  ─or─  concat steps   │
           │  3. compare_tensors()                  │
           │          │                             │
           │          ▼                             │
           │  ComparisonRecord / SkipRecord         │
           └──────────────┬────────────────────────┘
                          │
                          ▼
                  print + summary
```
