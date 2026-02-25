# comparator — Internal Architecture

> This document describes internal implementation details for developers.
> It is NOT user-facing documentation.

## Key Terms

| Term                     | Definition                                                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tensor group**   | All dump files sharing the same logical tensor name. In `logical` mode this spans all ranks and steps; in `raw` mode it is per-rank.              |
| **Unshard**        | Reassembling a sharded tensor from multiple ranks into one complete tensor, guided by `dims` annotations.                                           |
| **Alignment plan** | A token-level mapping between two dumps with different batch compositions, built from aux tensors (`input_ids`, `positions`, `seq_lens`, etc.). |

## Data Flow

**entrypoint.py** orchestrates the overall pipeline:

```
read_meta(baseline), read_meta(target)
  → match_rows()          pair tensor groups by name
  → for each match:
      compare_tensor_group()    ← tensor_group_comparator.py (see below)
  → print records + summary
```

**tensor_group_comparator.py** handles one matched tensor group:

```
compare_tensor_group(match)
  → for each step:
      load files → unshard across ranks → one tensor per step
  → if alignment_plan: align tokens across steps (reorder to match)
    else:              concat steps
  → compare_tensors(baseline, target) → ComparisonRecord
```
