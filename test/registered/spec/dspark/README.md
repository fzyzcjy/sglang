# DSpark test suite

## Rules

DSpark tests are classified into exactly **four categories** (see below). Every
new test must belong to one of them — they are the only failure modes the suite
guards:

1. **Model implementation parity** (GPU) — draft model forward vs DeepSpec SoT
2. **SGLang protocol / unit** (CPU) — config, scheduler, geometry, env, gates
3. **Model e2e / accuracy** (GPU) — real checkpoint + lossless/accuracy floor
4. **Attention backend / cuda-graph** (GPU UT) — verify metadata + graph capture/replay

**Filing rule (temporary):** while the suite is still being built up, all four
categories live together in this folder (`test/registered/spec/dspark/`). A
final reorg into per-category subfolders (mirroring `attention/unittests/dsv4/`)
is deferred until the suite stabilizes — do **not** create ad-hoc subfolders
before then; just tag each test with its category in the section below.

## Categories (target framework)

The four categories and what each guards:

1. **Model implementation parity** (GPU) — draft model forward (backbone +
   Markov head + confidence head) vs the DeepSpec source-of-truth modeling.
   "Did we port the math correctly?"
2. **SGLang protocol / unit** (CPU) — pure-Python logic: length contract, config
   parsing, scheduler algorithm, ragged geometry, env parsing, worker dispatch
   rules, lossless invariants. No model forward, no GPU.
3. **Model e2e / accuracy** (GPU) — real server + real checkpoint, asserts
   end-to-end losslessness / accuracy (greedy bit-equal, GSM8K floor). Skips
   cleanly when checkpoints are unavailable.
4. **Attention backend / cuda-graph** (GPU UT) — verify-path attention metadata
   construction + cuda-graph capture/replay. Distinct from #1: tests the backend
   machinery, not the model math.

## Existing files are **legacy**

The files currently in this folder were written before this 4-category framework
was adopted, and are organized by **tested object/feature** (e.g.
`test_dspark_cutoff`, `test_dsv4_confidence`, `test_dspark_full_realn`), not by
category. Consequences:

- A single file can span multiple categories (e.g. `test_dspark_invariants.py`
  mixes scheduler-property unit tests with cuda-graph-replay structural tests).
- A single category is scattered across several files (e.g. model-parity logic
  lives in `test_dspark_model_parity`, `test_dspark_markov_parity`,
  `test_dspark_confidence_parity`, *and* `test_dsv4_confidence`).

**Do not** rely on the file name to infer a test's category, and **do not**
attempt a file-level reorg yet — it would break logical cohesion. The per-file
classification below is a rough guide only, to be reconciled into per-category
subfolders once the suite stabilizes (mirroring `attention/unittests/dsv4/`).

### Rough per-file mapping (legacy — not authoritative)

| File | Rough category |
|---|---|
| `test_dspark_model_parity.py` | 1 parity |
| `test_dspark_markov_parity.py` | 1 parity |
| `test_dspark_confidence_parity.py` | 1 parity |
| `test_dsv4_confidence.py` | 1 parity (inline reference, not DeepSpec) |
| `test_dspark_length_contract.py` | 2 protocol/unit |
| `test_dspark_scheduler.py` | 2 protocol/unit |
| `test_dspark_sps_table.py` | 2 protocol/unit |
| `test_ragged_verify_layout.py` | 2 protocol/unit |
| `test_ragged_verify_grid.py` | 2 protocol/unit |
| `test_ragged_verify_env.py` | 2 protocol/unit |
| `test_dspark_cutoff.py` | 2 protocol/unit |
| `test_dspark_full_realn.py` | 2 protocol/unit |
| `test_dspark_ragged_verify.py` | 2 protocol/unit |
| `test_dspark_draft_backend_selection.py` | 2 protocol/unit |
| `test_dspark_distributed_guards.py` | 2 protocol/unit |
| `test_dsv4_worker_smoke.py` | 2 protocol/unit |
| `test_dspark_invariants.py` | 2 + 4 (spans both) |
| `test_dspark_lossless.py` | 3 e2e |
| `test_dspark_tp_lossless.py` | 3 e2e |
| `test_dspark_flag_matrix.py` | 3 e2e |
| `test_dspark_rejection_harness.py` | 4 backend/kernel UT (no server) |
| `test_dsv4_ragged_meta.py` | 4 backend metadata + graph key |
| `test_dsv4_aux_capture_cp_guard.py` | 4 backend capture guard |
| `test_dsv4_worker_parity.py` | 4 backend (skip stub, BLOCKED) |

## Not run by CI (reference fixtures)

`test/srt/speculative/_dspark_reference/` — a vendored copy of the DeepSpec
qwen3/gemma4 modeling (`markov_head.py`, `draft_ops.py`, `sampling.py`,
`qwen3/`, `gemma4/`). Imported by the parity tests (category 1) as the numerical
"standard answer". Not collected as tests itself; lives under `test/srt/` (CI
only globs `test/registered/**`).

## Open gaps

- **Category 4 (attention backend/graph) is the biggest gap** — the dsv4 GPU
  verify path is unvalidated. `test_dsv4_worker_parity.py` is a `@skip` stub
  (the `[P0-V3]` spike: does `DeepseekV4AttnBackend` TARGET_VERIFY survive
  `num_draft_tokens=gamma+1`?). Compare with `attention/unittests/dsv4/`, which
  has a full coverage matrix for the non-spec path.
- **`compact` (real-N) e2e** in `test_dspark_flag_matrix.py` is `@skip` BLOCKED
  on the ragged-verify routing decision (the backend `graph_num_tokens == total`
  contract mismatch).
- **`schedule_verify_lens_greedy`** is dead production code (only the topk
  variant is wired); covered by tests but never called at runtime.
