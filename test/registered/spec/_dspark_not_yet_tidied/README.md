# DSpark test suite

## Rules

DSpark tests are classified into exactly **four categories** (see below). Every
new test must belong to one of them — they are the only failure modes the suite
guards:

1. **Model implementation parity** (GPU) — draft model forward vs DeepSpec SoT
2. **SGLang protocol / unit** (CPU) — config, scheduler, geometry, env, gates
3. **Model e2e / accuracy** (GPU) — real checkpoint + lossless/accuracy floor
4. **Attention backend / cuda-graph** (GPU UT) — verify metadata + graph capture/replay

**Filing rule:** the category-1 model-parity tests (production draft forward vs
the DeepSpec / DeepSeek-V4-Flash source-of-truth, both dsv4 and dense) live in
the `comparison/` subfolder, which holds exactly those SoT-comparison tests. The
other three categories live together in this folder
(`test/registered/spec/dspark/`); a further reorg into per-category subfolders
(mirroring `attention/unittests/dsv4/`) is deferred until the suite stabilizes —
do **not** create other ad-hoc subfolders before then; just tag each test with
its category in the section below.

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
| `test_deepseek_v4_worker_parity.py` | 1 parity (un-skipped; delegates to T1 harness) |
| `test_deepseek_v4_block_forward_sot_parity.py` | 1 parity (T1 guardrail + negative causal test) |
| `test_dsv4_draft_probs_unit.py` | 2 protocol/unit (T2 draft_probs losslessness) |
| `test_deepseek_v4_dynamic_batch.py` | 1 parity (T3 per-row independence) |
| `test_deepseek_v4_tp_parity.py` | 1 parity (T4 TP=2 vs TP=1) |
| `test_dsv4_injection.py` | 2 + 1 (T5 translate unit + GPU round-trip) |
| `test_deepseek_v4_component_parity.py` | 1 parity (granularity-A: kv-proj / projection / hc_head / markov / confidence vs SoT) |
| `test_dense_block_forward_parity.py` | 1 + 4 (dense granularity-B: draft forward through MHA backend vs SoT whole-block) |

## Not run by CI (reference fixtures)

`test/manual/_dspark_reference/` — a vendored copy of the DeepSpec
qwen3/gemma4 modeling (`markov_head.py`, `draft_ops.py`, `sampling.py`,
`qwen3/`, `gemma4/`) plus the DSpark V4 source-of-truth oracle: the
`deepseek_v4/` package (`sot_attention.py` = the pure-torch non-causal sparse
attention oracle, `modeling.py` = the fuller DSpark draft modeling —
`RefTransformer.forward_spec`, the DSpark heads, and the mHC math — both vendored
from the DeepSeek-V4-Flash-DSpark reference model.py/kernel.py) and the T1/T3/T4
production block-forward fixture (`deepseek_v4/sglang_block_forward_harness.py`). Imported by the
comparison parity tests (category 1) as the numerical "standard answer" /
production driver. Not collected as tests itself; lives under `test/manual/` (CI
only requires every file under `test/registered/**` to be a registered test).

## Open gaps

- **dsv4 block-forward parity is now guarded (T1).**
  `test_deepseek_v4_block_forward_sot_parity.py` drives the production
  `DeepseekV4AttnBackend` non-causal full-block builder vs the external SoT oracle
  and includes a negative causal-flip test; `test_deepseek_v4_worker_parity.py` is
  un-skipped and delegates to the same harness. Both skip cleanly until the
  model+backend agents land the contract (`DSparkV4DraftOutput` +
  `get_dspark_swa_page_indices`), then run on GPU.
- **`compact` (real-N) e2e** in `test_dspark_flag_matrix.py` is `@skip` BLOCKED
  on the ragged-verify routing decision (the backend `graph_num_tokens == total`
  contract mismatch).
- **Flat-SPS default makes the scheduler a no-op.** Without
  `SGLANG_DSPARK_SPS_TABLE_PATH`, `_build_sps_cost_table` returns a flat
  constant-SPS table, so the verify-token budget degenerates to
  verify-all-up-to-max and the §5.2 hardware-aware scheduling is inert until a
  profiled table ships (tracked with the SPS profiler/CLI work). The
  `verify_lens >= 1` anchor contract (`DSparkScheduleConfig.min_verify_len`
  default 1 + the topk lower-bound clamp) MUST land before any profiled table is
  supplied: a non-flat table yields a small budget K and would otherwise drive
  `verify_len` to 0, which `RaggedVerifyLayout` rejects.
