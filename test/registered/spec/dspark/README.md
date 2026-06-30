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

DSpark (semi-AR block speculative decoding) tests, organized by **what they
prove** and **where they run**. The suite targets four categories; everything in
`test/registered/spec/dspark/` should fall into one of them.

## Categories

### 1. Model implementation parity (GPU)

Draft model forward (backbone + Markov head + confidence head) bit-matches the
DeepSpec source-of-truth modeling. Runs the real `nn.Module` forward on GPU and
compares against a vendored reference (`test/srt/speculative/_dspark_reference/`).
This is the "did we port the math correctly" layer.

| File | What it proves | GPU |
|---|---|---|
| `test_dspark_model_parity.py` | Full draft model forward (qwen3 + gemma4) vs DeepSpec SoT | 1-gpu-small |
| `test_dspark_markov_parity.py` | Markov head `sample_block` (corrected logits + draft_probs) vs SoT | 1-gpu-small |
| `test_dspark_confidence_parity.py` | Confidence head raw logit vs SoT (P0-F tap point) | CPU stub* |

\* `confidence_parity` runs on CPU with random weights (no checkpoint needed);
the full-model GPU parity lives in `test_dspark_model_parity.py`.

### 2. SGLang protocol / unit (CPU)

Pure-Python logic with no model forward and no GPU: length contract, config
parsing, scheduler algorithm, ragged geometry, env parsing, worker dispatch
rules, and the three lossless invariants. Fast, deterministic, the bulk of the
suite by file count.

| File | What it proves |
|---|---|
| `test_dspark_length_contract.py` | gamma/gamma+1 length contract + config parsing (`dspark_utils`) |
| `test_dspark_scheduler.py` | SPS Algorithm 1: budget + topk scheduling (`dspark_scheduler`) |
| `test_dspark_sps_table.py` | SPS step-function lookup + serialization |
| `test_dspark_invariants.py` | Lossless invariants: non-anticipating (property), WAR barrier (structure), capability truth table |
| `test_ragged_verify_layout.py` | `RaggedVerifyLayout` geometry (cumsum / round_up / validation) |
| `test_ragged_verify_grid.py` | `round_up_grid` + `build_graph_num_tokens_grid` |
| `test_ragged_verify_env.py` | `SGLANG_RAGGED_VERIFY` parsing + legacy aliases |
| `test_dspark_cutoff.py` | `_cap_correct_len` + cutoff gating |
| `test_dspark_full_realn.py` | compact verify geometry: `_compact_verify_ids`, `_scatter_compact_to_strided`, accept dispatch, layout grid |
| `test_dspark_ragged_verify.py` | flashinfer ragged verify `qo_indptr` / `kv_indices` + mode gate |
| `test_dspark_draft_backend_selection.py` | draft attention backend selection (V4 sparse vs dense) |
| `test_dspark_distributed_guards.py` | `_handle_dspark` dp/pp fail-fast guards |
| `test_dsv4_confidence.py` | dsv4 confidence head: P0-F tap, base logits, weight remap |
| `test_dsv4_worker_smoke.py` | dsv4 worker capability dispatch + hidden geometry |
| `test_dsv4_ragged_meta.py` | dsv4 backend `_target_verify_graph_key` + resolve gating |
| `test_dsv4_aux_capture_cp_guard.py` | aux-hidden capture x CP fail-loud truth table |

### 3. Model e2e / accuracy (GPU)

Launches a real server with a real checkpoint and asserts end-to-end
losslessness / accuracy. This is the "does it actually work serving" layer.
Depends on checkpoint availability; skips cleanly when checkpoints are missing.

| File | What it proves | GPU |
|---|---|---|
| `test_dspark_lossless.py` | eager greedy token-for-token == baseline + GSM8K floor (qwen3, gemma4) | 1-gpu-large |
| `test_dspark_tp_lossless.py` | TP=2 cross-rank determinism + accuracy floor | 2-gpu-large |
| `test_dspark_flag_matrix.py` | static / cap-accept / compact lossless matrix (compact entry BLOCKED) | 1-gpu-large |
| `test_dspark_rejection_harness.py` | exact-analytic rejection kernel (drives `chain_speculative_sampling_triton`) | 1-gpu-small |

### 4. Attention backend / cuda-graph (GPU UT)

Verifies the attention-metadata construction and cuda-graph capture/replay for
the verify path. Distinct from model parity (#1) because it tests the backend
machinery (metadata builders, graph capture, replay buffers), not the model
math.

> **Note**: the dsv4 GPU verify path (the [P0-V3] spike) is **not yet validated**.
> The only entry here is a skip stub:

| File | What it proves | GPU |
|---|---|---|
| `test_dsv4_worker_parity.py` | dsv4 worker decode vs SoT `forward_spec` + losslessness — **`@skip` BLOCKED on dsv4 GPU enablement** | 1-gpu-small |

## Not run by CI (reference fixtures)

`test/srt/speculative/_dspark_reference/` — a vendored copy of the DeepSpec
qwen3/gemma4 modeling (`markov_head.py`, `draft_ops.py`, `sampling.py`,
`qwen3/`, `gemma4/`). Imported by the model-parity tests (#1) as the numerical
"standard answer". Not collected as tests itself; lives under `test/srt/` (CI
only globs `test/registered/**`).

## Open gaps

- **dsv4 GPU verify (`test_dsv4_worker_parity.py`)** is a skip stub — the
  `[P0-V3]` spike (does `DeepseekV4AttnBackend` TARGET_VERIFY survive
  `num_draft_tokens=gamma+1`?) is unvalidated. This is the single biggest
  unknown in the suite.
- **`compact` (real-N) e2e** in `test_dspark_flag_matrix.py` is `@skip` BLOCKED
  on the ragged-verify routing decision (the backend `graph_num_tokens == total`
  contract mismatch).
- **`schedule_verify_lens_greedy`** is dead production code (only the topk
  variant is wired); covered by tests but never called at runtime.
- **Flat-SPS default makes the scheduler a no-op.** Without
  `SGLANG_DSPARK_SPS_TABLE_PATH`, `_build_sps_cost_table` returns a flat
  constant-SPS table, so the verify-token budget degenerates to
  verify-all-up-to-max and the §5.2 hardware-aware scheduling is inert until a
  profiled table ships (tracked with the SPS profiler/CLI work). The
  `verify_lens >= 1` anchor contract (`DSparkScheduleConfig.min_verify_len`
  default 1 + the topk lower-bound clamp) MUST land before any profiled table is
  supplied: a non-flat table yields a small budget K and would otherwise drive
  `verify_len` to 0, which `RaggedVerifyLayout` rejects.
