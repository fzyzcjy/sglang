# DSpark SPS Cost Table

DSpark's hardware-aware confidence-prefix scheduler (paper §3.2.2 / §5.2) sizes the
per-request verify window by maximizing a system-level throughput objective
`Theta = tau * SPS(B)`, where `tau` is the expected number of accepted tokens and
`SPS(B)` is the engine's steps-per-second at batch-token count `B`. `SPS(B)` is a
discrete, cliff-shaped hardware-capacity curve that must be measured once and stored
as a lightweight lookup table the scheduler queries in O(1) at runtime.

## Expected scheduler-on workflow

Without `--speculative-dspark-sps-table-path` the scheduler starts **uninitialized**:
a flat constant-SPS table, so `Theta` becomes proportional to `tau`, the budget
degenerates to verify-all, and every request keeps the full verify window (lossless,
but zero throughput gain by itself; a startup WARNING flags it). The uninitialized
start is also the natural cold start for [online profiling](#online-profiling), which
learns the real curve in place.

To start from a measured curve instead, the offline workflow has two steps:

1. **Profile the table offline** against a plain (non-speculative) server. The profiler
   connects to a running server, sweeps decode batch sizes, reads each batch's
   steady-state inter-token latency, and converts it to steps-per-second:

   ```bash
   # Launch a plain (NO speculative flags) server first, then:
   python -m sglang.benchmark.dspark_sps_profiler \
       --base-url http://localhost:30000 \
       --out ~/sglang_artifacts/dspark_sps_table.json
   ```

   Pick `--input-len` / `--output-len` near the target workload: the table is implicitly
   conditioned on the swept context regime. The profiler self-checks the result for
   monotonicity and cliff preservation.

2. **Pass the table to the DSpark server** so the scheduler loads it:

   ```bash
   --speculative-dspark-sps-table-path ~/sglang_artifacts/dspark_sps_table.json
   ```

Offline profiling is a deliberate engineering choice over the paper's literal
"profile once at engine init": it has no startup cost, is reproducible, and lets the
input/output-length regime be aligned to the target workload. The mechanism (profile
once, lightweight table, O(1) cliff-preserving lookup) is identical.

## Table semantics

- The lookup floors `B` down to the nearest captured probe (`bisect_right`), never
  interpolating, so the hardware cliffs in `SPS(B)` are preserved.
- The KV-history approximation in the profiler over-reads history relative to a real
  verify step, which under-estimates steps-per-second and makes the scheduler slightly
  conservative (the verify window opens a touch short, never over-extends) -- a
  direction-safe approximation, not a correctness concern.
- Losslessness does not depend on the table: the verify accept is capped per request by
  `_cap_correct_len` and the bonus token is re-read from the target distribution, so a
  stale or flat table only affects scheduling quality, never output correctness.

## Handling non-monotone probes (outliers)

`SPS(B)` should be non-increasing in batch tokens (more tokens per step never runs
faster), so the profiler self-checks the swept curve and logs a warning when a probe's
steps-per-second rises more than 10% above the previous batch-token's, e.g.:

```text
Non-monotone SPS across probes: batch_tokens=768 SPS=46.858 rose above the previous
probe's SPS=20.222 by >10%; verify the server is at steady state (warmup / output_len
long enough).
```

Such a rise is a measurement artifact, not real hardware behavior. Two flavors:

- **Transient** -- that batch never reached steady state (warmup, or `--output-len` too
  short to average out startup). Re-measuring makes it move.
- **Systematic** -- a peculiarity of that exact batch size (a cuda-graph padding tier, a
  MoE expert-balance sweet spot, or a throughput-window quirk). Re-measuring reproduces
  it. This still must not go into the table verbatim: it is physically implausible as a
  per-step cost and will mislead the scheduler.

**Why it must be fixed, not just noted.** The runtime lookup floors `B` to the nearest
captured probe (`bisect_right`) and never interpolates, so a single inflated probe at
`batch_tokens = X` is returned for the *entire* range `[X, next_probe)`. The scheduler
then under-estimates step cost there and **over-opens** the verify window across that
whole range.

**Fix workflow:**

1. Re-measure the flagged batch size(s) with `--repeats 3` (median rejects a transient
   outlier) and/or a longer `--output-len` (more steady-state steps averaged). If the
   value now sits monotonically between its neighbors, use it directly.
2. If it is *reproducible* but still non-monotone (systematic artifact), patch it to a
   monotone-conservative value -- interpolate between its clean neighbors, or clamp it to
   the previous probe's SPS. Lower-than-real is the safe direction (the scheduler stays
   conservative, per the KV-history note above).
3. Patch the value into the table JSON's `sample_steps_per_sec` (the file the server
   loads), not just your notes -- keep `sample_batch_tokens` aligned and the curve
   non-increasing. Keep the raw (unpatched) table alongside for audit.

Patching is safe: a table only affects scheduling quality, never output correctness (see
above), so correcting an implausible probe can only improve the schedule. When many
probes wobble, prefer re-running the whole sweep with `--repeats 3` over hand-patching.

## Online profiling

Set `SGLANG_DSPARK_ENABLE_SPS_ONLINE_PROFILE=1` to re-measure the table from the live
DSpark server itself and hot-swap the scheduler's copy periodically. Unlike the offline
proxy, this times the *deployed verify path* -- compact packing, cuda-graph bucket
quantization, and shared KV history included -- so the KV-history conservatism above
disappears; the table converges to what verify steps actually cost under the real
workload's context regime.

Mechanics (see `OnlineSpsProfiler` in `dspark_sps_table.py`):

- One sample per pair of consecutive decode steps: host wall-clock between the budget
  planner's calls (the step rate `SPS` actually means, CPU overhead included),
  attributed to the scheduler's own cost coordinate `B = bs + K`. Prefill steps and
  idle gaps break the pairing and are never sampled.
- Samples land in the bin grid of the initial table (its probes), each bin keeping a
  rolling window of recent samples. Every `SGLANG_DSPARK_SPS_ONLINE_REBUILD_INTERVAL`
  decode steps (default 1000) the table is rebuilt: a bin holding at least
  `SGLANG_DSPARK_SPS_ONLINE_MIN_BIN_SAMPLES` samples (default 32) takes the window
  median; an unmeasured bin keeps the initial (offline) value.
- Starting table: pass a profiled `--speculative-dspark-sps-table-path` to refine it
  per bin, or omit it to start uninitialized. In the uninitialized case the offline
  sweep taper becomes the bin grid and unmeasured bins are filled from the nearest
  measured neighbor instead of the flat constant, so the budget keeps exploring B
  ranges it has not observed yet.
- Rank-local, no cross-rank sync: only the `verify_lens` broadcast source rank's table
  affects the schedule; peer ranks' tables drift harmlessly.
- The non-monotone caveat above applies to online medians too, but the rolling window
  re-measures continuously, so an outlier ages out instead of requiring hand-patching.

## Notes

- The scheduler's two-steps-prior causal barrier (the K-source reads confidence stashed
  a fixed number of decode steps earlier) and the sort-source split (admission is ranked
  by the current live confidence while the budget K is derived from the lagged
  confidence) follow paper §5.2. These affect scheduling quality only; correctness is
  guaranteed by the accept-cap above.
- The COMPACT (real-N) verify mode that turns the schedule into an actual throughput gain
  is not yet the default and is tracked separately.
