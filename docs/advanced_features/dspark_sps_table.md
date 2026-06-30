# DSpark SPS Cost Table

DSpark's hardware-aware confidence-prefix scheduler (paper §3.2.2 / §5.2) sizes the
per-request verify window by maximizing a system-level throughput objective
`Theta = tau * SPS(B)`, where `tau` is the expected number of accepted tokens and
`SPS(B)` is the engine's steps-per-second at batch-token count `B`. `SPS(B)` is a
discrete, cliff-shaped hardware-capacity curve that must be measured once and stored
as a lightweight lookup table the scheduler queries in O(1) at runtime.

## Expected scheduler-on workflow

A profiled table path is **required when the scheduler is enabled** (cap-accept /
compact). Without `--speculative-dspark-sps-table-path` the scheduler **raises at
startup** rather than silently degrading. To deliberately run with a flat constant-SPS
table instead, pass `--speculative-dspark-sps-table-path=const`: `Theta` becomes
proportional to `tau`, the budget degenerates to verify-all, and every request keeps
`verify_len == gamma` (zero throughput gain).

The expected workflow has two steps:

1. **Profile the table offline** against a plain (non-speculative) server. The profiler
   connects to a running server, sweeps decode batch sizes, reads each batch's
   steady-state inter-token latency, and converts it to steps-per-second:

   ```bash
   # Launch a plain (NO speculative flags) server first, then:
   python -m sglang.benchmark.dspark_sps_profiler \
       --model None --base-url http://localhost:30000 \
       --batch-size 1 2 4 8 16 32 64 128 \
       --input-len 512 --output-len 1024 \
       --out ~/sglang_artifacts/dspark_sps_table.json
   ```

   Pick `--input-len` / `--output-len` near the target workload: the table is implicitly
   conditioned on the swept context regime. The profiler self-checks the result for
   monotonicity and cliff preservation.

2. **Pass the table to the DSpark server** so the scheduler loads it:

   ```bash
   --speculative-dspark-sps-table-path ~/sglang_artifacts/dspark_sps_table.json
   ```

   To deliberately run with the flat constant-SPS table instead (verify-all, zero
   throughput gain), pass the literal `const`:

   ```bash
   --speculative-dspark-sps-table-path=const
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

## Notes

- The scheduler's two-steps-prior causal barrier (the K-source reads confidence stashed
  a fixed number of decode steps earlier) and the sort-source split (admission is ranked
  by the current live confidence while the budget K is derived from the lagged
  confidence) follow paper §5.2. These affect scheduling quality only; correctness is
  guaranteed by the accept-cap above.
- The COMPACT (real-N) verify mode that turns the schedule into an actual throughput gain
  is not yet the default and is tracked separately.
</content>
</invoke>
