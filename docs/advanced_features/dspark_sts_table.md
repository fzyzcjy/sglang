# DSpark STS Calibration Table

DSpark's confidence head emits a per-draft-position raw accept-rate logit. The
confidence-prefix scheduler turns these into a per-request survival curve
(`cumprod` of the per-position acceptance probabilities) and sizes each verify
window from it. **Sequential Temperature Scaling (STS)** (paper §3.2.1) is an
offline calibration applied to that logit: a per-position temperature `T_k` is
fit on held-out data and applied at inference as `sigmoid(raw_k / T_k)` before the
`cumprod`. By default `T_k = 1` everywhere, i.e. plain `sigmoid` (identity).

## Why calibrate

A confidence head is typically **overconfident**: its per-position survival
probabilities are pushed toward 0/1, so the `cumprod` survival the scheduler
consumes is distorted and the expected-accept-length estimate `tau` it optimizes
is biased. STS fits each `T_k` to minimize the Expected Calibration Error (ECE) of
the survival prediction against the observed prefix-acceptance rate, so the
scheduler sees well-calibrated survival curves and sizes windows more accurately.

Calibration only changes the survival **numbers** fed to the scheduler. It never
changes correctness: losslessness is guaranteed independently by the per-request
accept-cap (`_cap_correct_len`), after which the bonus token is re-read from the
target's true distribution. A missing, stale, or mis-fit table only affects
scheduling quality, never output tokens. With no table the path is byte-identical
to the un-calibrated identity `sigmoid`.

## Offline workflow

The checkpoint does not ship STS temperatures, so they are fit offline in three
steps.

1. **Collect calibration data** from a real DSpark run. Launch a DSpark server with
   the collection env var set to a path stem and serve a representative workload.
   Each step dumps the pre-temperature `(logits, prefix_mask)` of the just-verified
   draft block into `<stem>.<counter>.pt` shards:

   ```bash
   SGLANG_DSPARK_STS_COLLECT_PATH=~/sglang_artifacts/sts/shard \
   python -m sglang.launch_server --model ... <dspark speculative flags>
   ```

   STS must be **off during collection** (omit
   `--speculative-dspark-confidence-sts-path`); launching with the collection env
   and a non-identity table set together fails fast at startup, so the collected
   logits are guaranteed pre-calibration. Collect with greedy
   decoding for the cleanest labels: `prefix_mask[r, k]` is the uncapped
   argmax-match leading-correct-draft prefix (it excludes the bonus token and is
   computed before any verify-window cap), matching what the head predicts.

2. **Fit the temperatures** over the collected shards:

   ```bash
   python -m sglang.benchmark.dspark_sts_fit \
       --data-glob '~/sglang_artifacts/sts/shard.*.pt' \
       --out ~/sglang_artifacts/sts/calib.json \
       --num-bins 15
   ```

   The fitter grid-searches `T_k` per position to minimize the ECE of the `cumprod`
   survival against `prefix_mask[:, k]`, holding earlier positions at their chosen
   best, and writes a `DSparkStsCalibration` JSON (temperatures plus
   dataset/num_samples/ECE metadata). It prints a per-position before/after ECE
   summary.

3. **Launch with the calibration** so the planner loads it onto the head:

   ```bash
   --speculative-dspark-confidence-sts-path ~/sglang_artifacts/sts/calib.json
   ```

   The planner validates that the fitted gamma matches the runtime gamma and
   assigns the `[gamma]` fp32 temperature tensor to the head's `sts_temperatures`
   buffer. A path given when no confidence head is present (static mode / head-less
   checkpoint) is warned about and ignored.

## Notes

- The default (no `--speculative-dspark-confidence-sts-path`, no
  `SGLANG_DSPARK_STS_COLLECT_PATH`) is byte-identical to the un-calibrated path:
  the scalar-`1.0` `sts_temperatures` buffer makes `sigmoid(raw / 1.0)` exact
  identity.
- The fit is conditioned on the collected workload, like the SPS cost table
  (see `docs/advanced_features/dspark_sps_table.md`); collect on a workload near
  the target deployment.
- The collector flushes a shard every fixed number of steps; the final partial
  shard is only written if the run drains cleanly, so collect long enough that a
  dropped remainder is negligible.
