# DP-attention padding-mode experiments (adhoc, not for merge)

Throwaway harness for investigating whether `DpPaddingMode.SUM_LEN` makes
attention projections recompute padding tokens under DP attention.

This branch is based on `fdebc938f7`, which is the sglang commit baked into the
`lmsysorg/sglang:latest` image (v0.5.16) that all runs used. The relevant
`DpPaddingMode` / `prepare_mlp_sync_batch` / `communicator` code at this commit
is byte-identical to `upstream/main` at `3c9efaf3e1`.

## Layout

| file | role |
| --- | --- |
| `patch_dpdbg.py` | round 1 instrumentation: padding-mode override + per-step `[DPPAD]` + `[DPATTN]` (rows entering attention in `qwen3_moe`) |
| `patch_gemm_probe.py` | round 2 instrumentation: per-module GEMM `M` probe (`[GEMMPROBE]`), `[PCG]` prefill-CUDA-graph padding log, padding-mode override |
| `run_e2e.sh` | round 1 runner: Qwen3-30B-A3B-FP8, 8 GPUs, one padding variant per invocation |
| `run_probe.sh` | round 2 runner: DeepSeek-V2-Lite, 2 GPUs, `<pad_variant> <prefill_cg_backend>` |
| `bench_dp_collectives.py` | standalone NCCL microbenchmark: MAX_LEN vs SUM_LEN vs gatherv gather+combine |
| `analyze_dp_padding_cost.py` | CPU-only analytic cost table for both modes |
| `summarize_dppad_log.py` | aggregates `[DPPAD]` lines into per-step DP balance statistics |
| `summarize_gemm_probe.py` | aggregates `[GEMMPROBE]` lines into average `M` per module |

## Environment switches added by the patches

| env | values | effect |
| --- | --- | --- |
| `SGLANG_DBG_DP_PAD` | `""` / `heuristic` / `max` | `""` keeps upstream behaviour (forced SUM_LEN for extend); `heuristic` drops the PR #10414 condition so only the communication-cost rule applies; `max` always picks MAX_LEN when `dp_size > 1` |
| `SGLANG_DBG_DP_LOG` | `1` | emit `[DPPAD]`, `[DPATTN]`, `[PCG]` per-step lines |
| `SGLANG_DBG_GEMM_M` | `1` | install the per-module row-count hooks and dump `[GEMMPROBE]` every 200 steps |

## Reproducing

The patches are applied in place to an otherwise clean checkout of
`fdebc938f7`; the two instrumentation commits on this branch record the exact
resulting source state, so `git checkout <commit>` reproduces a run without
re-running the patcher.

```bash
# round 1 (8 GPUs)
python scratch_dp_padding_exp/patch_dpdbg.py <sglang-python-root>/sglang
bash scratch_dp_padding_exp/run_e2e.sh stock       # then heuristic, max

# round 2 (2 GPUs)
python scratch_dp_padding_exp/patch_gemm_probe.py <sglang-python-root>/sglang
bash scratch_dp_padding_exp/run_probe.sh stock disabled   # pad x prefill-cg matrix
```
