# LoRA MoE per-kernel testbeds (EP8 bs64 Kimi-K2.5-NVFP4 decode)

Self-contained **perf-bench + correctness-test** scripts for the individual MoE kernels on the
Kimi-K2.5-NVFP4 LoRA decode path, so each kernel can be benchmarked and checked in isolation
(outside a full 2-node serving run). Built to drive per-kernel optimization.

## Scenario these reproduce

Per-rank shapes of an **EP8, batch-size-64 decode** step (Kimi-K2.5-NVFP4, `tp8 ep8 dp1 nnodes2`,
LoRA on, `modelopt_fp4`): 48 experts/rank, `top_k=8`, `hidden=7168`, `inter=2048`,
`num_tokens=64`, `max_num_padded_tokens=3200`. These were captured from a real e2e run; the exact
per-kernel in/out shape+dtype+stride+device evidence is in the shape report (see *References*).

## Kernels covered (8 kernels, 3 scripts)

| script | kernels | how it calls them |
|---|---|---|
| `bench_triton_gemm_prep.py` | `_fused_virtual_topk_ids_kernel`, `moe_align_block_size_kernel`, `count_and_sort_expert_tokens_kernel` | `_fused_virtual_topk_ids` (triton) + `moe_align_block_size` (the native `sgl_kernel::moe_align_block_size`, which launches both the align and the count_and_sort kernels) |
| `bench_kimi_gate.py` | `kimi_k2_moe_fused_gate` | `sgl_kernel.kimi_k2_moe_fused_gate` (the Kimi routing gate) |
| `bench_fp4_lora_moe_kernels.py` | `permuteKernel`, `nvfp4QuantAndPerTokenScaleKernel` (×2: gate_up + down), `activationKernel` | the standalone runner shim (see below) |

The "triton-gemm prep" three kernels are intentionally in **one** script (they run back-to-back as
the LoRA virtual-experts routing prep).

## The standalone-runner shim (for the fp4-LoRA compute kernels)

`permuteKernel` / `nvfp4QuantAndPerTokenScaleKernel` / `activationKernel` have **no standalone
Python binding** — they only ever run inside `FP4BlockScaleLoraLauncher::run`. To bench them in
isolation, three thin C++ runners are exported from the flashinfer-trtllm-moe overlay module
(`python/sglang/jit_kernel/flashinfer_trtllm_moe/data/csrc/trtllm_fused_moe_kernel_launcher.cu`):

```
bench_permute(hidden_in, idx_map, total_pad, permuted_out, num_tokens, top_k, hidden_size)
bench_nvfp4_quant(in_bf16, idx_map?, out_fp4, out_sf, out_ptsf, m, n, tile)
bench_activation(gate_up, lora_delta, idx_map, total_pad, activated_out, lora_input_out, inner_dim, num_tokens, top_k)
```

Each takes **pre-allocated** in/out tensors and only builds the kernel `Data` struct + launches the
kernel (no device allocation → safe to capture in a CUDA graph for timing). The `Data` setup mirrors
`FP4BlockScaleLoraLauncher::run` exactly. They are accessed from Python via
`get_sgl_trtllm_moe_sm100_raw_module()` (see `bench_fp4_lora_moe_kernels.py`).

## Timing methodology

`bench_ms()` (identical in all 3 scripts) captures **`inner=200` back-to-back kernel calls in one
CUDA graph** and divides the replayed time by `inner`. This amortizes the fixed per-replay launch /
dispatch overhead to ~0 and exposes the true steady-state device time — a single-call
`do_bench(graph.replay)` floors at ~8-10 µs for any tiny op (it measures launch overhead, not the
kernel). This matches graph-on e2e semantics; do **not** replace it with per-iter `cudaSynchronize`
CPU timing (systematically inflates ~µs kernels and dilutes speedups).

## How to run (single GPU)

The scripts import `sglang` / `sgl_kernel` / the flashinfer-trtllm-moe JIT module, so run them on a
machine with the same build (e.g. one GPU of the serving pod, with no server occupying it):

```bash
cd /path/to/sglang
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_triton_gemm_prep.py   --mode correctness
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_triton_gemm_prep.py   --mode bench
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_kimi_gate.py           --mode correctness
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_kimi_gate.py           --mode bench
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_fp4_lora_moe_kernels.py --mode correctness
CUDA_VISIBLE_DEVICES=0 python3 benchmark/kernels/lora_moe_triton_prep/bench_fp4_lora_moe_kernels.py --mode bench
```

Each script defaults to the decode-bs64 production shapes; shape knobs are CLI flags
(`--bs/--num-tokens/--hidden/...`). `bench_fp4_lora_moe_kernels.py` triggers a one-time JIT rebuild
of the overlay module on first import (the shim lives there).

## Measured results (B200, decode bs64)

`--mode bench` device time vs the e2e profile (profile numbers are the per-kernel times observed in
production traces):

| kernel | testbed | e2e profile | correctness |
|---|---|---|---|
| `_fused_virtual_topk_ids` | 1.23 µs | 1.5 | PASS (vs torch ref, 0 mismatch) |
| `moe_align` + `count_and_sort` | 4.75 µs | 2.7 + 4.7 | PASS (output invariants) |
| `kimi_k2_moe_fused_gate` | 4.85 µs | 5 | PASS (expert-set exact; weight err 3e-8) |
| `permuteKernel` | 3.39 µs | 7 | PASS (gather, 0 mismatch) |
| `nvfp4 quant #1` (gate_up, m=3200) | 11.66 µs | 14 | PASS (finite) |
| `activationKernel` | 11.64 µs | 14-16 | PASS (finite) |
| `nvfp4 quant #2` (down, m=512) | 2.82 µs | (part of the 14) | PASS (finite) |

- **Correctness modes** compare against a torch reference where one exists cleanly
  (`_fused_virtual_topk_ids`, `kimi_k2_moe_fused_gate` expert selection + weights, `permute` gather);
  for the fp4 quant / activation (no easy independent fp4 reference) they assert output
  shape + finiteness. All shapes are asserted to match the captured e2e shapes.
- **Speed vs profile:** most match within measurement noise. `permute` and `moe_align+count_and_sort`
  measure faster than the profile sum — expected, because the testbed is isolated warm steady-state
  (graph replay) whereas the e2e profile includes launch gaps and cross-stream contention.

## Key finding (optimization lead)

The nvfp4 quant is invoked twice. **quant #1** (gate_up input) processes
`max_num_padded_tokens = 3200` rows → **11.66 µs**, but only `num_tokens*top_k = 512` of those are
real tokens (the rest are padding). **quant #2** (down input) uses the
`expanded_idx_to_permuted_idx` map to process only the **512** real rows → **2.82 µs**.

→ If quant #1 also used the index map to skip padding rows, it would scale to ~2 µs — matching the
baseline cutlass NVFP4 quantize (`<3 µs`). The ~6.25× padding amplification (3200 vs 512) is the
root cause of the slow quant, and the same amplification applies to `permute` and `activation`
(all three run on the 3200-row padded buffer).

## References

- Per-kernel e2e shape evidence (the shapes these testbeds reproduce): the shape-capture report,
  captured from a real EP8 bs64 decode run with adhoc instrumentation (since reverted).
- Timing template: `benchmark/kernels/lora_moe_expand/bench_expand_add_down.py`.
</content>
