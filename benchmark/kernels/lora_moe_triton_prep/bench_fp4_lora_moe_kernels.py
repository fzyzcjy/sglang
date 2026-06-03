"""Self-contained perf-bench + correctness-test for the fp4-LoRA MoE compute
kernels that live inside FP4BlockScaleLoraLauncher::run (EP8 bs64 Kimi decode):

    moe::dev::permute::permuteKernel                       (~7us)
    tensorrt_llm::kernels::nvfp4QuantAndPerTokenScaleKernel (~14us, x2: gate_up + down)
    moe::dev::activation::activationKernel                  (~14-16us)

These have NO standalone python binding, so the overlay module exports three
single-kernel runners (bench_permute / bench_nvfp4_quant / bench_activation);
this script pre-allocates the in/out tensors (shapes from SHAPE_REPORT.md decode
bs64) and calls them directly.

Timing = CUDA-graph-replay over `inner` back-to-back calls, divided by inner
(amortizes the per-replay launch overhead). These kernels are MEMORY-BOUND, so
to avoid measuring warm-L2 (a graph that reuses ONE buffer set keeps the working
set resident in L2 and reports an unrealistically fast number), the calls ROTATE
over `--n-sets` independent buffer sets whose combined footprint exceeds the L2
(GB200 L2 = 135 MB). Each call therefore reads data that the intervening calls
have already evicted -> cold-L2 / true-HBM steady state, matching the e2e where
every kernel invocation reads freshly-written HBM.

Decode bs64 (per-rank EP8): num_tokens=64 top_k=8 hidden=7168 inter=2048
gate_up_n=4096 num_experts=384 local_experts=48 tile=8 max_num_padded_tokens=3200.

Usage (on the GPU pod):
  python3 bench_fp4_lora_moe_kernels.py --mode bench
  python3 bench_fp4_lora_moe_kernels.py --mode bench --n-sets 1   # warm-L2, for comparison
  python3 bench_fp4_lora_moe_kernels.py --mode correctness
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.testing

from sglang.jit_kernel.flashinfer_trtllm_moe.core import (
    get_sgl_trtllm_moe_sm100_raw_module,
)


def swizzled_sf_size(m, n, tile):
    """computeSwizzledLayoutSFSize: round m to (8 if tile<128 else 128), n/16 to 4."""
    m_round = ((m + 7) // 8) * 8 if tile < 128 else ((m + 127) // 128) * 128
    nsf = ((n // 16 + 3) // 4) * 4
    return m_round * nsf


def make_one_set(num_tokens, top_k, hidden, inter, gate_up_n, maxpad, tile, dev):
    e = num_tokens * top_k  # real permuted rows
    # idx_map: expanded slot e -> permuted row e (injective; rows [0,e) used, rest padding).
    idx_map = torch.arange(e, dtype=torch.int32, device=dev)
    total_pad = torch.tensor([maxpad], dtype=torch.int32, device=dev)
    return dict(
        hidden_in=torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev),
        permuted=torch.zeros(maxpad, hidden, dtype=torch.bfloat16, device=dev),
        idx_map=idx_map,
        total_pad=total_pad,
        q1_fp4=torch.empty(maxpad, hidden // 2, dtype=torch.uint8, device=dev),
        q1_sf=torch.empty(
            swizzled_sf_size(maxpad, hidden, tile), dtype=torch.uint8, device=dev
        ),
        q1_ptsf=torch.empty(maxpad, dtype=torch.float32, device=dev),
        gate_up=torch.randn(maxpad, gate_up_n, dtype=torch.bfloat16, device=dev),
        lora_delta=torch.randn(
            num_tokens, top_k, gate_up_n, dtype=torch.bfloat16, device=dev
        )
        * 0.1,
        activated=torch.zeros(maxpad, inter, dtype=torch.bfloat16, device=dev),
        lora_input=torch.zeros(
            num_tokens, top_k, inter, dtype=torch.bfloat16, device=dev
        ),
        q2_fp4=torch.empty(maxpad, inter // 2, dtype=torch.uint8, device=dev),
        q2_sf=torch.empty(
            swizzled_sf_size(maxpad, inter, tile), dtype=torch.uint8, device=dev
        ),
        q2_ptsf=torch.empty(maxpad, dtype=torch.float32, device=dev),
    )


def bench_rotating(call, n_sets, inner=160, warmup=25, rep=100):
    """Per-call ms. `call(i)` runs the kernel on buffer set i; the graph rotates over
    n_sets sets so each call reads L2-cold data (sets together exceed L2)."""
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for j in range(inner):
                call(j % n_sets)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for j in range(inner):
            call(j % n_sets)
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(g.replay, warmup=warmup, rep=rep) / inner
    torch.cuda.synchronize()
    return float(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "correctness"], default="bench")
    ap.add_argument("--num-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--maxpad", type=int, default=3200)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument(
        "--n-sets",
        type=int,
        default=16,
        help="rotate over this many buffer sets so each call is L2-cold (GB200 L2=135MB; "
        "permute working set ~47MB -> 16 sets ~752MB). Use 1 to measure warm-L2.",
    )
    args = ap.parse_args()
    dev = "cuda"
    nt, tk, H, I = args.num_tokens, args.top_k, args.hidden, args.inter
    gun = 2 * I
    mp = args.maxpad
    m = get_sgl_trtllm_moe_sm100_raw_module()
    n_sets = max(1, args.n_sets)
    S = [make_one_set(nt, tk, H, I, gun, mp, args.tile, dev) for _ in range(n_sets)]

    def permute(i):
        t = S[i]
        return m.bench_permute(
            t["hidden_in"], t["idx_map"], t["total_pad"], t["permuted"], nt, tk, H
        )

    def quant1(i):
        t = S[i]
        return m.bench_nvfp4_quant(
            t["permuted"], None, t["q1_fp4"], t["q1_sf"], t["q1_ptsf"], mp, H, args.tile
        )

    def activation(i):
        t = S[i]
        return m.bench_activation(
            t["gate_up"],
            t["lora_delta"],
            t["idx_map"],
            t["total_pad"],
            t["activated"],
            t["lora_input"],
            gun,
            nt,
            tk,
        )

    def quant2(i):
        t = S[i]
        return m.bench_nvfp4_quant(
            t["activated"],
            t["idx_map"],
            t["q2_fp4"],
            t["q2_sf"],
            t["q2_ptsf"],
            nt * tk,
            I,
            args.tile,
        )

    if args.mode == "correctness":
        permute(0)
        torch.cuda.synchronize()
        e = nt * tk
        ref = S[0]["hidden_in"][torch.arange(e, device=dev) // tk]
        perr = int((S[0]["permuted"][:e] != ref).sum().item())
        print(f"{'PASS' if perr == 0 else 'FAIL'} permute gather mismatches={perr}")
        quant1(0)
        activation(0)
        quant2(0)
        torch.cuda.synchronize()
        fin = (
            torch.isfinite(S[0]["q1_ptsf"]).all().item()
            and torch.isfinite(S[0]["activated"][:e].float()).all().item()
            and torch.isfinite(S[0]["q2_ptsf"][:e]).all().item()
        )
        print(
            f"{'PASS' if fin else 'FAIL'} quant/activation finiteness={fin} "
            f"(q1_fp4{tuple(S[0]['q1_fp4'].shape)} activated{tuple(S[0]['activated'].shape)} "
            f"q2_fp4{tuple(S[0]['q2_fp4'].shape)})"
        )
        raise SystemExit(0 if (perr == 0 and fin) else 1)

    l2 = "L2-cold (rotated)" if n_sets > 1 else "WARM-L2 (n_sets=1)"
    us_p = bench_rotating(permute, n_sets) * 1000
    us_q1 = bench_rotating(quant1, n_sets) * 1000
    us_a = bench_rotating(activation, n_sets) * 1000
    us_q2 = bench_rotating(quant2, n_sets) * 1000
    print(
        f"BENCH fp4_lora_moe_kernels decode num_tokens={nt} maxpad={mp} hidden={H} inter={I} "
        f"[{l2}, n_sets={n_sets}]:\n"
        f"  permuteKernel                 = {us_p:7.2f} us\n"
        f"  nvfp4 quant #1 (gate_up, m={mp})  = {us_q1:7.2f} us\n"
        f"  activationKernel              = {us_a:7.2f} us\n"
        f"  nvfp4 quant #2 (down, m={nt*tk})  = {us_q2:7.2f} us"
    )


if __name__ == "__main__":
    main()
