"""Self-contained perf-bench + correctness-test for the fp4-LoRA MoE compute
kernels that live inside FP4BlockScaleLoraLauncher::run (EP8 bs64 Kimi decode):

    moe::dev::permute::permuteKernel                       (~7us)
    tensorrt_llm::kernels::nvfp4QuantAndPerTokenScaleKernel (~14us, x2: gate_up + down)
    moe::dev::activation::activationKernel                  (~14-16us)

These have NO standalone python binding, so the SHAPECAP branch exports three
adhoc single-kernel runners (shapecap_permute / shapecap_nvfp4_quant /
shapecap_activation) from the flashinfer_trtllm_moe overlay module; this script
pre-allocates the in/out tensors (shapes from SHAPE_REPORT.md decode bs64) and
calls them directly. Timing = CUDA-graph-replay inner=200 amortized.

Decode bs64 (per-rank EP8): num_tokens=64 top_k=8 hidden=7168 inter=2048
gate_up_n=4096 num_experts=384 local_experts=48 tile=8 max_num_padded_tokens=3200.

Usage (on the GPU pod):
  python3 bench_fp4_lora_moe_kernels.py --mode bench
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


def make_tensors(num_tokens, top_k, hidden, inter, gate_up_n, maxpad, dev):
    torch.manual_seed(0)
    e = num_tokens * top_k  # real permuted rows
    # idx_map: expanded slot e -> permuted row e (injective; rows [0,e) used, rest padding).
    idx_map = torch.arange(e, dtype=torch.int32, device=dev)
    total_pad = torch.tensor([maxpad], dtype=torch.int32, device=dev)
    t = dict(
        # permute
        hidden_in=torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev),
        permuted=torch.zeros(maxpad, hidden, dtype=torch.bfloat16, device=dev),
        idx_map=idx_map,
        total_pad=total_pad,
        # quant1 (gate_up input): m=maxpad, n=hidden
        q1_fp4=torch.empty(maxpad, hidden // 2, dtype=torch.uint8, device=dev),
        q1_sf=torch.empty(
            swizzled_sf_size(maxpad, hidden, 8), dtype=torch.uint8, device=dev
        ),
        q1_ptsf=torch.empty(maxpad, dtype=torch.float32, device=dev),
        # activation: gate_up [maxpad, gate_up_n] -> activated [maxpad, inter]
        gate_up=torch.randn(maxpad, gate_up_n, dtype=torch.bfloat16, device=dev),
        lora_delta=torch.randn(
            num_tokens, top_k, gate_up_n, dtype=torch.bfloat16, device=dev
        )
        * 0.1,
        activated=torch.zeros(maxpad, inter, dtype=torch.bfloat16, device=dev),
        lora_input=torch.zeros(
            num_tokens, top_k, inter, dtype=torch.bfloat16, device=dev
        ),
        # quant2 (down input): m=num_tokens*top_k, n=inter
        q2_fp4=torch.empty(maxpad, inter // 2, dtype=torch.uint8, device=dev),
        q2_sf=torch.empty(
            swizzled_sf_size(maxpad, inter, 8), dtype=torch.uint8, device=dev
        ),
        q2_ptsf=torch.empty(maxpad, dtype=torch.float32, device=dev),
    )
    return t


def bench_ms(fn, warmup=25, rep=100, inner=200):
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(inner):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
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
    args = ap.parse_args()
    dev = "cuda"
    nt, tk, H, I = args.num_tokens, args.top_k, args.hidden, args.inter
    gun = 2 * I
    mp = args.maxpad
    m = get_sgl_trtllm_moe_sm100_raw_module()
    t = make_tensors(nt, tk, H, I, gun, mp, dev)

    permute = lambda: m.bench_permute(
        t["hidden_in"], t["idx_map"], t["total_pad"], t["permuted"], nt, tk, H
    )
    quant1 = lambda: m.bench_nvfp4_quant(
        t["permuted"], None, t["q1_fp4"], t["q1_sf"], t["q1_ptsf"], mp, H, args.tile
    )
    activation = lambda: m.bench_activation(
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
    quant2 = lambda: m.bench_nvfp4_quant(
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
        # permute: permuted[idx_map[e]] == hidden_in[e // top_k] for e in [0, nt*tk).
        permute()
        torch.cuda.synchronize()
        e = nt * tk
        ref = t["hidden_in"][torch.arange(e, device=dev) // tk]
        got = t["permuted"][:e]
        perr = int((got != ref).sum().item())
        print(f"{'PASS' if perr == 0 else 'FAIL'} permute gather mismatches={perr}")
        # quant + activation: run + finiteness / shape sanity (fp4 has no easy independent ref).
        quant1()
        activation()
        quant2()
        torch.cuda.synchronize()
        fin = (
            torch.isfinite(t["q1_ptsf"]).all().item()
            and torch.isfinite(t["activated"][:e].float()).all().item()
            and torch.isfinite(t["q2_ptsf"][:e]).all().item()
        )
        print(
            f"{'PASS' if fin else 'FAIL'} quant/activation finiteness={fin} "
            f"(q1_fp4{tuple(t['q1_fp4'].shape)} activated{tuple(t['activated'].shape)} "
            f"q2_fp4{tuple(t['q2_fp4'].shape)})"
        )
        raise SystemExit(0 if (perr == 0 and fin) else 1)

    us_p = bench_ms(permute) * 1000
    us_q1 = bench_ms(quant1) * 1000
    us_a = bench_ms(activation) * 1000
    us_q2 = bench_ms(quant2) * 1000
    print(
        f"BENCH fp4_lora_moe_kernels decode num_tokens={nt} maxpad={mp} hidden={H} inter={I}:\n"
        f"  permuteKernel                 = {us_p:7.2f} us\n"
        f"  nvfp4 quant #1 (gate_up, m={mp})  = {us_q1:7.2f} us\n"
        f"  activationKernel              = {us_a:7.2f} us\n"
        f"  nvfp4 quant #2 (down, m={nt*tk})  = {us_q2:7.2f} us"
    )


if __name__ == "__main__":
    main()
