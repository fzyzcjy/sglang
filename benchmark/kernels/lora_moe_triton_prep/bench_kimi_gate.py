"""Self-contained perf-bench + correctness-test for kimi_k2_moe_fused_gate
(sgl_kernel::kimi_k2_moe_fused_gate, the Kimi routing gate, ~5us), on the EP8
bs64 Kimi-K2.5-NVFP4 decode path.

Production decode shapes (SHAPE_REPORT.md, decode bs64):
  IN gating_output (64,384) f32, correction_bias (384,) f32, topk=8,
  num_expert_group=1, renormalize=True, routed_scaling_factor=2.827
  OUT topk_weights (64,8) f32, topk_ids (64,8) i32.

Reached e2e via biased_grouped_topk_gpu (topk.py) when num_experts==384 and
num_expert_group==1.

Usage (on the GPU pod):
  python3 bench_kimi_gate.py --mode bench
  python3 bench_kimi_gate.py --mode correctness
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.testing
from sgl_kernel import kimi_k2_moe_fused_gate


def make_inputs(bs, num_experts, device):
    torch.manual_seed(0)
    gating = torch.randn(bs, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32) * 0.1
    return gating, bias


def gate(gating, bias, topk, renormalize, rsf):
    """kimi_k2_moe_fused_gate, mirroring the biased_grouped_topk_gpu call site."""
    return kimi_k2_moe_fused_gate(
        gating,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=rsf,
        apply_routed_scaling_factor_on_output=False,
    )


def ref_gate(gating, bias, topk, renormalize, rsf):
    """fp reference: sigmoid score, select top-k by (score+bias), weights = score
    at selected, renormalized, * routed_scaling_factor. (num_expert_group=1 ->
    no real grouping.)"""
    scores = torch.sigmoid(gating.float())
    biased = scores + bias.float()
    idx = torch.topk(biased, topk, dim=-1).indices
    w = torch.gather(scores, -1, idx)
    if renormalize:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    # NOTE: the call uses apply_routed_scaling_factor_on_output=False, so rsf is
    # NOT folded into topk_weights here (it is applied to the MoE output later).
    return w, idx


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
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--rsf", type=float, default=2.827)
    args = ap.parse_args()
    dev = "cuda"
    gating, bias = make_inputs(args.bs, args.num_experts, dev)

    w, ids = gate(gating, bias, args.topk, True, args.rsf)
    print(
        f"SHAPES topk_weights={tuple(w.shape)}/{w.dtype} topk_ids={tuple(ids.shape)}/{ids.dtype}"
    )

    if args.mode == "correctness":
        rw, rids = ref_gate(gating, bias, args.topk, True, args.rsf)
        # Compare selected expert SETS per row (order may differ between impls).
        set_ok = all(
            set(ids[m].tolist()) == set(rids[m].tolist()) for m in range(args.bs)
        )
        # Compare per-row SORTED weights (value match regardless of order).
        wk = torch.sort(w, dim=-1).values
        wr = torch.sort(rw, dim=-1).values
        werr = float((wk - wr).abs().max().item())
        ok = set_ok and werr <= 5e-2
        print(
            f"{'PASS' if ok else 'FAIL'} gate: topk_id_set_match={set_ok} "
            f"max_weight_err={werr:.4e}"
        )
        raise SystemExit(0 if ok else 1)

    us = bench_ms(lambda: gate(gating, bias, args.topk, True, args.rsf)) * 1000
    print(
        f"BENCH kimi_gate bs={args.bs} experts={args.num_experts} topk={args.topk}: "
        f"{us:.2f} us"
    )


if __name__ == "__main__":
    main()
