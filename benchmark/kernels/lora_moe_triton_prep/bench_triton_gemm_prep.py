"""Self-contained perf-bench + correctness-test for the triton-gemm PREP group
(3 kernels, ONE script) on the EP8 bs64 Kimi-K2.5-NVFP4 LoRA decode path:

    _fused_virtual_topk_ids_kernel   (virtual_experts.py, ~1.5us)
    moe_align_block_size_kernel      (sgl_kernel moe_align, ~2.7us)
    count_and_sort_expert_tokens_kernel (same sgl_kernel call, ~4.7us)

The three run back-to-back as the LoRA virtual-experts routing prep:
    topk_ids --_fused_virtual_topk_ids--> virtual_topk_ids
             --moe_align_block_size(native, sgl_kernel)--> sorted_token_ids/expert_ids/num_post_pad
(`moe_align_block_size` internally launches BOTH the align kernel and the
count_and_sort kernel, so the two P2 align kernels are one Python call.)

Production decode shapes (from SHAPE_REPORT.md, decode bs64, per-rank EP8):
    bs=64, top_k=8, num_experts=384, max_loras=1, local_num_experts=48 (384/8),
    block_size=16 -> virtual_num_experts=384; e2e: topk_ids (64,8) i32 (numel 512)
    -> virtual_topk_ids (64,8) i32 -> sorted_ids (6287,) expert_ids (393,).

Usage (run on the GPU pod):
    python3 bench_triton_gemm_prep.py --mode bench
    python3 bench_triton_gemm_prep.py --mode correctness
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.testing

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import _fused_virtual_topk_ids


def make_inputs(bs, top_k, num_experts, local_num_experts, device):
    """Decode routing inputs. token_lora_mapping=0 (single adapter, lora id 0)."""
    torch.manual_seed(0)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    return topk_ids, token_lora_mapping


def virtual_topk(topk_ids, token_lora_mapping, num_experts, local_num_experts):
    """kernel 1: _fused_virtual_topk_ids_kernel (EP-local, single adapter)."""
    return _fused_virtual_topk_ids(
        topk_ids,
        token_lora_mapping,
        num_experts,
        shared_outer=False,
        max_loras=1,
        local_expert_offset=0,
        local_num_experts=local_num_experts,
    )


def align(virtual_topk_ids, block_size, virtual_num_experts):
    """kernels 2+3: moe_align_block_size_kernel + count_and_sort_expert_tokens_kernel
    (one sgl_kernel::moe_align_block_size call)."""
    return moe_align_block_size(virtual_topk_ids, block_size, virtual_num_experts)


def ref_virtual_topk(topk_ids, token_lora_mapping, num_experts, local_num_experts):
    """fp reference for _fused_virtual_topk_ids (single adapter, EP-local mask)."""
    bs, top_k = topk_ids.shape
    out = torch.empty_like(topk_ids)
    mask = torch.empty(bs, dtype=torch.bool, device=topk_ids.device)
    off, n_local = 0, local_num_experts
    for m in range(bs):
        lora = int(token_lora_mapping[m].item())
        mask[m] = lora >= 0
        safe = max(lora, 0)
        for k in range(top_k):
            base = int(topk_ids[m, k].item())
            owned = off <= base < off + n_local
            base = base if owned else -1
            res = base if base < 0 else base + safe * num_experts
            out[m, k] = res if (lora >= 0) else -1
    return out


def bench_ms(fn, warmup=25, rep=100, inner=200):
    """Per-call ms via CUDA-graph capture of `inner` back-to-back calls / inner
    (amortizes the fixed per-replay launch overhead -> true device time)."""
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
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--local-num-experts", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"

    topk_ids, tlm = make_inputs(
        args.bs, args.top_k, args.num_experts, args.local_num_experts, dev
    )
    vtopk, _, vne = virtual_topk(
        topk_ids, tlm, args.num_experts, args.local_num_experts
    )

    if args.mode == "correctness":
        ref = ref_virtual_topk(topk_ids, tlm, args.num_experts, args.local_num_experts)
        err = int((vtopk != ref).sum().item())
        print(f"{'PASS' if err == 0 else 'FAIL'} virtual_topk_ids mismatches={err}")
        # align invariants: post_pad divisible by block_size; expert_ids length = post_pad/block.
        sorted_ids, expert_ids, post_pad = align(vtopk, args.block_size, vne)
        pp = int(post_pad.item())
        ok = (pp % args.block_size == 0) and (
            expert_ids.numel() >= pp // args.block_size
        )
        print(
            f"{'PASS' if ok else 'FAIL'} align post_pad={pp} "
            f"(%block={pp % args.block_size}) expert_ids={expert_ids.numel()} "
            f"sorted_ids={sorted_ids.numel()}"
        )
        raise SystemExit(0 if (err == 0 and ok) else 1)

    # bench: shapes echo for the 2-ii cross-check.
    sorted_ids, expert_ids, post_pad = align(vtopk, args.block_size, vne)
    print(
        f"SHAPES vtopk={tuple(vtopk.shape)}/{vtopk.dtype} "
        f"sorted_ids={tuple(sorted_ids.shape)} expert_ids={tuple(expert_ids.shape)} "
        f"virtual_num_experts={vne}"
    )
    us_vtopk = (
        bench_ms(
            lambda: virtual_topk(
                topk_ids, tlm, args.num_experts, args.local_num_experts
            )
        )
        * 1000
    )
    us_align = bench_ms(lambda: align(vtopk, args.block_size, vne)) * 1000
    us_all = (
        bench_ms(
            lambda: align(
                virtual_topk(topk_ids, tlm, args.num_experts, args.local_num_experts)[
                    0
                ],
                args.block_size,
                vne,
            )
        )
        * 1000
    )
    print(
        f"BENCH prep bs={args.bs} top_k={args.top_k} experts={args.num_experts} "
        f"local_experts={args.local_num_experts} block={args.block_size}:\n"
        f"  _fused_virtual_topk_ids        = {us_vtopk:7.2f} us\n"
        f"  moe_align (+count_and_sort)    = {us_align:7.2f} us\n"
        f"  combined (vtopk+align)         = {us_all:7.2f} us"
    )


if __name__ == "__main__":
    main()
