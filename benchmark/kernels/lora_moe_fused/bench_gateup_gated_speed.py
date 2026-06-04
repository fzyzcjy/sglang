"""Speed check for the gate_up gated-split fix: it must be perf-neutral.

The fix makes the rank-specialized expand contract the up-shrink columns [R:2R] for
the up output half (instead of gate_A's [0:R]) via an in-kernel column remap
(``a_col = offs_r + where(pid_n*BLOCK_N >= N/2, R, 0)``). That adds only a per-tile
``tl.where``; it reads the same B weight once and does the same dot. This bench times
the gate_up expand with the gated remap (fix, intermediate width 2R) against an
otherwise-identical non-gated launch (intermediate width R, reads [0:R] for all tiles),
on the e2e Qwen3.5 decode shape, to confirm the fix does not regress speed.

  python3 bench_gateup_gated_speed.py
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
from sglang.srt.lora.trtllm_moe.specialized_expand import _invoke_moe_lora_expand_add

QWEN35_EP4 = {
    "num_experts": 256,
    "local_num_experts": 64,
    "local_expert_offset": 0,
    "top_k": 8,
}
INTER, RANK = 512, 16  # gate_up: N = 2*INTER = 1024, rank 16


def build_routing(topk_ids, tlm, ep, block_m):
    vt, _, vne = _fused_virtual_topk_ids(
        topk_ids,
        tlm,
        ep["num_experts"],
        shared_outer=False,
        max_loras=1,
        local_expert_offset=ep["local_expert_offset"],
        local_num_experts=ep["local_num_experts"],
    )
    sti, eid, ntp = moe_align_block_size(vt, block_m, vne)
    n = topk_ids.numel()
    tight = (
        triton.cdiv(n + min(n, ep["local_num_experts"] + 1) * (block_m - 1), block_m)
        * block_m
    )
    return sti[:tight], eid[: tight // block_m], ntp


def make_inputs(bs, ep, device, dtype, gated, seed=0, skew_a=0.9):
    gen = torch.Generator(device=device).manual_seed(seed)
    pop = torch.arange(
        1, ep["num_experts"] + 1, dtype=torch.float32, device=device
    ).pow(-skew_a)
    pop = pop[torch.randperm(ep["num_experts"], generator=gen, device=device)]
    topk_ids = torch.multinomial(
        pop.expand(bs, -1), ep["top_k"], replacement=False, generator=gen
    ).to(torch.int32)
    topk_weights = torch.ones(bs, ep["top_k"], device=device, dtype=torch.float32)
    tlm = torch.zeros(bs, device=device, dtype=torch.int32)
    cols = 2 * RANK if gated else RANK
    inter = (
        torch.randn(bs * ep["top_k"], cols, generator=gen, device=device, dtype=dtype)
        * 0.1
    )
    weight = (
        torch.randn(
            ep["num_experts"],
            2 * INTER,
            RANK,
            generator=gen,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    output = torch.zeros(bs, ep["top_k"], 2 * INTER, device=device, dtype=dtype)
    return topk_ids, topk_weights, tlm, inter, weight, output


def bench_variant(bs, ep, device, dtype, gated, rep_ms, groups):
    cfg = {
        "BLOCK_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 1,
        "GROUP_SIZE_M": 1,
        "BLOCK_SIZE_N": 128,
    }
    g = [make_inputs(bs, ep, device, dtype, gated, seed=i) for i in range(groups)]
    rts = [build_routing(x[0], x[2], ep, 16) for x in g]

    def call_all():
        for (topk_ids, topk_weights, tlm, inter, weight, output), (
            sti,
            eid,
            ntp,
        ) in zip(g, rts):
            _invoke_moe_lora_expand_add(
                inter,
                weight,
                output,
                topk_weights,
                topk_ids,
                sti,
                eid,
                ntp,
                cfg,
                False,
                False,
                force_block_size_n=128,
            )

    call_all()
    torch.cuda.synchronize()
    return float(triton.testing.do_bench_cudagraph(call_all, rep=rep_ms)) * 1e3 / groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--rep-ms", type=int, default=400)
    ap.add_argument("--groups", type=int, default=120)
    args = ap.parse_args()
    device, dtype, ep = "cuda", torch.bfloat16, QWEN35_EP4

    nongated = bench_variant(
        args.bs, ep, device, dtype, False, args.rep_ms, args.groups
    )
    gated = bench_variant(args.bs, ep, device, dtype, True, args.rep_ms, args.groups)
    print(f"gate_up expand bs={args.bs} N={2*INTER} rank={RANK}")
    print(f"  non-gated (reads [0:R] for all tiles): {nongated:.3f} us")
    print(f"  gated split (FIX, up reads [R:2R]):    {gated:.3f} us")
    print(
        f"  fix overhead: {gated - nongated:+.3f} us ({(gated/nongated - 1)*100:+.1f}%)  "
        f"{'PERF-NEUTRAL' if abs(gated/nongated - 1) < 0.05 else 'REGRESSION'}"
    )


if __name__ == "__main__":
    main()
