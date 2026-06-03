"""Self-contained perf-bench + correctness-test for the triton-gemm PREP group
on the EP8 bs64 Kimi-K2.5-NVFP4 LoRA decode path. The 3 prep kernels run
back-to-back and are benched as ONE combined pipeline (not individually), so a
future fused replacement can be compared apples-to-apples:

    topk_ids --_fused_virtual_topk_ids_kernel--> virtual_topk_ids        (~1.5us)
             --moe_align_block_size (sgl_kernel)--> sorted/expert/post_pad
               (this one launches BOTH moe_align_block_size_kernel ~2.7us
                and count_and_sort_expert_tokens_kernel ~4.7us)

Production decode shapes (SHAPE_REPORT.md, decode bs64, per-rank EP8):
    bs=64, top_k=8, num_experts=384, max_loras=1, local_num_experts=48 (384/8),
    block_size=16 -> virtual_num_experts=384.

These kernels read only a few KB/call (topk_ids is 64x8 i32), i.e. they are
latency/launch bound and their working set is far below L2 -> L2 state does not
affect the number (common_bench will print WARN: footprint<L2). Timing + buffer
rotation come from common_bench (CUDA-graph via triton do_bench_cudagraph).

Usage (on the GPU pod):
    python3 bench_triton_gemm_prep.py --mode bench
    python3 bench_triton_gemm_prep.py --mode correctness
"""

from __future__ import annotations

import argparse

import torch
from common_bench import bench_kernel, pick_n_sets, report_sets, set_bytes

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import _fused_virtual_topk_ids


def make_input_set(bs, top_k, num_experts, device):
    """Routing inputs for one rotation slot. token_lora_mapping=0 (single adapter)."""
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    return {"topk_ids": topk_ids, "tlm": token_lora_mapping}


def prep_pipeline(s, num_experts, local_num_experts, block_size):
    """The full triton-gemm prep: virtual topk ids -> native align (+count_and_sort)."""
    vtopk, _, vne = _fused_virtual_topk_ids(
        s["topk_ids"],
        s["tlm"],
        num_experts,
        shared_outer=False,
        max_loras=1,
        local_expert_offset=0,
        local_num_experts=local_num_experts,
    )
    return moe_align_block_size(vtopk, block_size, vne)


def ref_virtual_topk(topk_ids, tlm, num_experts, local_num_experts):
    bs, top_k = topk_ids.shape
    out = torch.empty_like(topk_ids)
    for mrow in range(bs):
        lora = int(tlm[mrow].item())
        safe = max(lora, 0)
        for k in range(top_k):
            base = int(topk_ids[mrow, k].item())
            owned = 0 <= base < local_num_experts
            base = base if owned else -1
            res = base if base < 0 else base + safe * num_experts
            out[mrow, k] = res if lora >= 0 else -1
    return out


def ref_align(vtopk, block, vne):
    """torch reference for sgl_kernel::moe_align_block_size (native wrapper passes
    num_experts+1 and the kernel uses the +1 offset: id -> id+1, so -1 maps to the
    sentinel bucket 0 and expert e to bucket e+1; each bucket is padded up to a
    block multiple; expert_ids labels each block bucket-1). Returns
    (num_tokens_post_padded, expert_ids_multiset)."""
    from collections import Counter

    buckets = (vtopk.reshape(-1) + 1).clamp(min=0)  # -1 -> 0 (sentinel), e -> e+1
    counts = torch.bincount(buckets, minlength=vne + 1)
    blocks = (counts + block - 1) // block
    post = int((blocks * block).sum().item())
    eids = Counter()
    for b in range(vne + 1):
        nb = int(blocks[b].item())
        if nb:
            eids[b - 1] += nb  # bucket 0 -> -1 sentinel label
    return post, eids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "correctness"], default="bench")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--local-num-experts", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--budget-gb", type=float, default=16.0)
    ap.add_argument("--n-sets", type=int, default=0, help="0 = auto (fill --budget-gb)")
    args = ap.parse_args()
    dev = "cuda"
    ne, lne, blk = args.num_experts, args.local_num_experts, args.block_size
    mk = lambda: make_input_set(args.bs, args.top_k, ne, dev)

    if args.mode == "correctness":
        s = mk()
        vtopk, _, vne = _fused_virtual_topk_ids(
            s["topk_ids"],
            s["tlm"],
            ne,
            shared_outer=False,
            max_loras=1,
            local_expert_offset=0,
            local_num_experts=lne,
        )
        ref = ref_virtual_topk(s["topk_ids"], s["tlm"], ne, lne)
        verr = int((vtopk != ref).sum().item())  # bitwise exact (integer ids)
        print(
            f"{'PASS' if verr == 0 else 'FAIL'} virtual_topk_ids bitwise mismatches={verr}"
        )
        from collections import Counter

        sorted_ids, expert_ids, post_pad = moe_align_block_size(vtopk, blk, vne)
        post_ref, eids_ref = ref_align(vtopk, blk, vne)
        pp = int(post_pad.item())
        nblk = pp // blk
        eids_k = Counter(e for e in expert_ids[:nblk].tolist())
        post_ok = pp == post_ref
        eids_ok = eids_k == eids_ref
        print(
            f"{'PASS' if (post_ok and eids_ok) else 'FAIL'} align vs ref: "
            f"post_pad={pp} (ref {post_ref}) expert_ids_multiset_match={eids_ok}"
        )
        raise SystemExit(0 if (verr == 0 and post_ok and eids_ok) else 1)

    per = set_bytes(mk())
    n_sets = pick_n_sets(per, args.budget_gb, args.n_sets)
    S = [mk() for _ in range(n_sets)]
    call = lambda i: prep_pipeline(S[i], ne, lne, blk)
    us = bench_kernel(call, n_sets) * 1000
    print(
        f"BENCH triton-gemm prep (COMBINED: virtual_topk_ids + moe_align + count_and_sort) "
        f"bs={args.bs} top_k={args.top_k} experts={ne} local_experts={lne} block={blk}"
    )
    print(f"  per_set={per/1e3:.1f}KB {report_sets(per, n_sets)}")
    print(f"  combined prep pipeline        = {us:7.2f} us")


if __name__ == "__main__":
    main()
