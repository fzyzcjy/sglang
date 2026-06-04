"""Prototype: fused single-adapter LoRA-A(shrink)+B(expand) decode kernel, A/B'd against
the production two-kernel path (cuBLAS shrink + Triton expand) in the SAME process with
burn-in (so the do_bench first-call bias and cross-process state don't confound).

Fused kernel: out[S,N] += scaling * (x[S,K] @ A[K,r]) @ B[r,N]. One launch, no HBM
intermediate. Each (n_tile, s_tile) CTA recomputes the full-K shrink tmp[BLOCK_S,r] in
registers, then does its N-slice of the expand. Triton has no grid-wide barrier, so the
shrink's K-reduction cannot be split across CTAs the way cuBLAS does -- the redundant
per-N-tile shrink is the cost we are measuring. Expectation: wins for cheap K (e.g.
shared_down K=128), likely loses for heavy K (o_proj K=1024).

  python3 bench_fused_ab.py --mode correctness
  python3 bench_fused_ab.py --mode bench
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def _fused_lora_ab_kernel(
    x,
    A,  # [R, K] (single adapter, slice 0)
    B,  # [N, R]
    out,  # [S, N], pre-filled with base; we add into it
    S,
    K,
    N,
    R,
    scaling,
    x_s0,
    x_s1,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    o_s0,
    o_s1,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    pid_n = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    r_off = tl.arange(0, BLOCK_R)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    s_mask = s_off < S
    r_mask = r_off < R

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    # Phase 1: tmp[BLOCK_S, BLOCK_R] = x[s, :] @ A[r, :].T  (full-K reduction, redundant
    # across N-tiles since Triton can't share it without a grid barrier).
    tmp = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_off < K
        x_tile = tl.load(
            x + (s_off[:, None] * x_s0 + k_off[None, :] * x_s1),
            mask=s_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        a_tile = tl.load(
            A + (r_off[:, None] * a_s0 + k_off[None, :] * a_s1),
            mask=r_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        tmp += tl.dot(x_tile, a_tile.T)

    # Phase 2: out[s, n] += scaling * tmp[s, r] @ B[n, r].T
    n_mask = n_off < N
    b_tile = tl.load(
        B + (n_off[:, None] * b_s0 + r_off[None, :] * b_s1),
        mask=n_mask[:, None] & r_mask[None, :],
        other=0.0,
    )
    acc = tl.dot(tmp.to(b_tile.dtype), b_tile.T) * scaling

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    o_ptr = out + (s_off[:, None] * o_s0 + n_off[None, :] * o_s1)
    o_mask = s_mask[:, None] & n_mask[None, :]
    acc += tl.load(o_ptr, mask=o_mask).to(tl.float32)
    tl.store(o_ptr, acc.to(out.dtype.element_ty), mask=o_mask)


def fused_lora_ab(x, A, B, base_out, scaling, cfg):
    S, K = x.shape
    R = A.shape[0]
    N = B.shape[0]
    BLOCK_S = cfg.get("BLOCK_S", triton.next_power_of_2(S))
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    BLOCK_R = triton.next_power_of_2(R)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(S, BLOCK_S))
    _fused_lora_ab_kernel[grid](
        x, A, B, base_out, S, K, N, R, scaling,
        x.stride(0), x.stride(1), A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), base_out.stride(0), base_out.stride(1),
        BLOCK_S, BLOCK_N, BLOCK_K, BLOCK_R,
        num_warps=cfg.get("num_warps", 4), num_stages=cfg.get("num_stages", 3),
    )
    return base_out


def bench_us(fn, burn=2, rep=100, n=0):
    fn(); torch.cuda.synchronize()
    for _ in range(burn):
        triton.testing.do_bench_cudagraph(fn, rep=rep)
    ms = triton.testing.do_bench_cudagraph(fn, rep=rep)
    return float(ms) * 1e3 / max(n, 1)


def make_groups(S, K, N, R, ng, dev):
    g = []
    for i in range(ng):
        gen = torch.Generator(device=dev).manual_seed(i)
        rnd = lambda *sh: torch.randn(*sh, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1
        g.append((rnd(S, K), rnd(R, K), rnd(N, R), rnd(S, N)))
    return g


SHAPES = [
    ("shared_down", 64, 128, 2048, 16),   # cheap K -> fusion should win
    ("o_proj",      64, 1024, 2048, 16),  # heavy K -> fusion likely loses
    ("in_proj_qkvz_likeK", 64, 2048, 2048, 16),  # heaviest K reference
]
CFG = {"BLOCK_S": 64, "BLOCK_N": 256, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "correctness"], default="bench")
    ap.add_argument("--block-n", type=int, default=0)
    ap.add_argument("--block-k", type=int, default=0)
    ap.add_argument("--block-s", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    cfg = dict(CFG)
    if args.block_n:
        cfg["BLOCK_N"] = args.block_n
    if args.block_k:
        cfg["BLOCK_K"] = args.block_k
    if args.block_s:
        cfg["BLOCK_S"] = args.block_s

    if args.mode == "correctness":
        for name, S, K, N, R in SHAPES:
            x, A, B, base = make_groups(S, K, N, R, 1, dev)[0]
            ref = base.float() + 2.0 * ((x.float() @ A.float().t()) @ B.float().t())
            out = fused_lora_ab(x, A.contiguous(), B.contiguous(), base.clone(), 2.0, cfg)
            err = (out.float() - ref).abs().max().item()
            rel = err / (ref.abs().max().item() + 1e-9)
            print(f"{'PASS' if rel < 1e-2 else 'FAIL'} {name:20s} max_abs={err:.3e} rel={rel:.2e}")
        return

    ng = 512
    for name, S, K, N, R in SHAPES:
        groups = make_groups(S, K, N, R, ng, dev)
        # baseline: cuBLAS shrink (matmul) + Triton expand (atomic add into base)
        from sglang.srt.lora.triton_ops.sgemm_lora_b_v2 import sgemm_lora_b_v2_fwd
        from sglang.srt.lora.utils import LoRABatchInfo
        bi = LoRABatchInfo(
            use_cuda_graph=True, bs=1, num_segments=1,
            seg_indptr=torch.tensor([0, S], dtype=torch.int64, device=dev),
            weight_indices=torch.zeros(1, dtype=torch.int32, device=dev),
            lora_ranks=torch.tensor([R], dtype=torch.int64, device=dev),
            scalings=torch.tensor([2.0], dtype=torch.float32, device=dev),
            max_len=S, seg_lens=torch.tensor([S], dtype=torch.int64, device=dev),
            permutation=torch.arange(S, dtype=torch.int32, device=dev),
            single_adapter=(0, R),
        )
        def base_call(x, A, B, base):
            tmp = torch.matmul(x, A.t())  # cuBLAS shrink -> [S,R]
            return sgemm_lora_b_v2_fwd(tmp, B.unsqueeze(0), bi, base_output=base)
        base_calls = [(lambda x=x, A=A, B=B, base=base.clone(): base_call(x, A, B, base)) for x, A, B, base in groups]
        fused_calls = [(lambda x=x, A=A, B=B, base=base.clone(): fused_lora_ab(x, A, B, base, 2.0, cfg)) for x, A, B, base in groups]
        t_base = bench_us(lambda: [c() for c in base_calls], n=ng)
        t_fused = bench_us(lambda: [c() for c in fused_calls], n=ng)
        verdict = "WIN" if t_fused < t_base else "lose"
        print(f"{name:20s} K={K:5d}: base(cublas-shrink+v2-expand)={t_base:.3f}  fused={t_fused:.3f}  -> {verdict} ({round((t_base-t_fused)/t_base*100,1)}%)")


if __name__ == "__main__":
    main()
