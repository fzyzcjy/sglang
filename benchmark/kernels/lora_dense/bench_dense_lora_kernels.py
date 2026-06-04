"""Self-contained micro-benchmark + correctness check for the small dense LoRA Triton
kernels: ``_sgemm_lora_a_kernel`` (shrink), ``_sgemm_lora_b_kernel`` (expand-add) and
``_gate_up_lora_b_kernel`` (2-slice expand-add).

Shapes are the FULL set measured e2e on Qwen3.5-35B-A3B-FP8 tp4/ep4 decode bs64 with a
single rank-16 adapter (SHAPECAP capture 2026-06-04), one entry per distinct module
signature -- see ``SHAPES`` below (in_proj_qkvz / qkv_proj / o_proj / shared expert
gate_up+down / lm_head).

Dispatch note: in the pre-merge e2e capture, production ``sgemm_lora_a_fwd`` took the
``F.linear`` fast path in the merged single-adapter decode batch; since the lora-mq-a
merge that path needs the ``SGLANG_OPT_LORA_CUBLAS(_A)`` env opt-in and the default is
the Triton kernel. The bench reports BOTH paths for every lora_a shape (the env variant
is labeled ``F.linear(env)``).

Decode batch_info reproduces production: merged single segment (bs=1, seg_lens=[64]),
permutation (SORTED_BY_ADAPTER=True), uniform adapter slot 0 / rank 16 / scaling 2.0,
use_cuda_graph=True.

Benchmark methodology: rotate N auto-sized buffer groups (footprint = ``--l2-mult`` x
L2, default 4x) inside one CUDA graph timed by ``triton.testing.do_bench_cudagraph``;
reported time = graph time / N. This amortizes host launch overhead to ~0 and prevents
any input from being served out of L2 on its next use.

  python3 bench_dense_lora_kernels.py --mode bench
  python3 bench_dense_lora_kernels.py --mode bench --only lm_head.B
  python3 bench_dense_lora_kernels.py --mode correctness
  python3 bench_dense_lora_kernels.py --mode profile --only o_proj.B --iters 4  # for ncu
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import triton
import triton.testing

from sglang.srt.lora.triton_ops.gate_up_lora_b import gate_up_lora_b_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_a_v2 import sgemm_lora_a_v2_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo

# All distinct dense-LoRA kernel signatures measured e2e (qwen3.5-35b tp4/ep4 decode
# bs64, rank 16). "calls/step" is per rank per decode step, for context.
#   kernel=sgemm_a: weights [1, stack_num*rank, K], x [s, K] -> out [s, stack_num*rank]
#   kernel=sgemm_b: weights [1, N, rank], x [s, rank] -> base_output [s, N] (+=)
#   kernel=gate_up_b: weights [1, 2*output_dim, rank], x [s, 2*rank] -> base [s, 2*output_dim] (+=)
SHAPES = [
    ("in_proj_qkvz.A", "sgemm_a", {"K": 2048, "stack_num": 4}),  # 30 calls/step
    ("qkv_proj.A", "sgemm_a", {"K": 2048, "stack_num": 3}),  # 10 calls/step
    ("shared_gate_up.A", "sgemm_a", {"K": 2048, "stack_num": 2}),  # 40 calls/step
    ("o_proj.A", "sgemm_a", {"K": 1024, "stack_num": 1}),  # 40 calls/step
    ("shared_down.A", "sgemm_a", {"K": 128, "stack_num": 1}),  # 40 calls/step
    ("lm_head.A", "sgemm_a", {"K": 2048, "stack_num": 1}),  # 1 call/step
    ("o_proj+shared_down.B", "sgemm_b", {"N": 2048}),  # 80 calls/step
    ("lm_head.B", "sgemm_b", {"N": 62080}),  # 1 call/step (vocab 248320 / tp4)
    ("shared_gate_up.B", "gate_up_b", {"output_dim": 128}),  # 40 calls/step
]


def disable_pdl(modules) -> None:
    """Launch kernels without PDL (launch_pdl/gdc_wait). The default PDL launch lets
    back-to-back identical kernels in the bench graph overlap launch tails, reporting
    a faster per-call time than an e2e nsys duration, which includes the gdc_wait
    stall on a DIFFERENT (often slower) producer kernel. --no-pdl gives the
    standalone-execution number for comparing against e2e profile durations."""
    import sglang.srt.lora.triton_ops.kernel_utils as _ku

    def no_pdl():
        return False, {}

    _ku.get_pdl_launch_metadata = no_pdl
    for mod in modules:
        mod.get_pdl_launch_metadata = no_pdl
    globals()["get_pdl_launch_metadata"] = no_pdl


def make_merged_decode_batch_info(
    s: int,
    rank: int,
    scaling: float,
    device,
    shuffle_permutation: bool = False,
) -> LoRABatchInfo:
    """Production single-adapter decode batch info: one merged segment (bs=1) with a
    token permutation. Since the lora-mq-a merge, sgemm_lora_a_fwd's F.linear fast
    path is selected by the SGLANG_OPT_LORA_CUBLAS(_A) env (see cublas_a_env), not by
    batch_info fields."""
    permutation = torch.arange(s, dtype=torch.int32, device=device)
    if shuffle_permutation:
        permutation = permutation[torch.randperm(s, device=device)]
    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, s], dtype=torch.int64, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([rank], dtype=torch.int64, device=device),
        scalings=torch.tensor([scaling], dtype=torch.float32, device=device),
        max_len=s,
        seg_lens=torch.tensor([s], dtype=torch.int64, device=device),
        permutation=permutation,
        single_adapter=(0, rank),
    )


def make_inputs(name, kernel, spec, s, rank, dtype, device, seed=0):
    """Returns (x, weights, base_output_or_None) for one SHAPES entry."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype) * 0.1

    if kernel == "sgemm_a":
        x = rnd(s, spec["K"])
        weights = rnd(1, spec["stack_num"] * rank, spec["K"])
        return x, weights, None
    if kernel == "sgemm_b":
        x = rnd(s, rank)
        weights = rnd(1, spec["N"], rank)
        return x, weights, rnd(s, spec["N"])
    assert kernel == "gate_up_b"
    x = rnd(s, 2 * rank)
    weights = rnd(1, 2 * spec["output_dim"], rank)
    return x, weights, rnd(s, 2 * spec["output_dim"])


def make_call(kernel, spec, x, weights, base_output, batch_info):
    if kernel == "sgemm_a":
        return lambda: sgemm_lora_a_fwd(
            x, weights, batch_info, stack_num=spec["stack_num"]
        )
    if kernel == "sgemm_b":
        return lambda: sgemm_lora_b_fwd(x, weights, batch_info, base_output=base_output)
    assert kernel == "gate_up_b"
    return lambda: gate_up_lora_b_fwd(
        x, weights, batch_info, spec["output_dim"], base_output=base_output
    )


def ref_output(kernel, spec, x, weights, base_output, rank, scaling):
    """fp32 reference. sgemm_a has no scaling and writes its own output; the two
    expand kernels scale and add into base_output."""
    w = weights[0].float()
    if kernel == "sgemm_a":
        return x.float() @ w.t()
    if kernel == "sgemm_b":
        return base_output.float() + scaling * (x.float() @ w.t())
    assert kernel == "gate_up_b"
    out = base_output.float().clone()
    output_dim = spec["output_dim"]
    for i in range(2):
        lo, hi = i * output_dim, (i + 1) * output_dim
        xi = x[:, i * rank : (i + 1) * rank].float()
        out[:, lo:hi] += scaling * (xi @ w[lo:hi, :rank].t())
    return out


def group_bytes_of(kernel, spec, s, rank) -> int:
    # sgemm_a outputs are allocated inside the wrapper (graph-pool addresses may be
    # reused across rotated calls during capture); they are <2% of the group bytes
    # (e.g. 8KB of ~520KB at K=2048), so the L2-eviction sizing is unaffected.
    if kernel == "sgemm_a":
        return 2 * (
            s * spec["K"]
            + spec["stack_num"] * rank * spec["K"]
            + s * spec["stack_num"] * rank
        )
    if kernel == "sgemm_b":
        return 2 * (s * rank + spec["N"] * rank + s * spec["N"])
    return 2 * (
        s * 2 * rank + 2 * spec["output_dim"] * rank + s * 2 * spec["output_dim"]
    )


def auto_num_groups(
    group_bytes: int, l2_mult: float, min_groups: int, max_groups: int
) -> int:
    """Enough buffer groups that the rotation footprint is ``l2_mult`` x L2, so no
    group survives in L2 until its next use. Err on the high side: an optimized kernel
    that reads less memory needs MORE groups for the same eviction guarantee."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "L2_cache_size", 128 * 1024 * 1024)
    need = math.ceil(l2_bytes * l2_mult / max(group_bytes, 1))
    n = max(min_groups, min(need, max_groups))
    if n * group_bytes < l2_mult * l2_bytes:
        print(
            f"WARNING: rotation footprint {n * group_bytes / 1e6:.0f} MB < "
            f"{l2_mult:.1f} x L2 ({l2_mult * l2_bytes / 1e6:.0f} MB); raise "
            f"--max-groups for full L2 eviction (small-shape kernel)"
        )
    return n


def bench_us_rotated(calls, rep_ms: int, burn_in: int = 2) -> float:
    """Capture all rotated calls in ONE CUDA graph via do_bench_cudagraph; per-call us =
    graph time / num_groups.

    burn_in discards the first ``burn_in`` do_bench_cudagraph measurements and keeps the
    next. The FIRST do_bench_cudagraph in a process is systematically ~30% faster than
    steady-state on GB200 (NOT a clock effect -- it persists with SM+mem clocks locked; it
    is a per-process cudagraph warmup state). Without burn-in, the first config measured in
    any single-process sweep looks artificially fast and the sweep picks a bogus optimum.
    The burned-in steady-state value is order-independent, reproducible, and matches what
    sustained e2e cuda-graph decode (continuous replay) actually sees. See
    2026-06-04-triton-dobench-first-call-bias.md."""

    def fn():
        for call in calls:
            call()

    fn()  # eager warmup: triton JIT compile outside graph capture
    torch.cuda.synchronize()
    for _ in range(burn_in):
        triton.testing.do_bench_cudagraph(fn, rep=rep_ms)
    ms = triton.testing.do_bench_cudagraph(fn, rep=rep_ms)
    return float(ms) * 1e3 / len(calls)


def variants_for(name, kernel):
    """sgemm_a gets both dispatch paths; the expand kernels only have the Triton path
    by default (their cuBLAS alternatives need SGLANG_OPT_LORA_CUBLAS_* opt-ins)."""
    if kernel == "sgemm_a":
        return [("triton", False), ("F.linear(env)", True)]
    return [("triton", False)]


class cublas_a_env:
    """Scoped SGLANG_OPT_LORA_CUBLAS_A=1 (read live by envs.*.get() at each call):
    selects sgemm_lora_a_fwd's cuBLAS/F.linear path, the pre-merge production decode
    dispatch (post-merge default is the Triton kernel)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            os.environ["SGLANG_OPT_LORA_CUBLAS_A"] = "1"

    def __exit__(self, *exc):
        if self.enabled:
            os.environ.pop("SGLANG_OPT_LORA_CUBLAS_A", None)
        return False


class dense_v2_env:
    """Scoped SGLANG_OPT_LORA_DENSE_V2=1 (read live by envs.*.get()): selects the
    single-adapter specialized split-K v2 dense LoRA kernels in the production fwds."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            os.environ["SGLANG_OPT_LORA_DENSE_V2"] = "1"

    def __exit__(self, *exc):
        if self.enabled:
            os.environ.pop("SGLANG_OPT_LORA_DENSE_V2", None)
        return False


def _old_then_v2(name, kernel, spec, x, weights, base, bi):
    """Returns (old_out, v2_out) as fp32 for one shape. Each expand kernel gets its own
    base_output clone so the in-place += does not cross-contaminate."""
    if kernel == "sgemm_a":
        old = sgemm_lora_a_fwd(x, weights, bi, stack_num=spec["stack_num"])
        new = sgemm_lora_a_v2_fwd(x, weights, bi, stack_num=spec["stack_num"])
        return old.float(), new.float()
    if kernel == "sgemm_b":
        from sglang.srt.lora.triton_ops.sgemm_lora_b_v2 import sgemm_lora_b_v2_fwd

        old = sgemm_lora_b_fwd(x, weights, bi, base_output=base.clone())
        new = sgemm_lora_b_v2_fwd(x, weights, bi, base_output=base.clone())
        return old.float(), new.float()
    assert kernel == "gate_up_b"
    from sglang.srt.lora.triton_ops.gate_up_lora_b_v2 import gate_up_lora_b_v2_fwd

    old = gate_up_lora_b_fwd(x, weights, bi, spec["output_dim"], base_output=base.clone())
    new = gate_up_lora_b_v2_fwd(
        x, weights, bi, spec["output_dim"], base_output=base.clone()
    )
    return old.float(), new.float()


def run_v2_vs_old_guardrail(args, shapes, dtype, device) -> None:
    """new-vs-old guardrail. The old kernel is the reference.

    The two LoRA-B expand kernels (sgemm_b, gate_up_b) keep the OLD kernel's exact
    arithmetic, so they must be BITWISE IDENTICAL (max_abs_err == 0); a guardrail FAIL is
    raised otherwise. The LoRA-A shrink (sgemm_a) uses split-K, which reorders the float
    reduction, so it cannot be bitwise -- it is held to a tight numerical bound (the
    split-K fp32 accumulation is actually more accurate than the old kernel, so the
    difference is at the bf16-output quantization floor)."""
    # sgemm_a: split-K reorders the sum -> tight numerical bound (bf16 output ULP).
    A_ABS, A_REL = 2e-2, 5e-3
    failures = 0
    # Cover the partial-last-tile path too: bs not a multiple of BLOCK_S(=16), and bs=1.
    # The v2 kernels substitute max_len for the per-segment seg_len and re-tile the output,
    # so the s_offset<seg_len / n_offset<N masks are only exercised when bs % 16 != 0.
    for bs in sorted({args.bs, 17, 1}):
        for shuffle in [False, True]:
            for name, kernel, spec in shapes:
                bi = make_merged_decode_batch_info(
                    bs, args.rank, args.scaling, device, shuffle_permutation=shuffle
                )
                x, weights, base = make_inputs(
                    name, kernel, spec, bs, args.rank, dtype, device
                )
                old, new = _old_then_v2(name, kernel, spec, x, weights, base, bi)
                err = float((new - old).abs().max().item())
                rel = err / float(old.abs().max().item() + 1e-9)
                finite = bool(torch.isfinite(new).all() and torch.isfinite(old).all())
                if kernel == "sgemm_a":
                    ok = finite and (err <= A_ABS or rel <= A_REL)
                    mode = "close"
                else:
                    ok = finite and err == 0.0  # B kernels must be bitwise identical
                    mode = "BITWISE" if err == 0.0 else "DIFF"
                failures += int(not ok)
                print(
                    f"{'PASS' if ok else 'FAIL'} v2-vs-old {name:<22s} [{mode:<7s}] "
                    f"bs={bs:<3d} shuffled={int(shuffle)} max_abs_err={err:.4e} rel={rel:.2e}"
                )
    if failures:
        raise SystemExit(1)


def run_sweep_a_v2(args, shapes, dtype, device) -> None:
    """Sweep (BLOCK_S, BLOCK_K, SPLIT_K, num_warps, num_stages) for sgemm_a v2 on each
    sgemm_a shape; report the fastest config (PDL-rotated us)."""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    s = args.bs
    for name, kernel, spec in shapes:
        if kernel != "sgemm_a":
            continue
        K = spec["K"]
        num_s_tiles = triton.cdiv(s, 16)
        block_ks = [bk for bk in (32, 64, 128, 256) if bk <= max(K, 32)]
        candidates = []
        for block_k in block_ks:
            num_k_tiles = triton.cdiv(K, block_k)
            for split_k in sorted(
                {
                    1,
                    min(num_k_tiles, 4),
                    min(num_k_tiles, 8),
                    min(num_k_tiles, 16),
                    min(num_k_tiles, max(1, 2 * num_sms // num_s_tiles)),
                    num_k_tiles,
                }
            ):
                for num_warps in (2, 4):
                    for num_stages in (2, 3, 4):
                        candidates.append(
                            {
                                "BLOCK_K": block_k,
                                "SPLIT_K": split_k,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                        )
        group_bytes = group_bytes_of(kernel, spec, s, args.rank)
        num_groups = args.num_groups or auto_num_groups(
            group_bytes, args.l2_mult, args.min_groups, args.max_groups
        )
        groups = [
            make_inputs(name, kernel, spec, s, args.rank, dtype, device, seed=g)
            for g in range(num_groups)
        ]
        bi = make_merged_decode_batch_info(s, args.rank, args.scaling, device)
        best = None
        for cfg in candidates:
            calls = [
                (
                    lambda x=x, w=w: sgemm_lora_a_v2_fwd(
                        x, w, bi, stack_num=spec["stack_num"], config=cfg
                    )
                )
                for x, w, _ in groups
            ]
            try:
                us = bench_us_rotated(calls, args.rep_ms)
            except Exception as e:
                continue
            if best is None or us < best[0]:
                best = (us, cfg)
        us, cfg = best
        print(
            f"SWEEP-A-V2 {name:<22s} K={K} best={us:.2f} us  "
            f"BLOCK_K={cfg['BLOCK_K']} SPLIT_K={cfg['SPLIT_K']} "
            f"warps={cfg['num_warps']} stages={cfg['num_stages']}  "
            f"(grid={num_s_tiles}x{cfg['SPLIT_K']}={num_s_tiles * cfg['SPLIT_K']} CTA)"
        )


def run_correctness(args, shapes, dtype, device) -> None:
    failures = 0
    for shuffle in [False, True]:
        for name, kernel, spec in shapes:
            for variant, use_cublas_a in variants_for(name, kernel):
                bi = make_merged_decode_batch_info(
                    args.bs,
                    args.rank,
                    args.scaling,
                    device,
                    shuffle_permutation=shuffle,
                )
                x, weights, base = make_inputs(
                    name, kernel, spec, args.bs, args.rank, dtype, device
                )
                base_run = base.clone() if base is not None else None
                with cublas_a_env(use_cublas_a):
                    out = make_call(kernel, spec, x, weights, base_run, bi)()
                if kernel != "sgemm_a":
                    out = base_run
                ref = ref_output(
                    kernel, spec, x, weights, base, args.rank, args.scaling
                )
                err = float((out.float() - ref).abs().max().item())
                rel = err / float(ref.abs().max().item() + 1e-9)
                # bf16 output quantization makes the achievable ABS error scale with the
                # output magnitude (large for the K=2048 shrinks), so accept either bound.
                ok = err <= args.tol or rel <= args.rtol
                failures += int(not ok)
                print(
                    f"{'PASS' if ok else 'FAIL'} {name:<22s} {variant:<14s} "
                    f"shuffled={int(shuffle)} max_abs_err={err:.4e} rel={rel:.2e}"
                )
    if failures:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["bench", "correctness", "profile", "guardrail", "sweepa"],
        default="bench",
    )
    ap.add_argument(
        "--v2",
        action="store_true",
        help="enable SGLANG_OPT_LORA_DENSE_V2 (specialized split-K dense kernels)",
    )
    ap.add_argument("--only", default=None, help="run a single SHAPES entry by name")
    ap.add_argument("--bs", type=int, default=64, help="decode batch size (1 tok/req)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=2.0)
    ap.add_argument(
        "--num-groups",
        type=int,
        default=0,
        help="rotated buffer groups; 0 = auto-size to --l2-mult x L2",
    )
    ap.add_argument("--l2-mult", type=float, default=4.0)
    ap.add_argument("--min-groups", type=int, default=32)
    ap.add_argument("--max-groups", type=int, default=8192)
    ap.add_argument("--rep-ms", type=int, default=100)
    ap.add_argument("--iters", type=int, default=4, help="profile-mode eager sweeps")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument(
        "--no-pdl", action="store_true", help="disable PDL (see disable_pdl docstring)"
    )
    args = ap.parse_args()
    if args.no_pdl:
        import sglang.srt.lora.triton_ops.gate_up_lora_b as _g
        import sglang.srt.lora.triton_ops.sgemm_lora_a as _a
        import sglang.srt.lora.triton_ops.sgemm_lora_b as _b

        disable_pdl([_a, _b, _g])

    device = "cuda"
    dtype = torch.bfloat16
    shapes = [e for e in SHAPES if args.only is None or e[0] == args.only]
    assert shapes, f"--only {args.only!r} matches no SHAPES entry"

    if args.mode == "correctness":
        run_correctness(args, shapes, dtype, device)
        return
    if args.mode == "guardrail":
        run_v2_vs_old_guardrail(args, shapes, dtype, device)
        return
    if args.mode == "sweepa":
        run_sweep_a_v2(args, shapes, dtype, device)
        return

    s = args.bs
    for name, kernel, spec in shapes:
        for variant, use_cublas_a in variants_for(name, kernel):
            if args.v2 and variant != "triton":
                continue  # v2 replaces the triton path; skip the F.linear variant
            bi = make_merged_decode_batch_info(s, args.rank, args.scaling, device)
            group_bytes = group_bytes_of(kernel, spec, s, args.rank)
            num_groups = args.num_groups or auto_num_groups(
                group_bytes, args.l2_mult, args.min_groups, args.max_groups
            )
            groups = [
                make_inputs(name, kernel, spec, s, args.rank, dtype, device, seed=g)
                for g in range(num_groups)
            ]
            calls = [make_call(kernel, spec, x, w, base, bi) for x, w, base in groups]

            label = "v2" if args.v2 else variant
            if args.mode == "profile":
                with cublas_a_env(use_cublas_a), dense_v2_env(args.v2):
                    for _ in range(2):
                        calls[0]()
                    torch.cuda.synchronize()
                    for _ in range(args.iters):
                        for call in calls:
                            call()
                    torch.cuda.synchronize()
                print(f"PROFILE {name} [{label}]: {args.iters} x {num_groups} groups")
                continue

            with cublas_a_env(use_cublas_a), dense_v2_env(args.v2):
                us = bench_us_rotated(calls, args.rep_ms)
            dims = " ".join(f"{k}={v}" for k, v in spec.items())
            print(
                f"BENCH {name:<22s} [{label:<14s}] s={s} r={args.rank} {dims:<22s} "
                f"groups={num_groups} ({group_bytes * num_groups / 1e6:.0f} MB rotated): "
                f"{us:.2f} us"
            )


if __name__ == "__main__":
    main()
