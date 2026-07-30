"""8-GPU microbenchmark: SGLang DP-attention gather+combine, MAX_LEN vs SUM_LEN.

Reproduces exactly what sglang does per sparse layer:
  MAX_LEN : all_gather_into_tensor(global[max*D], local[max]) ... reduce_scatter_tensor
  SUM_LEN : global.fill_(0); copy local slice; all_reduce(global[sum]) ... all_reduce + slice

Launch:
  torchrun --nproc_per_node 8 bench_dp_collectives.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List

import torch
import torch.distributed as dist

HIDDEN = 4096
DTYPE = torch.bfloat16
WARMUP = 5
ITERS = 20

SCENARIOS: List[tuple[str, List[int]]] = [
    ("balanced_8x8192", [8192] * 8),
    ("balanced_8x4096", [4096] * 8),
    ("balanced_8x1024", [1024] * 8),
    ("skew_one_long", [8192, 512, 512, 512, 512, 512, 512, 512]),
    ("skew_prefill_plus_decode", [8192, 64, 64, 64, 64, 64, 64, 64]),
    ("skew_mild", [8192, 4096, 4096, 4096, 2048, 2048, 1024, 1024]),
]


@dataclass
class Result:
    scenario: str
    mode: str
    global_rows: int
    local_rows: int
    gather_ms: float
    combine_ms: float
    total_ms: float


def timed(fn: Callable[[], None]) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / ITERS * 1e3


def run_scenario(scenario: str, tokens: List[int], rank: int, world_size: int) -> List[Result]:
    results: List[Result] = []
    max_len = max(tokens)
    sum_len = sum(tokens)

    # --- MAX_LEN: local padded to max_len, all_gather + reduce_scatter ---
    local_max = torch.randn(max_len, HIDDEN, dtype=DTYPE, device="cuda")
    global_max = torch.empty(max_len * world_size, HIDDEN, dtype=DTYPE, device="cuda")
    gather_max = timed(lambda: dist.all_gather_into_tensor(global_max, local_max))
    combine_max = timed(lambda: dist.reduce_scatter_tensor(local_max, global_max))
    results.append(
        Result(
            scenario=scenario,
            mode="MAX_LEN",
            global_rows=max_len * world_size,
            local_rows=max_len,
            gather_ms=gather_max,
            combine_ms=combine_max,
            total_ms=gather_max + combine_max,
        )
    )
    del local_max, global_max
    torch.cuda.empty_cache()

    # --- SUM_LEN: local is own count, fill+copy+all_reduce, combine = all_reduce+slice ---
    local_rows = tokens[rank]
    offset = sum(tokens[:rank])
    local_sum = torch.randn(max(local_rows, 1), HIDDEN, dtype=DTYPE, device="cuda")
    global_sum = torch.empty(sum_len, HIDDEN, dtype=DTYPE, device="cuda")

    def gather_sum_len() -> None:
        global_sum.fill_(0)
        if local_rows > 0:
            global_sum[offset : offset + local_rows].copy_(local_sum[:local_rows])
        dist.all_reduce(global_sum)

    def combine_sum_len() -> None:
        dist.all_reduce(global_sum)
        if local_rows > 0:
            local_sum[:local_rows].copy_(global_sum[offset : offset + local_rows])

    gather_s = timed(gather_sum_len)
    combine_s = timed(combine_sum_len)
    results.append(
        Result(
            scenario=scenario,
            mode="SUM_LEN",
            global_rows=sum_len,
            local_rows=local_rows,
            gather_ms=gather_s,
            combine_ms=combine_s,
            total_ms=gather_s + combine_s,
        )
    )
    del local_sum, global_sum
    torch.cuda.empty_cache()

    # --- GATHERV proxy: variable-length gather costs the same as an all_gather
    # over a sum_len buffer (each rank contributes its own slice, no padding and
    # no 2x all_reduce). Emulated with an equal-split all_gather of sum_len rows,
    # which is the same total bytes on the wire.
    equal_rows = max(sum_len // world_size, 1)
    local_v = torch.randn(equal_rows, HIDDEN, dtype=DTYPE, device="cuda")
    global_v = torch.empty(equal_rows * world_size, HIDDEN, dtype=DTYPE, device="cuda")
    gather_v = timed(lambda: dist.all_gather_into_tensor(global_v, local_v))
    combine_v = timed(lambda: dist.reduce_scatter_tensor(local_v, global_v))
    results.append(
        Result(
            scenario=scenario,
            mode="GATHERV~",
            global_rows=equal_rows * world_size,
            local_rows=equal_rows,
            gather_ms=gather_v,
            combine_ms=combine_v,
            total_ms=gather_v + combine_v,
        )
    )
    del local_v, global_v
    torch.cuda.empty_cache()

    return results


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    all_results: List[Result] = []
    for scenario, tokens in SCENARIOS:
        assert len(tokens) == world_size, (scenario, len(tokens), world_size)
        all_results.extend(run_scenario(scenario, tokens, rank, world_size))

    if rank == 0:
        print(
            f"\nhidden={HIDDEN} dtype={DTYPE} world={world_size} iters={ITERS} "
            f"(times are rank-0 wall clock per call)\n"
        )
        header = (
            f"{'scenario':28} {'mode':8} {'global rows':>11} {'local rows':>10} "
            f"{'gather ms':>10} {'combine ms':>11} {'total ms':>9}"
        )
        print(header)
        print("-" * len(header))
        prev = None
        base = None
        for r in all_results:
            print(
                f"{r.scenario if r.mode == 'MAX_LEN' else '':28} {r.mode:8} "
                f"{r.global_rows:11d} {r.local_rows:10d} {r.gather_ms:10.3f} "
                f"{r.combine_ms:11.3f} {r.total_ms:9.3f}"
            )
            if r.mode == "SUM_LEN" and prev is not None:
                print(f"{'':28} {'sum/max':8} {'':11} {'':10} {'':10} {'':11} "
                      f"{r.total_ms / prev.total_ms:8.2f}x")
            if r.mode == "GATHERV~" and base is not None:
                print(f"{'':28} {'gv/max':8} {'':11} {'':10} {'':10} {'':11} "
                      f"{r.total_ms / base.total_ms:8.2f}x")
                print()
            if r.mode == "MAX_LEN":
                base = r
            prev = r

        out = os.environ.get("BENCH_OUT")
        if out:
            with open(out, "w") as f:
                json.dump([asdict(r) for r in all_results], f, indent=2)
            print(f"wrote {out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
