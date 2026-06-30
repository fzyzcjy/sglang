"""
Profile a DSpark SPS cost table (JSON) from an already-running non-spec server.

Instead of standing up its own DSpark worker and hand-rolling the scheduler's
batch-result threading to time a verify step, this profiler is a thin wrapper
around ``sglang.benchmark.one_batch_server``: it connects to a plain server the
user launched separately (NO speculative flags) and sweeps decode batch sizes,
reading the steady-state average inter-token latency (ITL) of each batch.

Core insight: one uniform verify step over ``batch_tokens`` tokens costs about
the same as one non-spec decode step over the same total token count -- both are
a single forward over that many tokens. A non-spec decode at batch size ``bs``
forwards exactly ``bs`` tokens (one per request), so:

    batch_tokens  = bs
    steps_per_sec = output_throughput / bs = 1000 / ITL_ms

(see ``one_batch_server.get_report_summary``: ``ITL_ms = 1000 * bs / output_throughput``).
We compute steps_per_sec from the in-memory ``BenchOneCaseResult`` (full
precision), not by re-reading the rounded ``result.jsonl``.

KV-history approximation caveat (direction-safe, not a blocker): a non-spec
decode at batch N reads N independent KV histories, but a real verify step over
N tokens reads only ``N / (gamma + 1)`` histories (the gamma+1 query tokens of
one request share a history). So the non-spec proxy over-reads history ->
over-estimates per-step latency -> under-estimates steps_per_sec -> the
scheduler is slightly conservative (verify window opens a touch short, never
over-extends). The table is also implicitly conditioned on the context regime
swept by ``--input-len`` / ``--output-len``; pick them near the target workload.

CLI framework note: the whole sglang bench ecosystem (including the wrapped
``one_batch_server``) registers its arguments with argparse via
``ServerArgs.add_cli_args`` / ``BenchArgs.add_cli_args``. To reuse that
registration verbatim rather than hand-copying dozens of server/bench options,
this profiler deliberately uses argparse here instead of the usual typer
preference; only ``--out`` / ``--max-batch-tokens`` / ``--repeats`` are added.

# Usage (connect to a running non-spec server)
python -m sglang.benchmark.dspark_sps_profiler \
    --model None --base-url http://localhost:30000 \
    --batch-size 1 2 4 8 16 32 64 128 \
    --input-len 512 --output-len 1024 \
    --out ~/main/artifacts/sglang/dspark_sps_table.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Optional

from sglang.benchmark.one_batch_server import (
    BenchArgs,
    BenchOneCaseResult,
    run_benchmark_internal,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    load_sps_table_from_path,
    profile_sps_table,
)

logger = logging.getLogger(__name__)

CONVERSION_FORMULA = (
    "batch_tokens = batch_size; "
    "steps_per_sec = output_throughput / batch_size = 1000 / ITL_ms; "
    "ITL_ms = 1000 * batch_size / output_throughput"
)


@dataclasses.dataclass
class ProfilerArgs:
    out: str = "~/main/artifacts/sglang/dspark_sps_table.json"
    max_batch_tokens: Optional[int] = None
    repeats: int = 1
    self_check: bool = True

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--out",
            type=str,
            default=ProfilerArgs.out,
            help="Output JSON path for the SpsCostTable. Defaults under "
            "~/main/artifacts/sglang/ (raw experiment material, not checked in). "
            "The raw bench result.jsonl is written next to it, and a "
            "<out>.manifest.json ties the two together.",
        )
        parser.add_argument(
            "--max-batch-tokens",
            type=int,
            default=ProfilerArgs.max_batch_tokens,
            help="Override the table's max_batch_tokens (clamp bound). Defaults to "
            "the largest swept batch size. Set this to production "
            "max_running_requests * (gamma + 1) so lookups above the largest "
            "sample clamp to the last probe.",
        )
        parser.add_argument(
            "--repeats",
            type=int,
            default=ProfilerArgs.repeats,
            help="Times to repeat the whole batch-size sweep; per batch_tokens the "
            "median steps_per_sec is taken. Default 1 (a long output_len already "
            "averages many decode steps into a stable ITL).",
        )
        parser.add_argument(
            "--no-self-check",
            dest="self_check",
            action="store_false",
            help="Skip the read-back + lookup self-check of the written table.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> ProfilerArgs:
        return cls(
            out=args.out,
            max_batch_tokens=args.max_batch_tokens,
            repeats=args.repeats,
            self_check=args.self_check,
        )


@dataclasses.dataclass
class DerivedRow:
    batch_size: int
    batch_tokens: int
    output_throughput: float
    itl_ms: float
    steps_per_sec: float


def derive_row(result: BenchOneCaseResult) -> Optional[DerivedRow]:
    if result.batch_size < 1 or result.output_throughput <= 0:
        logger.warning(
            "Skipping degenerate bench case: batch_size=%s, output_throughput=%s "
            "(need batch_size >= 1 and output_throughput > 0).",
            result.batch_size,
            result.output_throughput,
        )
        return None

    batch_tokens = result.batch_size
    steps_per_sec = result.output_throughput / result.batch_size
    itl_ms = 1000.0 * result.batch_size / result.output_throughput
    return DerivedRow(
        batch_size=result.batch_size,
        batch_tokens=batch_tokens,
        output_throughput=result.output_throughput,
        itl_ms=itl_ms,
        steps_per_sec=steps_per_sec,
    )


def median_probes_per_batch_tokens(
    rows: list[DerivedRow],
) -> list[tuple[int, float]]:
    by_batch_tokens: dict[int, list[float]] = {}
    for row in rows:
        by_batch_tokens.setdefault(row.batch_tokens, []).append(row.steps_per_sec)
    return [
        (batch_tokens, statistics.median(steps))
        for batch_tokens, steps in sorted(by_batch_tokens.items())
    ]


def run_self_check(*, out_path: Path) -> None:
    table = load_sps_table_from_path(str(out_path))
    if len(table.sample_batch_tokens) != len(table.sample_steps_per_sec):
        raise RuntimeError("Reloaded table has mismatched probe / SPS lengths.")

    # Every stored probe must look up to a positive SPS, and looking up below the
    # smallest probe must clamp to it (the loader's bisect-floor + clamp
    # contract). The per-step throughput should not increase as the batch grows
    # (a larger forward costs at least as much), so the table is expected to be
    # monotone non-increasing across probes -- a >10% rise only warns, since the
    # bench ITL can jitter, but it should not happen with a stable sweep.
    previous_sps: Optional[float] = None
    for batch_tokens, _ in zip(table.sample_batch_tokens, table.sample_steps_per_sec):
        looked_up = table.lookup(batch_tokens)
        if looked_up <= 0:
            raise RuntimeError(
                f"Reloaded table lookup at batch_tokens={batch_tokens} returned "
                f"non-positive SPS {looked_up}."
            )
        if previous_sps is not None and looked_up > previous_sps * 1.10:
            logger.warning(
                "Non-monotone SPS across probes: batch_tokens=%s SPS=%.3f rose "
                "above the previous probe's SPS=%.3f by >10%%; verify the server "
                "is at steady state (warmup / output_len long enough).",
                batch_tokens,
                looked_up,
                previous_sps,
            )
        previous_sps = looked_up

    below_floor = table.lookup(table.sample_batch_tokens[0] - 1)
    if below_floor != table.sample_steps_per_sec[0]:
        raise RuntimeError(
            "Reloaded table lookup below the smallest probe did not clamp to the "
            f"first SPS ({below_floor} != {table.sample_steps_per_sec[0]})."
        )
    logger.info(
        "Self-check passed: reloaded %s probes, all lookups positive and "
        "below-floor clamp holds.",
        len(table.sample_batch_tokens),
    )


def write_manifest(
    *,
    manifest_path: Path,
    result_path: Path,
    server_args: ServerArgs,
    bench_args: BenchArgs,
    profiler_args: ProfilerArgs,
    rows: list[DerivedRow],
) -> None:
    manifest = {
        "base_url": bench_args.base_url,
        "input_len": list(bench_args.input_len),
        "output_len": list(bench_args.output_len),
        "batch_size_sweep": list(bench_args.batch_size),
        "repeats": profiler_args.repeats,
        "tp_size": server_args.tp_size,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "conversion_formula": CONVERSION_FORMULA,
        "kv_history_caveat": (
            "Non-spec decode at batch N reads N KV histories; a real verify step "
            "over N tokens reads N/(gamma+1). The proxy over-reads history -> "
            "over-estimates latency -> under-estimates steps_per_sec -> scheduler "
            "is conservative (never over-extends). Direction-safe."
        ),
        "result_jsonl": result_path.name,
        "derived_rows": [
            {
                "batch_size": row.batch_size,
                "batch_tokens": row.batch_tokens,
                "output_throughput": row.output_throughput,
                "itl_ms": row.itl_ms,
                "steps_per_sec": row.steps_per_sec,
            }
            for row in rows
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def profile(
    server_args: ServerArgs, bench_args: BenchArgs, profiler_args: ProfilerArgs
) -> None:
    out_path = Path(profiler_args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = out_path.with_name(out_path.stem + ".result.jsonl")
    manifest_path = out_path.with_name(out_path.name + ".manifest.json")

    # The bench appends each case to result_filename; start from a clean file so a
    # rerun does not accumulate stale rows across invocations.
    if result_path.exists():
        result_path.unlink()

    bench_args = bench_args_for_profiling(
        bench_args=bench_args, result_path=result_path
    )

    rows: list[DerivedRow] = []
    for repeat in range(max(1, profiler_args.repeats)):
        results, _server_info = run_benchmark_internal(server_args, bench_args)
        logger.info(
            "Sweep repeat %s/%s returned %s bench cases.",
            repeat + 1,
            profiler_args.repeats,
            len(results),
        )
        for result in results:
            row = derive_row(result)
            if row is not None:
                rows.append(row)

    if not rows:
        raise RuntimeError(
            "No usable bench cases (all had output_throughput <= 0 or were "
            "skipped by the server's capacity guards); check the batch-size "
            "sweep against the server's max_running_requests / KV capacity."
        )

    probes = median_probes_per_batch_tokens(rows)
    table = profile_sps_table(
        probes=probes,
        max_batch_tokens=profiler_args.max_batch_tokens,
    )

    out_path.write_text(table.to_json(), encoding="utf-8")
    logger.info(
        "Wrote SpsCostTable (%s probes) to %s",
        len(table.sample_batch_tokens),
        out_path,
    )

    write_manifest(
        manifest_path=manifest_path,
        result_path=result_path,
        server_args=server_args,
        bench_args=bench_args,
        profiler_args=profiler_args,
        rows=rows,
    )
    logger.info("Wrote manifest to %s", manifest_path)

    if profiler_args.self_check:
        run_self_check(out_path=out_path)


def bench_args_for_profiling(*, bench_args: BenchArgs, result_path: Path) -> BenchArgs:
    # Reuse the user's bench sweep (base_url / batch_size / input_len / output_len)
    # but force the dump path and the modes the profiler relies on: warmup on (so
    # ITL is steady-state), report on (the conversion mirrors its ITL formula),
    # and the raw rows dumped next to the table.
    return dataclasses.replace(
        bench_args,
        result_filename=str(result_path),
        skip_warmup=False,
        show_report=True,
    )


def main(
    server_args: ServerArgs, bench_args: BenchArgs, profiler_args: ProfilerArgs
) -> None:
    if not bench_args.base_url:
        raise ValueError(
            "dspark_sps_profiler connects to an already-running non-spec server; "
            "pass --model None --base-url <url> (it never launches a server)."
        )
    if server_args.speculative_algorithm is not None:
        raise ValueError(
            "Profile against a NON-speculative server: drop --speculative-* flags "
            f"(got --speculative-algorithm {server_args.speculative_algorithm!r}). "
            "The SPS table is built from the non-spec decode-step cost."
        )

    profile(server_args, bench_args, profiler_args)


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    ProfilerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    profiler_args = ProfilerArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    main(server_args, bench_args, profiler_args)


if __name__ == "__main__":
    cli_main()
