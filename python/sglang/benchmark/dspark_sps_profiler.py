"""
Profile a DSpark SPS cost table (JSON) from an already-running non-spec server.

This profiler connects to a plain server the user launched separately (NO
speculative flags) and sweeps decode batch sizes, reading the steady-state
average inter-token latency (ITL) of each batch. It reuses ``one_batch_server``'s
single-case measurement primitive (``run_one_case``) but owns a small,
purpose-built CLI: it never launches a server, so it deliberately does NOT expose
the full ``ServerArgs`` / ``BenchArgs`` surface -- only a handful of its own args.
The one non-spec server fact it needs (tokenizer path, tp_size, running/KV
capacity, and the assert that the server is non-spec) is read once over HTTP from
``/server_info``.

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

# Usage (connect to a running non-spec server)
python -m sglang.benchmark.dspark_sps_profiler \
    --base-url http://localhost:30000 \
    --batch-size 1 2 4 8 16 32 64 128 \
    --input-len 512 --output-len 1024 \
    --out ~/main/artifacts/sglang/dspark_sps_table.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Optional

import msgspec
import requests

from sglang.benchmark.one_batch_server import (
    DEFAULT_TIMEOUT,
    BenchOneCaseResult,
    run_one_case,
    should_skip_due_to_max_running_requests,
    should_skip_due_to_token_capacity,
)
from sglang.benchmark.utils import get_tokenizer
from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    SpsCostTable,
    load_sps_table_from_path,
    profile_sps_table,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT = "~/main/artifacts/sglang/dspark_sps_table.json"
DEFAULT_BATCH_SIZE = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_INPUT_LEN = [512]
DEFAULT_OUTPUT_LEN = [1024]
WARMUP_INPUT_LEN = 1024
WARMUP_OUTPUT_LEN = 16
PROFILE_SEED = 42

CONVERSION_FORMULA = (
    "batch_tokens = batch_size; "
    "steps_per_sec = output_throughput / batch_size = 1000 / ITL_ms; "
    "ITL_ms = 1000 * batch_size / output_throughput"
)
KV_HISTORY_CAVEAT = (
    "Non-spec decode at batch N reads N KV histories; a real verify step over N "
    "tokens reads N/(gamma+1). The proxy over-reads history -> over-estimates "
    "latency -> under-estimates steps_per_sec -> scheduler is conservative "
    "(never over-extends). Direction-safe."
)


class ServerContext(msgspec.Struct, frozen=True):
    base_url: str
    tokenizer_path: str
    tp_size: int
    skip_max_running_requests_threshold: float
    skip_token_capacity_threshold: float


class DerivedRow(msgspec.Struct, frozen=True):
    batch_size: int
    batch_tokens: int
    output_throughput: float
    itl_ms: float
    steps_per_sec: float


class ProfileOutcome(msgspec.Struct, frozen=True):
    table: SpsCostTable
    rows: list[DerivedRow]


def fetch_server_context(
    *, base_url: str, local_tokenizer_path: Optional[str]
) -> ServerContext:
    response = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    info = response.json()

    speculative_algorithm = info.get("speculative_algorithm")
    if speculative_algorithm is not None:
        raise ValueError(
            f"Profile against a NON-speculative server: {base_url} reports "
            f"speculative_algorithm={speculative_algorithm!r}. The SPS table is "
            "built from the non-spec decode-step cost; relaunch the target "
            "without --speculative-* flags."
        )

    tokenizer_path = local_tokenizer_path or info.get("tokenizer_path")
    if not tokenizer_path:
        raise RuntimeError(
            "Could not resolve a tokenizer path from /server_info; pass "
            "--local-tokenizer-path explicitly."
        )

    internal_states = info.get("internal_states") or [{}]
    internal_state = internal_states[0]
    dp_size = internal_state.get("dp_size") or 1
    max_running_per_dp = internal_state.get("effective_max_running_requests_per_dp", -1)
    if max_running_per_dp and max_running_per_dp > 0:
        skip_max_running = float(max_running_per_dp * dp_size)
    else:
        logger.warning(
            "Server did not report effective_max_running_requests_per_dp (%s); "
            "not clamping the batch-size sweep against the running cap.",
            max_running_per_dp,
        )
        skip_max_running = float("inf")

    skip_token_capacity = 0.0
    for state in internal_states:
        skip_token_capacity += state.get("memory_usage", {}).get(
            "token_capacity", 1_000_000_000
        )

    return ServerContext(
        base_url=base_url,
        tokenizer_path=tokenizer_path,
        tp_size=int(info.get("tp_size", 1) or 1),
        skip_max_running_requests_threshold=skip_max_running,
        skip_token_capacity_threshold=skip_token_capacity,
    )


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


def build_sps_table(
    *, results: list[BenchOneCaseResult], max_batch_tokens: Optional[int]
) -> ProfileOutcome:
    rows: list[DerivedRow] = []
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
    table = profile_sps_table(probes=probes, max_batch_tokens=max_batch_tokens)
    return ProfileOutcome(table=table, rows=rows)


def _should_skip_case(
    *, context: ServerContext, batch_size: int, input_len: int, output_len: int
) -> bool:
    return should_skip_due_to_max_running_requests(
        batch_size, context.skip_max_running_requests_threshold
    ) or should_skip_due_to_token_capacity(
        batch_size, input_len, output_len, context.skip_token_capacity_threshold
    )


def run_bench_cases(
    *,
    context: ServerContext,
    tokenizer: object,
    batch_sizes: list[int],
    input_lens: list[int],
    output_lens: list[int],
    repeats: int,
    result_path: Path,
) -> list[BenchOneCaseResult]:
    # Warm up once per unique batch size (hot CUDA graphs / caches) so the first
    # measured case already reports steady-state ITL. Warmup rows are not dumped.
    for batch_size in sorted(set(batch_sizes)):
        if _should_skip_case(
            context=context,
            batch_size=batch_size,
            input_len=WARMUP_INPUT_LEN,
            output_len=WARMUP_OUTPUT_LEN,
        ):
            continue
        run_one_case(
            context.base_url,
            batch_size=batch_size,
            input_len=WARMUP_INPUT_LEN,
            output_len=WARMUP_OUTPUT_LEN,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
        )

    results: list[BenchOneCaseResult] = []
    for repeat in range(max(1, repeats)):
        for batch_size, input_len, output_len in itertools.product(
            batch_sizes, input_lens, output_lens
        ):
            if _should_skip_case(
                context=context,
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
            ):
                continue
            results.append(
                run_one_case(
                    context.base_url,
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
                    run_name="dspark_sps",
                    result_filename=str(result_path),
                    tokenizer=tokenizer,
                )
            )
        logger.info("Completed sweep repeat %s/%s.", repeat + 1, max(1, repeats))
    return results


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
    context: ServerContext,
    input_lens: list[int],
    output_lens: list[int],
    batch_sizes: list[int],
    repeats: int,
    rows: list[DerivedRow],
) -> None:
    manifest = {
        "base_url": context.base_url,
        "input_len": input_lens,
        "output_len": output_lens,
        "batch_size_sweep": batch_sizes,
        "repeats": repeats,
        "tp_size": context.tp_size,
        "seed": PROFILE_SEED,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "conversion_formula": CONVERSION_FORMULA,
        "kv_history_caveat": KV_HISTORY_CAVEAT,
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
    *,
    base_url: str,
    batch_size: list[int],
    input_len: list[int],
    output_len: list[int],
    out: str,
    max_batch_tokens: Optional[int],
    repeats: int,
    self_check: bool,
    local_tokenizer_path: Optional[str],
) -> None:
    if not base_url:
        raise ValueError(
            "dspark_sps_profiler connects to an already-running non-spec server; "
            "pass --base-url <url> (it never launches a server)."
        )

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = out_path.with_name(out_path.stem + ".result.jsonl")
    manifest_path = out_path.with_name(out_path.name + ".manifest.json")

    # The bench appends each case to result_filename; start from a clean file so a
    # rerun does not accumulate stale rows across invocations.
    if result_path.exists():
        result_path.unlink()

    context = fetch_server_context(
        base_url=base_url, local_tokenizer_path=local_tokenizer_path
    )
    tokenizer = get_tokenizer(context.tokenizer_path)

    results = run_bench_cases(
        context=context,
        tokenizer=tokenizer,
        batch_sizes=batch_size,
        input_lens=input_len,
        output_lens=output_len,
        repeats=repeats,
        result_path=result_path,
    )
    outcome = build_sps_table(results=results, max_batch_tokens=max_batch_tokens)

    out_path.write_text(outcome.table.to_json(), encoding="utf-8")
    logger.info(
        "Wrote SpsCostTable (%s probes) to %s",
        len(outcome.table.sample_batch_tokens),
        out_path,
    )

    write_manifest(
        manifest_path=manifest_path,
        result_path=result_path,
        context=context,
        input_lens=input_len,
        output_lens=output_len,
        batch_sizes=batch_size,
        repeats=repeats,
        rows=outcome.rows,
    )
    logger.info("Wrote manifest to %s", manifest_path)

    if self_check:
        run_self_check(out_path=out_path)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a DSpark SPS cost table (JSON) from an already-running "
            "non-spec server."
        )
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base URL of an already-running NON-spec server, e.g. "
        "http://localhost:30000. The profiler never launches a server.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZE,
        help="Decode batch sizes to sweep. Each maps to a batch_tokens probe in "
        "the SPS table.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        nargs="+",
        default=DEFAULT_INPUT_LEN,
        help="Prompt length(s) per request. The table is implicitly conditioned "
        "on this context regime; pick it near the target workload.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        nargs="+",
        default=DEFAULT_OUTPUT_LEN,
        help="Decode length(s) per request. Long enough to average many "
        "steady-state decode steps into a stable ITL.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help="Output JSON path for the SpsCostTable. Defaults under "
        "~/main/artifacts/sglang/ (raw experiment material, not checked in). The "
        "raw bench result.jsonl is written next to it, and a <out>.manifest.json "
        "ties the two together.",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help="Override the table's max_batch_tokens (clamp bound). Defaults to "
        "the largest swept batch size. Set this to production "
        "max_running_requests * (gamma + 1) so lookups above the largest sample "
        "clamp to the last probe.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
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
    parser.add_argument(
        "--local-tokenizer-path",
        type=str,
        default=None,
        help="Override the tokenizer path (defaults to the one reported by "
        "/server_info).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Python logging level for the profiler.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(message)s",
    )
    random.seed(PROFILE_SEED)

    profile(
        base_url=args.base_url,
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
        out=args.out,
        max_batch_tokens=args.max_batch_tokens,
        repeats=args.repeats,
        self_check=args.self_check,
        local_tokenizer_path=args.local_tokenizer_path,
    )


if __name__ == "__main__":
    cli_main()
