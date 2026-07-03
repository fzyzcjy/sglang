from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Optional

import msgspec
import requests

from sglang.benchmark.one_batch_server import (
    DEFAULT_TIMEOUT,
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
DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_INPUT_LEN = 16
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TARGET_STEADY_STEPS = 128
DEFAULT_MIN_STEADY_STEPS = 32
DEFAULT_ROUND_TIMEOUT_SECONDS = 300.0
ROUND_WARMUP_STEPS = 8
ROUND_STEP_SLACK = 64
WARMUP_ROUND_STEADY_STEPS = 16
POLL_INTERVAL_SECONDS = 2.0
LOAD_JOIN_TIMEOUT_SECONDS = 60.0
MATCH_FRACTION_WARN = 0.9
MATCH_FRACTION_ERROR = 0.5
PROFILE_SEED = 42
REQUIRED_SIMULATE_ACC_LEN = 1.0
RANDOM_TOKEN_LOW = 1000
RANDOM_TOKEN_HIGH_MARGIN = 1000

STATIC_CONDITIONING_CAVEAT = (
    "Profiled with SGLANG_RAGGED_VERIFY_MODE=static: a verify step of B tokens "
    "comes from B/(gamma+1) requests, the fewest possible for that B. A "
    "compact-mode step with the same B usually spans more requests and reads "
    "more KV history, so the table slightly over-estimates steps_per_sec and "
    "the scheduler may admit slightly more than optimal. Much smaller bias "
    "than the retired non-spec decode proxy, and in the opposite direction."
)
CONVERSION_FORMULA = (
    "batch_tokens = num_running_reqs_per_rank * verify_num_draft_tokens; "
    "steps_per_sec = 1 / median(server-side step_time over aligned steady steps)"
)


class SpsRow(msgspec.Struct, frozen=True):
    forward_ct: int
    num_running_reqs: int
    num_verify_tokens: int
    step_time: float


class RecordSource(msgspec.Struct, frozen=True):
    name: str
    payload_key: str
    enable_hint: str
    step_time_ms: bool


SPS_RECORD_SOURCE = RecordSource(
    name="sps",
    payload_key="dspark_sps_record",
    enable_hint="SGLANG_DSPARK_ENABLE_SPS_RECORD=1 (and SGLANG_RAGGED_VERIFY_MODE=static)",
    step_time_ms=False,
)
INFO_RECORD_SOURCE = RecordSource(
    name="info",
    payload_key="dspark_info_record",
    enable_hint=(
        "SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time "
        "(and SGLANG_RAGGED_VERIFY_MODE=static)"
    ),
    step_time_ms=True,
)
RECORD_SOURCES = {
    SPS_RECORD_SOURCE.name: SPS_RECORD_SOURCE,
    INFO_RECORD_SOURCE.name: INFO_RECORD_SOURCE,
}


class ServerContext(msgspec.Struct, frozen=True):
    base_url: str
    tokenizer_path: str
    tp_size: int
    dp_size: int
    verify_num_draft_tokens: int
    simulate_acc_len: float
    cuda_graph_max_bs: Optional[int]
    skip_max_running_requests_threshold: float
    skip_token_capacity_threshold: float
    record_source: RecordSource


class RoundSettings(msgspec.Struct, frozen=True):
    input_len: int
    temperature: float
    target_steady_steps: int
    min_steady_steps: int
    round_timeout_seconds: float
    ramp_token_slack: int = 0


class LoadInfo(msgspec.Struct, frozen=True):
    num_requests: int
    max_new_tokens: int
    wall_seconds: float
    reached_target: bool


class RoundOutcome(msgspec.Struct, frozen=True):
    batch_size: int
    batch_size_per_rank: int
    batch_tokens: int
    steps_per_sec: float
    num_steady_steps: int
    match_fraction: float
    per_rank_median_step_time: list[float]
    rank_rows: list[list[SpsRow]]
    load_info: LoadInfo


def profile(
    *,
    base_url: str,
    batch_sizes: list[int],
    settings: RoundSettings,
    out: str,
    max_batch_tokens: Optional[int],
    repeats: int,
    self_check: bool,
    local_tokenizer_path: Optional[str],
    recorder_source: str,
) -> None:
    if not base_url:
        raise ValueError(
            "dspark_sps_profiler connects to an already-running DSpark server "
            "(SGLANG_RAGGED_VERIFY_MODE=static, SGLANG_DSPARK_ENABLE_SPS_RECORD=1); "
            "pass --base-url <url> (it never launches a server)."
        )

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_path = out_path.with_name(out_path.stem + ".records.jsonl")
    rounds_path = out_path.with_name(out_path.stem + ".rounds.jsonl")
    manifest_path = out_path.with_name(out_path.name + ".manifest.json")
    for path in (records_path, rounds_path):
        if path.exists():
            path.unlink()

    context = fetch_server_context(
        base_url=base_url,
        local_tokenizer_path=local_tokenizer_path,
        record_source=RECORD_SOURCES[recorder_source],
    )
    vocab_size = len(get_tokenizer(context.tokenizer_path))
    batch_sizes = sorted(set(batch_sizes))
    validate_sweep_against_server(context=context, batch_sizes=batch_sizes)
    rng = random.Random(PROFILE_SEED)

    run_warmup_round(
        context=context,
        vocab_size=vocab_size,
        batch_sizes=batch_sizes,
        settings=settings,
        rng=rng,
    )

    rounds: list[RoundOutcome] = []
    for repeat in range(max(1, repeats)):
        for batch_size_per_rank in batch_sizes:
            outcome = run_one_round(
                context=context,
                vocab_size=vocab_size,
                batch_size_per_rank=batch_size_per_rank,
                settings=settings,
                rng=rng,
            )
            if outcome is None:
                continue
            logger.info(
                "Round bs=%s (per-rank %s, batch_tokens=%s) repeat=%s/%s: "
                "steps_per_sec=%.3f over %s steady steps (match_fraction=%.2f, "
                "wall=%.1fs, per-rank median step_time=%s)",
                outcome.batch_size,
                outcome.batch_size_per_rank,
                outcome.batch_tokens,
                repeat + 1,
                max(1, repeats),
                outcome.steps_per_sec,
                outcome.num_steady_steps,
                outcome.match_fraction,
                outcome.load_info.wall_seconds,
                ["%.4f" % value for value in outcome.per_rank_median_step_time],
            )
            append_round_files(
                records_path=records_path,
                rounds_path=rounds_path,
                outcome=outcome,
                repeat=repeat,
            )
            rounds.append(outcome)

    if not rounds:
        raise RuntimeError(
            "No usable rounds (all were skipped by capacity guards or failed); "
            "check the batch-size sweep against the server's "
            "max_running_requests / KV capacity."
        )

    table = build_table_from_rounds(rounds=rounds, max_batch_tokens=max_batch_tokens)
    out_path.write_text(table.to_json(), encoding="utf-8")
    logger.info(
        "Wrote SpsCostTable (%s probes) to %s",
        len(table.sample_batch_tokens),
        out_path,
    )

    write_manifest(
        manifest_path=manifest_path,
        records_path=records_path,
        rounds_path=rounds_path,
        context=context,
        batch_sizes=batch_sizes,
        settings=settings,
        repeats=repeats,
        rounds=rounds,
    )
    logger.info("Wrote manifest to %s", manifest_path)

    if self_check:
        run_self_check(out_path=out_path)


def fetch_server_context(
    *,
    base_url: str,
    local_tokenizer_path: Optional[str],
    record_source: RecordSource,
) -> ServerContext:
    response = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    info = response.json()

    speculative_algorithm = info.get("speculative_algorithm")
    if speculative_algorithm != "DSPARK":
        raise ValueError(
            f"Profile against a DSpark server: {base_url} reports "
            f"speculative_algorithm={speculative_algorithm!r}. The SPS table is "
            "measured from real static-mode DSpark verify steps; relaunch with "
            "--speculative-algorithm DSPARK and SGLANG_RAGGED_VERIFY_MODE=static."
        )
    if info.get("disable_cuda_graph"):
        raise ValueError(
            "The server runs with --disable-cuda-graph; an SPS table measured "
            "without cuda graphs is uselessly slow. Relaunch with cuda graphs "
            "enabled."
        )

    internal_states = info.get("internal_states") or []
    if not internal_states:
        raise RuntimeError(f"{base_url}/server_info returned no internal_states.")
    sps_payloads = [state.get(record_source.payload_key) for state in internal_states]
    for rank_index, payload in enumerate(sps_payloads):
        if payload is None:
            raise ValueError(
                f"DP rank {rank_index} reports no {record_source.payload_key}; "
                f"launch the server with {record_source.enable_hint}."
            )
        if payload.get("mode") != "static":
            raise ValueError(
                f"{record_source.payload_key}.mode must be 'static', got "
                f"{payload.get('mode')!r} on DP rank {rank_index}."
            )
        if record_source is INFO_RECORD_SOURCE:
            components = payload.get("components") or []
            missing = {"core", "step_cpu_time"} - set(components)
            if missing:
                raise ValueError(
                    f"DP rank {rank_index} {record_source.payload_key} is missing "
                    f"component(s) {sorted(missing)}; launch with "
                    f"{record_source.enable_hint}."
                )
        if payload.get("simulate_acc_len") != REQUIRED_SIMULATE_ACC_LEN:
            raise ValueError(
                f"DP rank {rank_index} reports simulate_acc_len="
                f"{payload.get('simulate_acc_len')!r}, but SPS profiling "
                f"requires exactly SGLANG_SIMULATE_ACC_LEN="
                f"{REQUIRED_SIMULATE_ACC_LEN} (spec fully ineffective: every "
                "step advances every request by exactly the bonus token, so "
                "the per-step KV conditioning is deterministic instead of "
                "drifting with the model's accept behavior)."
            )
    verify_num_draft_tokens = {
        int(payload["verify_num_draft_tokens"]) for payload in sps_payloads
    }
    simulate_acc_lens = {float(payload["simulate_acc_len"]) for payload in sps_payloads}
    if len(simulate_acc_lens) != 1:
        raise RuntimeError(
            f"DP ranks disagree on simulate_acc_len: {sorted(simulate_acc_lens)}."
        )
    if len(verify_num_draft_tokens) != 1:
        raise RuntimeError(
            "DP ranks disagree on verify_num_draft_tokens: "
            f"{sorted(verify_num_draft_tokens)}."
        )

    tokenizer_path = local_tokenizer_path or info.get("tokenizer_path")
    if not tokenizer_path:
        raise RuntimeError(
            "Could not resolve a tokenizer path from /server_info; pass "
            "--local-tokenizer-path explicitly."
        )

    internal_state = internal_states[0]
    dp_size = int(internal_state.get("dp_size") or 1)
    cuda_graph_max_bs = resolve_cuda_graph_max_bs(internal_state=internal_state)
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
        dp_size=dp_size,
        verify_num_draft_tokens=verify_num_draft_tokens.pop(),
        simulate_acc_len=simulate_acc_lens.pop(),
        cuda_graph_max_bs=cuda_graph_max_bs,
        skip_max_running_requests_threshold=skip_max_running,
        skip_token_capacity_threshold=skip_token_capacity,
        record_source=record_source,
    )


def resolve_cuda_graph_max_bs(*, internal_state: dict) -> Optional[int]:
    cuda_graph_config = internal_state.get("cuda_graph_config")
    if not isinstance(cuda_graph_config, dict):
        return None
    decode_config = cuda_graph_config.get("decode")
    if not isinstance(decode_config, dict):
        return None
    captured_bs = decode_config.get("bs")
    if isinstance(captured_bs, list) and captured_bs:
        return int(max(captured_bs))
    max_bs = decode_config.get("max_bs")
    if max_bs is not None:
        return int(max_bs)
    return None


def validate_sweep_against_server(
    *, context: ServerContext, batch_sizes: list[int]
) -> None:
    if context.cuda_graph_max_bs is None:
        logger.warning(
            "Could not resolve the server's captured cuda-graph max batch size; "
            "not clamping the sweep against it."
        )
        return
    max_per_rank = max(batch_sizes)
    if max_per_rank > context.cuda_graph_max_bs:
        raise ValueError(
            f"The sweep reaches {max_per_rank} running requests per DP rank but "
            "the server captured decode cuda graphs only up to bs="
            f"{context.cuda_graph_max_bs}; steps beyond it run eager and poison "
            "the table. Relaunch the server with a larger --cuda-graph-max-bs "
            "or shrink --max-batch-size."
        )


def build_request_count_sweep(max_num_reqs: int) -> list[int]:
    if max_num_reqs < 1:
        raise ValueError(f"max_num_reqs must be >= 1, got {max_num_reqs}.")
    raw = [
        1,
        2,
        4,
        8,
        *range(16, 64, 8),
        *range(64, 128, 16),
        *range(128, 256, 32),
        *range(256, max_num_reqs + 1, 64),
    ]
    sweep = sorted({value for value in raw if 1 <= value <= max_num_reqs})
    if sweep[-1] != max_num_reqs:
        sweep.append(max_num_reqs)
    return sweep


def round_max_new_tokens(*, settings: RoundSettings, context: ServerContext) -> int:
    # Under the required simulate_acc_len == 1.0 every request advances
    # exactly ONE token per verify step (bonus only, zero correct drafts), so
    # the per-request token budget for N steps is N, not N * (gamma+1).
    # Multiplying by the verify window made requests outlive the target by
    # (gamma+1)x, so rounds that missed the poll-based abort ran ~6x longer
    # than designed. Natural exhaustion now bounds every round at the step
    # budget even if the abort never fires.
    commit_tokens_per_step = (
        max(
            0,
            min(
                round(context.simulate_acc_len - 1), context.verify_num_draft_tokens - 1
            ),
        )
        + 1
    )
    # Long inputs need a ramp allowance on top of the step budget: prefilling
    # the whole batch can take minutes, and requests that finish their step
    # budget before the last request enters decode make full-batch alignment
    # unreachable (observed: input_len 8192, bs 384 -> 0 aligned steps).
    total_steps = ROUND_WARMUP_STEPS + settings.target_steady_steps + ROUND_STEP_SLACK
    return total_steps * commit_tokens_per_step + settings.ramp_token_slack


def run_warmup_round(
    *,
    context: ServerContext,
    vocab_size: int,
    batch_sizes: list[int],
    settings: RoundSettings,
    rng: random.Random,
) -> None:
    warmup_settings = RoundSettings(
        input_len=settings.input_len,
        temperature=settings.temperature,
        target_steady_steps=WARMUP_ROUND_STEADY_STEPS,
        min_steady_steps=1,
        round_timeout_seconds=settings.round_timeout_seconds,
        ramp_token_slack=settings.ramp_token_slack,
    )
    try:
        run_one_round(
            context=context,
            vocab_size=vocab_size,
            batch_size_per_rank=min(8, max(batch_sizes)),
            settings=warmup_settings,
            rng=rng,
        )
    except Exception:
        logger.warning("Warmup round failed; continuing.", exc_info=True)


def run_one_round(
    *,
    context: ServerContext,
    vocab_size: int,
    batch_size_per_rank: int,
    settings: RoundSettings,
    rng: random.Random,
) -> Optional[RoundOutcome]:
    batch_size = batch_size_per_rank * context.dp_size
    max_new_tokens = round_max_new_tokens(settings=settings, context=context)
    if should_skip_due_to_max_running_requests(
        batch_size, context.skip_max_running_requests_threshold
    ) or should_skip_due_to_token_capacity(
        batch_size,
        settings.input_len,
        max_new_tokens,
        context.skip_token_capacity_threshold,
    ):
        return None

    flush_cache(base_url=context.base_url)
    watermarks = [
        max((row.forward_ct for row in rows), default=-1)
        for rows in fetch_rank_rows(
            base_url=context.base_url, record_source=context.record_source
        )
    ]

    start_time = time.monotonic()
    load_thread = start_load(
        base_url=context.base_url,
        num_requests=batch_size,
        input_len=settings.input_len,
        max_new_tokens=max_new_tokens,
        temperature=settings.temperature,
        vocab_size=vocab_size,
        rng=rng,
    )
    reached_target = wait_for_aligned_steps(
        context=context,
        watermarks=watermarks,
        batch_size_per_rank=batch_size_per_rank,
        target_aligned_steps=ROUND_WARMUP_STEPS + settings.target_steady_steps,
        timeout_seconds=settings.round_timeout_seconds,
    )
    abort_all_requests(base_url=context.base_url)
    load_thread.join(timeout=LOAD_JOIN_TIMEOUT_SECONDS)
    if load_thread.is_alive():
        logger.warning(
            "Load batch for bs=%s did not return within %.0fs after abort; "
            "continuing with the collected records.",
            batch_size,
            LOAD_JOIN_TIMEOUT_SECONDS,
        )
    wall_seconds = time.monotonic() - start_time
    if not reached_target:
        logger.warning(
            "Round bs=%s hit the %.0fs timeout before collecting %s aligned "
            "steps; proceeding with what was collected.",
            batch_size,
            settings.round_timeout_seconds,
            ROUND_WARMUP_STEPS + settings.target_steady_steps,
        )

    rank_rows = fetch_rank_rows(
        base_url=context.base_url, record_source=context.record_source
    )
    if len(rank_rows) != len(watermarks):
        raise RuntimeError(
            f"DP rank count changed mid-profile: {len(watermarks)} -> "
            f"{len(rank_rows)}."
        )
    new_rank_rows = [
        [row for row in rows if row.forward_ct > watermark]
        for rows, watermark in zip(rank_rows, watermarks)
    ]
    return postprocess_round(
        rank_rows=new_rank_rows,
        batch_size_per_rank=batch_size_per_rank,
        dp_size=context.dp_size,
        verify_num_draft_tokens=context.verify_num_draft_tokens,
        min_steady_steps=settings.min_steady_steps,
        load_info=LoadInfo(
            num_requests=batch_size,
            max_new_tokens=max_new_tokens,
            wall_seconds=round(wall_seconds, 3),
            reached_target=reached_target,
        ),
    )


def start_load(
    *,
    base_url: str,
    num_requests: int,
    input_len: int,
    max_new_tokens: int,
    temperature: float,
    vocab_size: int,
    rng: random.Random,
) -> threading.Thread:
    token_high = vocab_size - RANDOM_TOKEN_HIGH_MARGIN
    if token_high <= RANDOM_TOKEN_LOW:
        raise ValueError(f"vocab_size={vocab_size} too small for random prompts.")
    input_ids = [
        [rng.randrange(RANDOM_TOKEN_LOW, token_high) for _ in range(input_len)]
        for _ in range(num_requests)
    ]
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": False,
    }

    def _post() -> None:
        try:
            requests.post(base_url + "/generate", json=payload, timeout=DEFAULT_TIMEOUT)
        except Exception:
            logger.warning(
                "Load batch POST /generate failed (expected on abort for some "
                "server versions).",
                exc_info=True,
            )

    thread = threading.Thread(target=_post, daemon=True)
    thread.start()
    return thread


def wait_for_aligned_steps(
    *,
    context: ServerContext,
    watermarks: list[int],
    batch_size_per_rank: int,
    target_aligned_steps: int,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            rank_rows = fetch_rank_rows(
                base_url=context.base_url, record_source=context.record_source
            )
        except Exception:
            logger.warning("Polling /server_info failed; retrying.", exc_info=True)
            continue
        new_rank_rows = [
            [row for row in rows if row.forward_ct > watermark]
            for rows, watermark in zip(rank_rows, watermarks)
        ]
        if len(new_rank_rows) != len(watermarks):
            continue
        aligned = count_aligned_steps(
            rank_rows=new_rank_rows, batch_size_per_rank=batch_size_per_rank
        )
        logger.debug(
            "Aligned-step poll: %d/%d aligned steps", aligned, target_aligned_steps
        )
        if aligned >= target_aligned_steps:
            return True
    return False


def count_aligned_steps(
    *, rank_rows: list[list[SpsRow]], batch_size_per_rank: int
) -> int:
    if any(not rows for rows in rank_rows):
        return 0
    by_ct_per_rank = [{row.forward_ct: row for row in rows} for rows in rank_rows]
    common_cts = set(by_ct_per_rank[0])
    for by_ct in by_ct_per_rank[1:]:
        common_cts &= set(by_ct)
    return sum(
        1
        for ct in common_cts
        if all(
            by_ct[ct].num_running_reqs == batch_size_per_rank
            for by_ct in by_ct_per_rank
        )
    )


def abort_all_requests(*, base_url: str) -> None:
    response = requests.post(
        base_url + "/abort_request",
        json={"abort_all": True},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()


def flush_cache(*, base_url: str) -> None:
    try:
        requests.post(base_url + "/flush_cache", timeout=DEFAULT_TIMEOUT)
    except Exception:
        logger.warning("POST /flush_cache failed; continuing.", exc_info=True)


def fetch_rank_rows(
    *, base_url: str, record_source: RecordSource
) -> list[list[SpsRow]]:
    response = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    internal_states = response.json().get("internal_states") or []
    rank_rows: list[list[SpsRow]] = []
    for state in internal_states:
        payload = state.get(record_source.payload_key) or {}
        rows: list[SpsRow] = []
        for record in payload.get("records", []):
            step_time = _row_step_time(record=record, record_source=record_source)
            if step_time is None:
                continue
            rows.append(
                SpsRow(
                    forward_ct=int(record["forward_ct"]),
                    num_running_reqs=int(record["num_running_reqs"]),
                    num_verify_tokens=int(record["num_verify_tokens"]),
                    step_time=step_time,
                )
            )
        rank_rows.append(rows)
    return rank_rows


def _row_step_time(*, record: dict, record_source: RecordSource) -> Optional[float]:
    if not record_source.step_time_ms:
        return float(record["step_time"])
    value = record.get("step_cpu_ms")
    return None if value is None else float(value) / 1000.0


def postprocess_round(
    *,
    rank_rows: list[list[SpsRow]],
    batch_size_per_rank: int,
    dp_size: int,
    verify_num_draft_tokens: int,
    min_steady_steps: int,
    load_info: LoadInfo,
) -> RoundOutcome:
    batch_size = batch_size_per_rank * dp_size
    expected_tokens = batch_size_per_rank * verify_num_draft_tokens

    if len(rank_rows) != dp_size:
        raise RuntimeError(
            f"Expected records from {dp_size} DP ranks, got {len(rank_rows)}."
        )

    by_ct_per_rank: list[dict[int, SpsRow]] = []
    for rank_index, rows in enumerate(rank_rows):
        if not rows:
            raise RuntimeError(
                f"DP rank {rank_index} produced no new decode-step records this "
                "round; the load generator did not reach it (DP imbalance or "
                "the round was too short)."
            )
        by_ct_per_rank.append({row.forward_ct: row for row in rows})

    common_cts = set(by_ct_per_rank[0])
    for by_ct in by_ct_per_rank[1:]:
        common_cts &= set(by_ct)
    if not common_cts:
        raise RuntimeError(
            "DP ranks share no common forward_ct in this round; their step "
            "counters are misaligned, so per-step cross-rank checks are "
            "impossible. This breaks the uniformity assumption of the table."
        )

    aligned_cts: list[int] = []
    for ct in sorted(common_cts):
        rows_at_ct = [by_ct[ct] for by_ct in by_ct_per_rank]
        if all(row.num_running_reqs == batch_size_per_rank for row in rows_at_ct):
            for rank_index, row in enumerate(rows_at_ct):
                if row.num_verify_tokens < expected_tokens:
                    raise RuntimeError(
                        f"DP rank {rank_index} at forward_ct={ct} reports "
                        f"num_verify_tokens={row.num_verify_tokens}, expected at "
                        f"least {expected_tokens} (= {batch_size_per_rank} reqs x "
                        f"{verify_num_draft_tokens}); ranks are not running the "
                        "uniform static verify the table assumes. The recorded "
                        "count is the replayed graph tier, which may exceed the "
                        "candidate count when a bs is not an exact capture tier."
                    )
            aligned_cts.append(ct)

    if len(aligned_cts) < ROUND_WARMUP_STEPS + min_steady_steps:
        raise RuntimeError(
            f"Round bs={batch_size} never stabilized: only {len(aligned_cts)} "
            f"of {len(common_cts)} common decode steps had every rank at the "
            f"target {batch_size_per_rank} requests (need at least "
            f"{ROUND_WARMUP_STEPS + min_steady_steps}). Increase "
            "--round-timeout / --target-steady-steps, or inspect the raw "
            "records for retraction / DP imbalance."
        )

    window_cts = [
        ct for ct in sorted(common_cts) if aligned_cts[0] <= ct <= aligned_cts[-1]
    ]
    match_fraction = len(aligned_cts) / len(window_cts)
    if match_fraction < MATCH_FRACTION_ERROR:
        raise RuntimeError(
            f"Round bs={batch_size} is unstable mid-round: only "
            f"{match_fraction:.0%} of the {len(window_cts)} decode steps inside "
            "the steady window ran at the target per-rank batch (retraction or "
            "DP imbalance, not just ramp-in/drain). Inspect the raw records."
        )
    if match_fraction < MATCH_FRACTION_WARN:
        logger.warning(
            "Round bs=%s: %.0f%% of %s steady-window decode steps ran at the "
            "target per-rank batch; treat this probe with suspicion.",
            batch_size,
            match_fraction * 100.0,
            len(window_cts),
        )

    steady_cts = aligned_cts[ROUND_WARMUP_STEPS:]
    per_ct_step_times = [
        statistics.fmean(by_ct[ct].step_time for by_ct in by_ct_per_rank)
        for ct in steady_cts
    ]
    per_rank_median_step_time = [
        statistics.median(by_ct[ct].step_time for ct in steady_cts)
        for by_ct in by_ct_per_rank
    ]
    median_step_time = statistics.median(per_ct_step_times)

    return RoundOutcome(
        batch_size=batch_size,
        batch_size_per_rank=batch_size_per_rank,
        batch_tokens=expected_tokens,
        steps_per_sec=1.0 / median_step_time,
        num_steady_steps=len(steady_cts),
        match_fraction=match_fraction,
        per_rank_median_step_time=per_rank_median_step_time,
        rank_rows=rank_rows,
        load_info=load_info,
    )


def build_table_from_rounds(
    *, rounds: list[RoundOutcome], max_batch_tokens: Optional[int]
) -> SpsCostTable:
    by_batch_tokens: dict[int, list[float]] = {}
    for outcome in rounds:
        by_batch_tokens.setdefault(outcome.batch_tokens, []).append(
            outcome.steps_per_sec
        )
    probes = [
        (batch_tokens, statistics.median(values))
        for batch_tokens, values in sorted(by_batch_tokens.items())
    ]
    return profile_sps_table(probes=probes, max_batch_tokens=max_batch_tokens)


def round_summary_dict(*, outcome: RoundOutcome, repeat: int) -> dict:
    return {
        "repeat": repeat,
        "batch_size": outcome.batch_size,
        "batch_size_per_rank": outcome.batch_size_per_rank,
        "batch_tokens": outcome.batch_tokens,
        "steps_per_sec": outcome.steps_per_sec,
        "num_steady_steps": outcome.num_steady_steps,
        "match_fraction": outcome.match_fraction,
        "per_rank_median_step_time": outcome.per_rank_median_step_time,
        "load_info": msgspec.to_builtins(outcome.load_info),
    }


def append_round_files(
    *,
    records_path: Path,
    rounds_path: Path,
    outcome: RoundOutcome,
    repeat: int,
) -> None:
    with records_path.open("a", encoding="utf-8") as fout:
        for rank_index, rows in enumerate(outcome.rank_rows):
            for row in rows:
                fout.write(
                    json.dumps(
                        {
                            "repeat": repeat,
                            "batch_size": outcome.batch_size,
                            "batch_size_per_rank": outcome.batch_size_per_rank,
                            "dp_rank": rank_index,
                            "forward_ct": row.forward_ct,
                            "num_running_reqs": row.num_running_reqs,
                            "num_verify_tokens": row.num_verify_tokens,
                            "step_time": row.step_time,
                        }
                    )
                    + "\n"
                )
    with rounds_path.open("a", encoding="utf-8") as fout:
        fout.write(
            json.dumps(round_summary_dict(outcome=outcome, repeat=repeat)) + "\n"
        )


def write_manifest(
    *,
    manifest_path: Path,
    records_path: Path,
    rounds_path: Path,
    context: ServerContext,
    batch_sizes: list[int],
    settings: RoundSettings,
    repeats: int,
    rounds: list[RoundOutcome],
) -> None:
    manifest = {
        "base_url": context.base_url,
        "tp_size": context.tp_size,
        "dp_size": context.dp_size,
        "verify_num_draft_tokens": context.verify_num_draft_tokens,
        "simulate_acc_len": context.simulate_acc_len,
        "batch_size_per_rank_sweep": batch_sizes,
        "settings": msgspec.to_builtins(settings),
        "repeats": repeats,
        "seed": PROFILE_SEED,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "conversion_formula": CONVERSION_FORMULA,
        "static_conditioning_caveat": STATIC_CONDITIONING_CAVEAT,
        "records_jsonl": records_path.name,
        "rounds_jsonl": rounds_path.name,
        "round_summaries": [
            round_summary_dict(outcome=outcome, repeat=0) for outcome in rounds
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_self_check(*, out_path: Path) -> None:
    table = load_sps_table_from_path(str(out_path))
    if len(table.sample_batch_tokens) != len(table.sample_steps_per_sec):
        raise RuntimeError("Reloaded table has mismatched probe / SPS lengths.")

    previous_sps: Optional[float] = None
    for batch_tokens in table.sample_batch_tokens:
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
                "was at steady state (no co-tenants, steady clocks).",
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


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a DSpark SPS cost table (JSON) from an already-running "
            "DSpark server in static ragged-verify mode with "
            "SGLANG_DSPARK_ENABLE_SPS_RECORD=1."
        )
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base URL of the already-running DSpark static-mode server, e.g. "
        "http://localhost:30000. The profiler never launches a server.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=None,
        help="Explicit PER-DP-RANK running-request counts to sweep; the load "
        "generator sends value * dp_size requests so every rank (GPU group) "
        "sits at the given batch. Overrides --max-batch-size when given.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Upper bound of the auto-generated tapered PER-DP-RANK "
        "request-count sweep (used only when --batch-size is not given), so "
        "per-rank token coverage is identical for any dp_size.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=DEFAULT_INPUT_LEN,
        help="Prompt length per request. Short: the table is conditioned on the "
        "decode-heavy regime.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for the load requests; default 1.0 to hit "
        "the same accept/sampling kernels as real serving.",
    )
    parser.add_argument(
        "--target-steady-steps",
        type=int,
        default=DEFAULT_TARGET_STEADY_STEPS,
        help="Aligned decode steps to collect per round before aborting the "
        "load batch. The batch is held at exactly the target running-request "
        "count the whole time (no early-finish drain tail).",
    )
    parser.add_argument(
        "--min-steady-steps",
        type=int,
        default=DEFAULT_MIN_STEADY_STEPS,
        help="Reject a probe built from fewer aligned steady steps than this "
        "(a handful of steps gives a jittery point).",
    )
    parser.add_argument(
        "--round-timeout",
        type=float,
        default=DEFAULT_ROUND_TIMEOUT_SECONDS,
        help="Per-round wall-clock budget in seconds to collect the target "
        "steps before giving up and using what was collected.",
    )
    parser.add_argument(
        "--ramp-token-slack",
        type=int,
        default=0,
        help="Extra per-request tokens on top of the step budget so requests "
        "outlive the whole-batch prefill ramp. Required for long --input-len "
        "at high batch sizes, where the ramp exceeds the request lifetime and "
        "full-batch alignment becomes unreachable; size it as roughly "
        "ramp_seconds / step_time.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help="Output JSON path for the SpsCostTable. Raw per-step records land "
        "next to it as <stem>.records.jsonl, one line per round as "
        "<stem>.rounds.jsonl, and <out>.manifest.json ties everything "
        "together.",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help="Override the table's max_batch_tokens metadata (defaults to the "
        "largest probed batch_tokens). Advisory production ceiling; lookups "
        "above the largest probe clamp to it either way.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Times to repeat the whole sweep; per batch_tokens the median "
        "steps_per_sec across repeats is taken.",
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
        "--recorder-source",
        type=str,
        choices=sorted(RECORD_SOURCES),
        default=SPS_RECORD_SOURCE.name,
        help="Which server-side per-step record feed to read: 'sps' (legacy "
        "SpsDataRecorder via SGLANG_DSPARK_ENABLE_SPS_RECORD) or 'info' (the "
        "DsparkInfoDumper 'core'+'step_cpu_time' components via "
        "SGLANG_DSPARK_DEBUG_DUMP). Both yield the same steps_per_sec table.",
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

    batch_sizes = (
        args.batch_size
        if args.batch_size is not None
        else build_request_count_sweep(args.max_batch_size)
    )
    settings = RoundSettings(
        input_len=args.input_len,
        temperature=args.temperature,
        target_steady_steps=args.target_steady_steps,
        min_steady_steps=args.min_steady_steps,
        round_timeout_seconds=args.round_timeout,
        ramp_token_slack=args.ramp_token_slack,
    )

    profile(
        base_url=args.base_url,
        batch_sizes=batch_sizes,
        settings=settings,
        out=args.out,
        max_batch_tokens=args.max_batch_tokens,
        repeats=args.repeats,
        self_check=args.self_check,
        local_tokenizer_path=args.local_tokenizer_path,
        recorder_source=args.recorder_source,
    )


if __name__ == "__main__":
    cli_main()
