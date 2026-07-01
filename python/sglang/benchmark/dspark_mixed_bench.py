"""Mixed-workload client driver for the DSpark per-request verify-length demo.

Fires a controllable mix of high-confidence datasets (gsm8k, aime25) and a
low-confidence dataset (arena-hard) at an OpenAI ``/v1/chat/completions``
endpoint, tagging every request with a stable ``rid`` so the server-side
DSpark decision dump can be grouped back by dataset label. It reuses
``sglang.benchmark.serving`` as a library for the OAI streaming client and
the ITL/TTFT accounting, and only adds the multi-dataset mixing, per-request
tagging, concurrency control, and JSONL recording on top.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import statistics
import time
from pathlib import Path
from typing import Annotated, Optional

import msgspec
import typer
from tqdm import tqdm

from sglang.benchmark import serving
from sglang.test.simple_eval_aime25 import QUERY_TEMPLATE as AIME_QUERY_TEMPLATE
from sglang.test.simple_eval_aime25 import normalize_aime_answer
from sglang.test.simple_eval_common import ANSWER_PATTERN
from sglang.test.simple_eval_gsm8k import (
    GSM8K_URL,
    get_answer_value,
    get_few_shot_examples,
    get_one_example,
)
from sglang.utils import download_and_cache_file, read_jsonl

logger = logging.getLogger(__name__)


class DatasetSpec(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    label: str
    temperature: float
    enable_thinking: bool
    score_kind: str


class TaggedRequest(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    rid: str
    label: str
    qid: int
    run: int
    prompt: str
    spec: DatasetSpec
    gold_answer: Optional[str] = None


class RequestRecord(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    rid: str
    label: str
    qid: int
    run: int
    prompt_len: int
    output_len: int
    ttft: float
    latency: float
    tpot: float
    itl: list[float]
    success: bool
    error: str
    output_text: str
    correct: Optional[bool] = None
    # Per-request accept length the server reports in the response meta_info
    # (completion_tokens / verify_ct, incl. bonus) -- an OAI-side cross-check of the
    # dump-derived acc_len, independent of the DSpark decision dump.
    spec_accept_length: float = 0.0


def build_gsm8k_requests(
    *,
    num: int,
    runs: int,
    num_shots: int,
    spec: DatasetSpec,
    data_path: Optional[str],
) -> list[TaggedRequest]:
    if num <= 0:
        return []

    filename = data_path if data_path else download_and_cache_file(GSM8K_URL)
    all_lines = list(read_jsonl(filename))
    few_shot_prompt = get_few_shot_examples(all_lines, num_shots)
    # The few-shot pool is excluded from the eval set to avoid leakage, mirroring
    # sglang.test.simple_eval_gsm8k.GSM8KEval.
    eval_lines = all_lines[num_shots:][:num]

    requests: list[TaggedRequest] = []
    for qid, line in enumerate(eval_lines):
        prompt = few_shot_prompt + get_one_example(
            eval_lines, qid, include_answer=False
        )
        for run in range(runs):
            requests.append(
                TaggedRequest(
                    rid=f"{spec.label}::{qid:05d}::r{run}",
                    label=spec.label,
                    qid=qid,
                    run=run,
                    prompt=prompt,
                    spec=spec,
                    gold_answer=line["answer"],
                )
            )
    return requests


def _load_aime25_rows(data_path: Optional[str]) -> list[dict]:
    # Local jsonl (list of {question, answer}) takes precedence so a pod without
    # HF network access can pre-stage the 30-question set; otherwise pull the two
    # AIME2025 configs straight from HuggingFace (opencompass/AIME2025).
    if data_path:
        return [
            {"question": line["question"], "answer": str(line["answer"])}
            for line in read_jsonl(data_path)
        ]

    from datasets import load_dataset

    rows: list[dict] = []
    for config in ("AIME2025-I", "AIME2025-II"):
        dataset = load_dataset("opencompass/AIME2025", config, split="test")
        rows += [
            {"question": row["question"], "answer": str(row["answer"])}
            for row in dataset
        ]
    return rows


def build_aime25_requests(
    *,
    num: int,
    runs: int,
    spec: DatasetSpec,
    data_path: Optional[str],
) -> list[TaggedRequest]:
    if num <= 0:
        return []

    rows = _load_aime25_rows(data_path)[:num]

    requests: list[TaggedRequest] = []
    for qid, row in enumerate(rows):
        prompt = AIME_QUERY_TEMPLATE.format(question=row["question"])
        for run in range(runs):
            requests.append(
                TaggedRequest(
                    rid=f"{spec.label}::{qid:05d}::r{run}",
                    label=spec.label,
                    qid=qid,
                    run=run,
                    prompt=prompt,
                    spec=spec,
                    gold_answer=row["answer"],
                )
            )
    return requests


def _arena_prompt(line: dict) -> str:
    # arena-hard-auto changed schema across releases: v0.1 stores the single-turn
    # question under turns[0].content, v2.0 stores it as a flat ``prompt`` string.
    turns = line.get("turns")
    if turns:
        return turns[0]["content"]
    prompt = line.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        return first["content"] if isinstance(first, dict) else str(first)
    raise ValueError(f"Unrecognized arena-hard line schema: keys={list(line.keys())}")


def build_arena_requests(
    *,
    num: int,
    runs: int,
    spec: DatasetSpec,
    data_path: str,
) -> list[TaggedRequest]:
    if num <= 0:
        return []

    lines = list(read_jsonl(data_path))[:num]

    requests: list[TaggedRequest] = []
    for qid, line in enumerate(lines):
        prompt = _arena_prompt(line)
        for run in range(runs):
            requests.append(
                TaggedRequest(
                    rid=f"{spec.label}::{qid:05d}::r{run}",
                    label=spec.label,
                    qid=qid,
                    run=run,
                    prompt=prompt,
                    spec=spec,
                    gold_answer=None,
                )
            )
    return requests


def _score_correct(*, req: TaggedRequest, generated_text: str) -> Optional[bool]:
    if req.gold_answer is None:
        return None
    if req.spec.score_kind == "gsm8k":
        return get_answer_value(generated_text) == get_answer_value(req.gold_answer)
    if req.spec.score_kind == "aime25":
        match = re.search(ANSWER_PATTERN, generated_text)
        extracted = match.group(1).strip() if match else None
        return normalize_aime_answer(extracted) == normalize_aime_answer(
            req.gold_answer
        )
    return None


def build_record(
    *, req: TaggedRequest, out: serving.RequestFuncOutput
) -> RequestRecord:
    itl = list(out.itl)
    if itl:
        tpot = sum(itl) / len(itl)
    elif out.output_len > 1:
        tpot = (out.latency - out.ttft) / (out.output_len - 1)
    else:
        tpot = 0.0

    correct = (
        _score_correct(req=req, generated_text=out.generated_text)
        if out.success
        else None
    )

    return RequestRecord(
        rid=req.rid,
        label=req.label,
        qid=req.qid,
        run=req.run,
        prompt_len=out.prompt_len,
        output_len=out.output_len,
        ttft=out.ttft,
        latency=out.latency,
        tpot=tpot,
        itl=itl,
        success=out.success,
        error=out.error,
        output_text=out.generated_text,
        correct=correct,
        spec_accept_length=out.spec_accept_length,
    )


async def run_workload(
    *,
    requests: list[TaggedRequest],
    base_url: str,
    model: str,
    max_osl: int,
    max_concurrency: int,
    disable_stream: bool,
    jsonl_path: Path,
) -> list[RequestRecord]:
    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    semaphore = asyncio.Semaphore(max_concurrency)
    file_lock = asyncio.Lock()
    encoder = msgspec.json.Encoder()
    records: list[RequestRecord] = []
    pbar = tqdm(total=len(requests), desc=f"conc={max_concurrency}")

    with jsonl_path.open("w") as fout:

        async def send_one(req: TaggedRequest) -> None:
            # Always generate to natural EOS (ignore_eos stays False): output length is
            # request-driven, capped at max_osl only as a runaway guard.
            #
            # Provide temperature/ignore_eos/thinking explicitly: the reused chat helper
            # only falls back to the module-global ``args`` when a key is absent from
            # extra_request_body, so being explicit keeps per-request control local.
            extra_request_body = {
                "rid": req.rid,
                "temperature": req.spec.temperature,
                "ignore_eos": False,
                "chat_template_kwargs": {"enable_thinking": req.spec.enable_thinking},
                # Ask for a trailing usage chunk so the reused streaming client can read
                # the real completion_tokens; without it out.output_len falls back to the
                # requested max and throughput/tpot come out wrong.
                "stream_options": {"include_usage": True},
            }
            if disable_stream:
                # Per-request spec_accept_length is only in choices[0].meta_info of a
                # non-streaming response (the streaming usage chunk omits it), so ask for
                # meta_info here for the OAI-side acc_len cross-check. The server rejects
                # return_meta_info under streaming, hence gating both on disable_stream.
                extra_request_body["return_meta_info"] = True
            request_func_input = serving.RequestFuncInput(
                prompt=req.prompt,
                api_url=api_url,
                prompt_len=len(req.prompt),
                output_len=max_osl,
                model=model,
                lora_name="",
                image_data=None,
                extra_request_body=extra_request_body,
            )

            async with semaphore:
                out = await serving.async_request_openai_chat_completions(
                    request_func_input=request_func_input
                )

            record = build_record(req=req, out=out)
            async with file_lock:
                fout.write(encoder.encode(record).decode() + "\n")
                fout.flush()
                records.append(record)
            pbar.update(1)

        await asyncio.gather(*(send_one(req) for req in requests))

    pbar.close()
    return records


def _mean(values: list[float]) -> Optional[float]:
    return round(statistics.fmean(values), 6) if values else None


def _median(values: list[float]) -> Optional[float]:
    return round(statistics.median(values), 6) if values else None


def write_summary(
    *,
    records: list[RequestRecord],
    wall_time: float,
    summary_path: Path,
) -> dict:
    by_label: dict[str, list[RequestRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    labels: dict[str, dict] = {}
    for label, label_records in sorted(by_label.items()):
        ok = [r for r in label_records if r.success]
        output_tokens = sum(r.output_len for r in ok)
        scored = [r for r in ok if r.correct is not None]
        accuracy = (
            round(sum(1 for r in scored if r.correct) / len(scored), 4)
            if scored
            else None
        )
        spec_acc = [r.spec_accept_length for r in ok if r.spec_accept_length > 0]
        labels[label] = {
            "num": len(label_records),
            "num_success": len(ok),
            "mean_ttft_s": _mean([r.ttft for r in ok]),
            "mean_tpot_s": _mean([r.tpot for r in ok]),
            "median_itl_s": _median([x for r in ok for x in r.itl]),
            "mean_output_len": _mean([float(r.output_len) for r in ok]),
            "output_tokens": output_tokens,
            "output_throughput_tok_s": (
                round(output_tokens / wall_time, 2) if wall_time > 0 else None
            ),
            "accuracy": accuracy,
            # OAI-side per-request accept length (mean over requests that reported it),
            # to cross-check the dump-derived acc_len from analyze_mixed.py.
            "oai_acc_len": _mean(spec_acc),
            "oai_acc_len_n": len(spec_acc),
        }

    summary = {
        "wall_time_s": round(wall_time, 3),
        "num_requests": len(records),
        "labels": labels,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(
    base_url: Annotated[
        str,
        typer.Option(
            help="Server base URL (OAI endpoint is <base>/v1/chat/completions)."
        ),
    ] = "http://127.0.0.1:30000",
    model: Annotated[
        str, typer.Option(help="Model id the server serves.")
    ] = "deepseek-ai/DeepSeek-V4-Flash",
    dataset: Annotated[
        str, typer.Option(help="gsm8k | arena-hard | aime25 | mixed.")
    ] = "mixed",
    num_gsm: Annotated[int, typer.Option(help="Distinct gsm8k questions.")] = 32,
    num_arena: Annotated[int, typer.Option(help="Distinct arena-hard questions.")] = 32,
    num_aime: Annotated[
        int, typer.Option(help="Distinct aime25 questions (max 30).")
    ] = 0,
    runs: Annotated[
        int,
        typer.Option(help="Repeat each question this many times to fill concurrency."),
    ] = 1,
    max_concurrency: Annotated[
        int, typer.Option(help="In-flight request cap (the concurrency knob).")
    ] = 64,
    max_osl: Annotated[
        int,
        typer.Option(
            help="Max output tokens; generation still stops at natural EOS (no ignore_eos)."
        ),
    ] = 8192,
    gsm_temp: Annotated[float, typer.Option(help="gsm8k temperature.")] = 0.0,
    gsm_thinking: Annotated[bool, typer.Option(help="gsm8k enable_thinking.")] = True,
    arena_temp: Annotated[float, typer.Option(help="arena-hard temperature.")] = 1.0,
    arena_thinking: Annotated[
        bool, typer.Option(help="arena-hard enable_thinking.")
    ] = True,
    aime_temp: Annotated[float, typer.Option(help="aime25 temperature.")] = 0.0,
    aime_thinking: Annotated[bool, typer.Option(help="aime25 enable_thinking.")] = True,
    num_shots: Annotated[int, typer.Option(help="gsm8k few-shot count.")] = 5,
    seed: Annotated[int, typer.Option(help="Shuffle seed for mixed interleave.")] = 0,
    out_dir: Annotated[
        Path, typer.Option(help="Output directory for jsonl + summary.")
    ] = Path("dspark_mixed_out"),
    gsm_data_path: Annotated[
        Optional[str], typer.Option(help="Local gsm8k test.jsonl (default: download).")
    ] = None,
    arena_data_path: Annotated[
        Optional[str],
        typer.Option(help="Local arena-hard question.jsonl (required for arena)."),
    ] = None,
    aime_data_path: Annotated[
        Optional[str],
        typer.Option(
            help="Local aime25 {question,answer} jsonl (default: HF download)."
        ),
    ] = None,
    disable_stream: Annotated[
        bool,
        typer.Option(
            help="Non-streaming requests + return_meta_info, to capture the OAI-side "
            "per-request spec_accept_length (oai_acc_len cross-check). Loses ITL/TTFT."
        ),
    ] = False,
) -> None:
    """Drive a single/mixed gsm8k + arena-hard + aime25 workload at a DSpark OAI endpoint."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    random.seed(seed)

    # The reused serving.async_request_openai_chat_completions reads a module-global
    # ``args`` for disable_stream / disable_ignore_eos; set it before any request.
    # disable_ignore_eos=True keeps the fallback on natural EOS even though every
    # request also sends ignore_eos=False explicitly.
    serving.set_global_args(
        argparse.Namespace(
            disable_stream=disable_stream,
            disable_ignore_eos=True,
            print_requests=False,
        )
    )

    gsm_spec = DatasetSpec(
        label="gsm8k",
        temperature=gsm_temp,
        enable_thinking=gsm_thinking,
        score_kind="gsm8k",
    )
    arena_spec = DatasetSpec(
        label="arena-hard",
        temperature=arena_temp,
        enable_thinking=arena_thinking,
        score_kind="none",
    )
    aime_spec = DatasetSpec(
        label="aime25",
        temperature=aime_temp,
        enable_thinking=aime_thinking,
        score_kind="aime25",
    )

    requests: list[TaggedRequest] = []
    if dataset in ("gsm8k", "mixed"):
        requests += build_gsm8k_requests(
            num=num_gsm,
            runs=runs,
            num_shots=num_shots,
            spec=gsm_spec,
            data_path=gsm_data_path,
        )
    if dataset in ("arena-hard", "mixed"):
        if num_arena > 0 and not arena_data_path:
            raise typer.BadParameter(
                "--arena-data-path is required for arena-hard / mixed datasets."
            )
        requests += build_arena_requests(
            num=num_arena,
            runs=runs,
            spec=arena_spec,
            data_path=arena_data_path or "",
        )
    if dataset in ("aime25", "mixed"):
        requests += build_aime25_requests(
            num=num_aime,
            runs=runs,
            spec=aime_spec,
            data_path=aime_data_path,
        )

    if not requests:
        raise typer.BadParameter(f"No requests built for dataset={dataset}.")

    # Interleave so the server's running batch actually mixes both datasets in steady
    # state; without the shuffle we would drain all gsm before any arena starts.
    random.shuffle(requests)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}-c{max_concurrency}-n{len(requests)}"
    jsonl_path = out_dir / f"records-{tag}.jsonl"
    summary_path = out_dir / f"summary-{tag}.json"

    logger.info(
        "dispatching %d requests dataset=%s concurrency=%d -> %s",
        len(requests),
        dataset,
        max_concurrency,
        jsonl_path,
    )

    start = time.perf_counter()
    records = asyncio.run(
        run_workload(
            requests=requests,
            base_url=base_url,
            model=model,
            max_osl=max_osl,
            max_concurrency=max_concurrency,
            disable_stream=disable_stream,
            jsonl_path=jsonl_path,
        )
    )
    wall_time = time.perf_counter() - start

    summary = write_summary(
        records=records, wall_time=wall_time, summary_path=summary_path
    )
    logger.info("summary -> %s", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
