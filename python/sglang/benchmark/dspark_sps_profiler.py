"""
Profile a DSpark target uniform verify step into an SPS cost table (JSON).

This stands up a real DSpark speculative-decoding worker (no HTTP server) using
the same low-level model / runner / cuda-graph init as
``sglang.benchmark.one_batch``, drives one uniform verify step per probed
running-batch size, and writes a ``SpsCostTable`` consumed at serving time via
``--speculative-dspark-sps-table-path``.

The library routine ``sglang.srt.speculative.dspark_sps_table.profile_sps_table``
owns the sweep / median / table-assembly logic; this script supplies the GPU
driver and the timed ``time_uniform_verify_step`` callable.

# Usage (single GPU)
python -m sglang.benchmark.dspark_sps_profiler \
    --model-path <target> \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path <dspark-draft> \
    --speculative-dspark-block-size 4 \
    --probe-request-counts 1 2 4 8 16 32 \
    --iters 7 \
    --out ~/main/artifacts/sglang/dspark_sps_table.json

The timer (see ``_VerifyStepDriver``) prefills once to seed the draft KV + spec
state, runs warmup decode steps so the target verify cuda graph is captured and
replayed, discards the first measured step, and times each subsequent step with
``time.perf_counter()`` bracketed by a CUDA-event device sync. Each probe prints
its raw step duration and derived SPS for human sanity-checking, and an optional
read-back self-check loads the written JSON and asserts the lookup is sane.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from array import array
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.dspark_sps_table import (
    load_sps_table_from_path,
    profile_sps_table,
)
from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import configure_logger, suppress_other_loggers

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProfilerArgs:
    probe_request_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    gamma: Optional[int] = None
    iters: int = 7
    warmup_steps: int = 3
    prefill_len: int = 256
    max_batch_tokens: Optional[int] = None
    out: str = "~/main/artifacts/sglang/dspark_sps_table.json"
    self_check: bool = True

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--probe-request-counts",
            type=int,
            nargs="+",
            default=list(ProfilerArgs.probe_request_counts),
            help="Running-batch sizes to probe (each maps to batch_tokens = "
            "num_requests * (gamma + 1)).",
        )
        parser.add_argument(
            "--gamma",
            type=int,
            default=None,
            help="DSpark draft block size gamma. Defaults to the worker's resolved "
            "gamma (from --speculative-dspark-block-size / the draft config).",
        )
        parser.add_argument(
            "--iters",
            type=int,
            default=ProfilerArgs.iters,
            help="Timed verify steps per probe (median is taken). Excludes the "
            "discarded first measured step.",
        )
        parser.add_argument(
            "--warmup-steps",
            type=int,
            default=ProfilerArgs.warmup_steps,
            help="Untimed decode steps run before measuring, so the target verify "
            "cuda graph is captured and replayed before timing.",
        )
        parser.add_argument(
            "--prefill-len",
            type=int,
            default=ProfilerArgs.prefill_len,
            help="Synthetic prompt length used to seed each request's KV before the "
            "timed decode steps.",
        )
        parser.add_argument(
            "--max-batch-tokens",
            type=int,
            default=None,
            help="Override the table's max_batch_tokens (clamp bound). Defaults to "
            "the largest probe's batch_tokens.",
        )
        parser.add_argument(
            "--out",
            type=str,
            default=ProfilerArgs.out,
            help="Output JSON path for the SpsCostTable. Defaults under "
            "~/main/artifacts/sglang/ (raw experiment material, not checked in).",
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
            probe_request_counts=tuple(args.probe_request_counts),
            gamma=args.gamma,
            iters=args.iters,
            warmup_steps=args.warmup_steps,
            prefill_len=args.prefill_len,
            max_batch_tokens=args.max_batch_tokens,
            out=args.out,
            self_check=args.self_check,
        )


class _BenchTreeCache:
    """Minimal tree-cache stand-in for standalone batch allocation (no scheduler)."""

    def __init__(self, *, page_size: int, device: str, token_to_kv_pool_allocator):
        self.page_size = page_size
        self.device = device
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def evict(self, params) -> None:
        pass


def build_dspark_worker(
    *, server_args: ServerArgs, port_args: PortArgs, gpu_id: int, tp_rank: int
) -> DSparkWorkerV2:
    suppress_other_loggers()

    spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if not spec_algorithm.is_dspark():
        raise ValueError(
            "dspark_sps_profiler requires --speculative-algorithm DSPARK, got "
            f"{server_args.speculative_algorithm!r}."
        )

    target_worker = TpModelWorker(
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        moe_ep_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        dp_rank=None,
        nccl_port=port_args.nccl_port,
    )

    worker_class = spec_algorithm.create_worker(server_args)
    if worker_class is not DSparkWorkerV2:
        raise RuntimeError(
            f"Expected DSparkWorkerV2 from the spec factory, got {worker_class}."
        )
    spec_worker = DSparkWorkerV2(
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        dp_rank=None,
        moe_ep_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        nccl_port=port_args.nccl_port,
        target_worker=target_worker,
    )

    target_worker.alloc_memory_pool()
    pool, allocator = target_worker.get_memory_pool()
    spec_worker.alloc_memory_pool(
        memory_pool_config=target_worker.model_runner.memory_pool_config,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
    )

    target_worker.init_attention_backends()
    spec_worker.init_attention_backends()

    target_worker.init_cuda_graphs()
    spec_worker.init_cuda_graphs()

    logger.info(
        "Built DSpark worker: gamma=%s, verify_num_draft_tokens=%s, "
        "max_total_num_tokens=%s.",
        spec_worker.gamma,
        spec_worker.verify_num_draft_tokens,
        target_worker.model_runner.max_total_num_tokens,
    )
    return spec_worker


def _make_prefill_reqs(*, num_requests: int, prefill_len: int) -> list[Req]:
    sampling_params = SamplingParams(temperature=0, max_new_tokens=1024)

    reqs: list[Req] = []
    for rid in range(num_requests):
        token_ids = torch.randint(
            low=0, high=10000, size=(prefill_len,), dtype=torch.int64
        ).tolist()
        req = Req(
            rid=str(rid),
            origin_input_text="",
            origin_input_ids=array("q", token_ids),
            sampling_params=sampling_params,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
        reqs.append(req)
    return reqs


class _VerifyStepDriver:
    """One DSpark decode step = one uniform target verify step.

    Built once per probe: prefills ``num_requests`` synthetic requests to seed
    the draft KV + spec state, runs untimed warmup decode steps (so the target
    verify cuda graph is captured and replayed), discards the first measured
    step, then ``step()`` times exactly one further decode step with a CUDA-event
    bracketed device sync. In the default static verify mode the decode step runs
    the full ``num_requests * (gamma + 1)`` uniform verify window, which is the
    cost the SpsCostTable models.
    """

    def __init__(
        self,
        *,
        worker: DSparkWorkerV2,
        num_requests: int,
        prefill_len: int,
        warmup_steps: int,
    ):
        self._worker = worker
        self._device = worker.device
        self._device_module = torch.get_device_module(self._device)

        model_runner = worker.target_worker.model_runner
        self._tree_cache = _BenchTreeCache(
            page_size=worker.server_args.page_size,
            device=self._device,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        )
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()

        reqs = _make_prefill_reqs(num_requests=num_requests, prefill_len=prefill_len)
        self._batch = self._prefill(reqs=reqs, model_runner=model_runner)

        for _ in range(max(0, warmup_steps)):
            self._decode_once()
        # Discard the first measured step: the cuda graph may still be warming
        # its replay path on the transition out of warmup.
        self._decode_once()

    def _prefill(self, *, reqs: list[Req], model_runner) -> ScheduleBatch:
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=self._tree_cache,
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
        )
        batch.prepare_for_extend()
        prefill_out = self._worker.forward_batch_generation(batch)
        self._thread_spec_state(batch=batch, result=prefill_out)
        return batch

    def _decode_once(self) -> GenerationBatchResult:
        self._batch.prepare_for_decode()
        result = self._worker.forward_batch_generation(self._batch)
        self._commit_accept_lens(batch=self._batch, result=result)
        self._thread_spec_state(batch=self._batch, result=result)
        return result

    def _commit_accept_lens(
        self, *, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        # Mirror the scheduler's batch-result processor: advance each request's
        # committed KV watermark by its accepted run (drafts + bonus). The next
        # prepare_for_decode reads kv_committed_len to over-allocate the verify
        # window, so the watermark must track seq_lens across steps.
        accept_lens = result.accept_lens.to("cpu").tolist()
        for req, num_accept_tokens in zip(batch.reqs, accept_lens):
            req.kv_committed_len += int(num_accept_tokens)
            req.spec_verify_ct += 1

    def _thread_spec_state(
        self, *, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        # The scheduler threads the worker's next_draft_input into the running
        # batch's spec_info and advances seq_lens by the accepted length; mirror
        # the non-overlap path. prepare_for_decode rebuilds seq_lens_cpu from the
        # committed watermark, so only the GPU seq_lens is threaded here.
        batch.spec_info = result.next_draft_input
        batch.output_ids = result.next_token_ids
        if result.new_seq_lens is not None:
            batch.seq_lens = result.new_seq_lens
        batch.input_ids = None
        batch.forward_mode = ForwardMode.DECODE

    def step(self) -> float:
        # Allocation / spec-input prep happens outside the timed region so the
        # measurement isolates the target uniform verify forward.
        self._batch.prepare_for_decode()

        start_event = self._device_module.Event(enable_timing=True)
        end_event = self._device_module.Event(enable_timing=True)

        self._device_module.synchronize()
        tic = time.perf_counter()
        start_event.record()
        result = self._worker.forward_batch_generation(self._batch)
        end_event.record()
        self._device_module.synchronize()
        wall_seconds = time.perf_counter() - tic

        # Advance the running state so a subsequent step remains a valid uniform
        # verify step (excluded from the timed region above).
        self._commit_accept_lens(batch=self._batch, result=result)
        self._thread_spec_state(batch=self._batch, result=result)

        # Prefer the CUDA-event delta (device-side, excludes host scheduling
        # jitter); fall back to the synced wall time on non-CUDA devices.
        event_seconds = start_event.elapsed_time(end_event) / 1000.0
        return event_seconds if event_seconds > 0 else wall_seconds


def make_time_uniform_verify_step(*, prefill_len: int, warmup_steps: int):
    def time_uniform_verify_step(
        worker: DSparkWorkerV2, num_requests: int, gamma: int
    ) -> float:
        driver = _VerifyStepDriver(
            worker=worker,
            num_requests=num_requests,
            prefill_len=prefill_len,
            warmup_steps=warmup_steps,
        )
        duration = driver.step()
        steps_per_sec = (1.0 / duration) if duration > 0 else float("inf")
        logger.info(
            "verify-step probe: num_requests=%s, gamma=%s, batch_tokens=%s, "
            "duration=%.6fs, steps_per_sec=%.3f",
            num_requests,
            gamma,
            num_requests * (gamma + 1),
            duration,
            steps_per_sec,
        )
        return duration

    return time_uniform_verify_step


def run_self_check(*, out_path: Path) -> None:
    table = load_sps_table_from_path(str(out_path))
    if len(table.sample_batch_tokens) != len(table.sample_steps_per_sec):
        raise RuntimeError("Reloaded table has mismatched probe / SPS lengths.")

    # Every stored probe must look up to a positive SPS, and looking up below
    # the smallest probe must clamp to it (the loader's bisect-floor + clamp
    # contract). The per-step throughput should not increase as the batch grows
    # (a larger uniform verify window costs at least as much), so the table is
    # expected to be monotone non-increasing across probes.
    previous_sps: Optional[float] = None
    for batch_tokens, expected_sps in zip(
        table.sample_batch_tokens, table.sample_steps_per_sec
    ):
        looked_up = table.lookup(batch_tokens)
        if looked_up <= 0:
            raise RuntimeError(
                f"Reloaded table lookup at batch_tokens={batch_tokens} returned "
                f"non-positive SPS {looked_up}."
            )
        if previous_sps is not None and looked_up > previous_sps * 1.10:
            logger.warning(
                "Non-monotone SPS across probes: batch_tokens=%s SPS=%.3f rose "
                "above the previous probe's SPS=%.3f by >10%%; verify the timer "
                "is stable (warmup / sync / graph capture).",
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


def profile(
    server_args: ServerArgs, profiler_args: ProfilerArgs, gpu_id: int, tp_rank: int
) -> None:
    initialize_runtime_configs(server_args)
    configure_logger(server_args, prefix=f" TP{tp_rank}")

    worker = build_dspark_worker(
        server_args=server_args,
        port_args=PortArgs.init_new(server_args),
        gpu_id=gpu_id,
        tp_rank=tp_rank,
    )
    gamma = profiler_args.gamma if profiler_args.gamma is not None else worker.gamma
    if gamma != worker.gamma:
        logger.warning(
            "Requested gamma=%s differs from the worker's resolved gamma=%s; "
            "profiling the worker's actual verify geometry (gamma=%s).",
            gamma,
            worker.gamma,
            worker.gamma,
        )
        gamma = worker.gamma

    time_uniform_verify_step = make_time_uniform_verify_step(
        prefill_len=profiler_args.prefill_len,
        warmup_steps=profiler_args.warmup_steps,
    )

    table = profile_sps_table(
        target_worker=worker,
        probe_request_counts=list(profiler_args.probe_request_counts),
        gamma=gamma,
        iters=profiler_args.iters,
        time_uniform_verify_step=time_uniform_verify_step,
        max_batch_tokens=profiler_args.max_batch_tokens,
    )

    out_path = Path(profiler_args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table.to_json(), encoding="utf-8")
    logger.info(
        "Wrote SpsCostTable (%s probes) to %s",
        len(table.sample_batch_tokens),
        out_path,
    )

    if profiler_args.self_check:
        run_self_check(out_path=out_path)


def initialize_runtime_configs(server_args: ServerArgs) -> None:
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config

    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)


def main(server_args: ServerArgs, profiler_args: ProfilerArgs) -> None:
    if not server_args.model_path:
        raise ValueError("Provide --model-path for the DSpark target model.")
    if server_args.tp_size != 1:
        raise ValueError(
            "dspark_sps_profiler currently supports single-GPU profiling "
            "(tp_size == 1); got tp_size="
            f"{server_args.tp_size}."
        )
    # The verify-step driver advances real KV; keep overlap scheduling off so the
    # worker runs synchronously (the SpsCostTable models a single static step).
    server_args.disable_overlap_schedule = True

    profile(server_args, profiler_args, gpu_id=0, tp_rank=0)


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ProfilerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    profiler_args = ProfilerArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    main(server_args, profiler_args)


if __name__ == "__main__":
    cli_main()
