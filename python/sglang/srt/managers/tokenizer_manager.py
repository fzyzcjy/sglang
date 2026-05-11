# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import copy
import logging
import os
import signal
import sys
import threading
from contextlib import nullcontext
from enum import Enum
from typing import Dict, List, Optional, Union

import fastapi
import uvloop
import zmq
import zmq.asyncio

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.corpus_controller import (
    CorpusController,
    CorpusControllerConfig,
)
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    FreezeGCReq,
    GenerateReqInput,
    HealthCheckOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.lora_controller import (
    LoraController,
    LoraControllerConfig,
)
from sglang.srt.managers.mm_utils import TensorTransportMode, wrap_shm_features
from sglang.srt.managers.multimodal_processor import MultimodalProcessor
from sglang.srt.managers.output_processor import (
    OutputProcessor,
    OutputProcessorConfig,
)
from sglang.srt.managers.pause_controller import (
    PauseController,
    PauseControllerConfig,
)
from sglang.srt.managers.raw_tokenizer_wrapper import RawTokenizerWrapper
from sglang.srt.managers.request_log_manager import RequestLogManager
from sglang.srt.managers.request_metrics_recorder import RequestMetricsRecorder
from sglang.srt.managers.request_preparer import (
    RequestPreparer,
    RequestPreparerConfig,
)
from sglang.srt.managers.request_state import ReqState, init_req
from sglang.srt.managers.request_validator import (
    RequestValidator,
    RequestValidatorConfig,
)
from sglang.srt.managers.response_emitter import (
    ResponseEmitter,
    ResponseEmitterConfig,
)
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.score_request_handler import (
    ScoreRequestHandler,
    ScoreRequestHandlerConfig,
)
from sglang.srt.managers.session_controller import (
    SessionController,
    SessionControllerConfig,
)
from sglang.srt.managers.tokenized_request_builder import (
    TokenizedRequestBuilder,
    TokenizedRequestBuilderConfig,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.weight_disk_update_controller import (
    WeightDiskUpdateController,
    WeightDiskUpdateControllerConfig,
)
from sglang.srt.observability.req_time_stats import (
    real_time,
    set_time_batch,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_tokenizer,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_gc_warning,
    freeze_gc,
    get_bool_env_var,
    get_or_create_event_loop,
    kill_process_tree,
)
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
)
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_REQUEST_STATE_WAIT_TIMEOUT = envs.SGLANG_REQUEST_STATE_WAIT_TIMEOUT.get()

logger = logging.getLogger(__name__)


class TokenizerManager(TokenizerControlMixin):
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.preferred_sampling_params = server_args.preferred_sampling_params
        self.crash_dump_folder = server_args.crash_dump_folder
        set_global_server_args_for_tokenizer(server_args)

        # Init model config
        self.init_model_config()

        # Initialize tokenizer and multimodal processor
        self.raw_tokenizer_wrapper = RawTokenizerWrapper.from_server_args(
            server_args=self.server_args,
            model_config=self.model_config,
        )

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init running status
        self.init_running_status()

        # Init weight update
        self.init_weight_update()

        # Init PD disaggregation and encoder disaggregation
        self.init_disaggregation()

        # Init metric collector and watchdog
        self.init_metric_collector_watchdog()

        # Result dispatcher (created early so controllers can register handlers in __post_init__)
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (FreezeGCReq, lambda x: None),
                (HealthCheckOutput, lambda x: None),
                (ActiveRanksOutput, self.update_active_ranks),
            ]
        )

        # Communicators (RPC fan-out) -- needed by owner-class ctors below.
        self.init_communicators(self.server_args)

        # Multimodal processor
        self.multimodal_processor = MultimodalProcessor.from_server_args(
            server_args=self.server_args,
            model_config=self.model_config,
            mm_processor=self.raw_tokenizer_wrapper.mm_processor,
        )

        # Tokenized request builder
        self.tokenized_request_builder = TokenizedRequestBuilder(
            tokenizer=self.raw_tokenizer_wrapper.tokenizer,
            config=TokenizedRequestBuilderConfig(
                vocab_size=self.model_config.vocab_size,
                preferred_sampling_params=self.preferred_sampling_params,
                sampling_params_class=SamplingParams,
                disaggregation_transfer_backend=self.server_args.disaggregation_transfer_backend,
            ),
        )

        # Request metrics recorder
        self.request_metrics_recorder = RequestMetricsRecorder(
            server_args=self.server_args,
            enable_metrics=self.enable_metrics,
            enable_priority_scheduling=self.enable_priority_scheduling,
        )

        # Request log manager
        self.request_log_manager = RequestLogManager.from_server_args(
            server_args=self.server_args,
        )

        # Request validator
        self.request_validator = RequestValidator(
            config=RequestValidatorConfig(
                context_len=self.context_len,
                num_reserved_tokens=self.num_reserved_tokens,
                is_generation=self.is_generation,
                validate_total_tokens=self.validate_total_tokens,
                allow_auto_truncate=self.server_args.allow_auto_truncate,
                enable_return_hidden_states=self.server_args.enable_return_hidden_states,
                enable_custom_logit_processor=self.server_args.enable_custom_logit_processor,
                limit_mm_data_per_request=self.server_args.limit_mm_data_per_request,
                is_matryoshka=self.model_config.is_matryoshka,
                matryoshka_dimensions=self.model_config.matryoshka_dimensions,
                hidden_size=self.model_config.hidden_size,
                model_path=self.model_config.model_path,
            ),
        )

        # Request preparer
        self.request_preparer = RequestPreparer(
            raw_tokenizer_wrapper=self.raw_tokenizer_wrapper,
            multimodal_processor=self.multimodal_processor,
            request_validator=self.request_validator,
            tokenized_request_builder=self.tokenized_request_builder,
            rid_to_state=self.rid_to_state,
            config=RequestPreparerConfig(
                skip_tokenizer_init=self.server_args.skip_tokenizer_init,
                enable_dp_attention=self.server_args.enable_dp_attention,
                enable_tokenizer_batch_encode=self.server_args.enable_tokenizer_batch_encode,
                is_generation=self.is_generation,
                disable_radix_cache=self.server_args.disable_radix_cache,
                is_multimodal=self.model_config.is_multimodal,
                architectures=self.model_config.hf_config.architectures,
                max_req_input_len=self.max_req_input_len,
                language_only=self.server_args.language_only,
                encoder_transfer_backend=self.server_args.encoder_transfer_backend,
            ),
        )

        # Score request handler
        self.score_request_handler = ScoreRequestHandler(
            tokenizer=self.raw_tokenizer_wrapper.tokenizer,
            rid_to_state=self.rid_to_state,
            generate_request=self.generate_request,
            config=ScoreRequestHandlerConfig(
                is_generation=self.is_generation,
                enable_mis=self.server_args.enable_mis,
                model_config=self.model_config,
            ),
        )

        # Pause controller
        self.pause_controller = PauseController(
            send_to_scheduler=self.send_to_scheduler,
            dispatcher=self._result_dispatcher,
            rid_to_state=self.rid_to_state,
            model_update_lock=self.model_update_lock,
            metrics_collector=self.request_metrics_recorder.metrics_collector,
            tokenizer=self.raw_tokenizer_wrapper.tokenizer,
            config=PauseControllerConfig(
                enable_metrics=self.enable_metrics,
                skip_tokenizer_init=self.server_args.skip_tokenizer_init,
                weight_version=self.server_args.weight_version,
            ),
        )

        # Weight disk update controller
        self.weight_disk_update_controller = WeightDiskUpdateController(
            send_to_scheduler=self.send_to_scheduler,
            dispatcher=self._result_dispatcher,
            pause_controller=self.pause_controller,
            model_update_lock=self.model_update_lock,
            server_args=self.server_args,
            auto_create_handle_loop=self.auto_create_handle_loop,
            config=WeightDiskUpdateControllerConfig(
                dp_size=self.server_args.dp_size,
                initial_load_format=self.server_args.load_format,
                checkpoint_engine_wait_weights_before_ready=self.server_args.checkpoint_engine_wait_weights_before_ready,
            ),
        )

        # Lora controller
        self.lora_controller = LoraController(
            server_args=self.server_args,
            auto_create_handle_loop=self.auto_create_handle_loop,
            update_lora_adapter_communicator=self.update_lora_adapter_communicator,
            config=LoraControllerConfig(
                enable_lora=self.server_args.enable_lora,
                max_loaded_loras=self.server_args.max_loaded_loras,
                dp_size=self.server_args.dp_size,
                initial_lora_paths=self.server_args.lora_paths,
            ),
        )

        # Corpus controller
        self.corpus_controller = CorpusController(
            add_external_corpus_communicator=self.add_external_corpus_communicator,
            remove_external_corpus_communicator=self.remove_external_corpus_communicator,
            list_external_corpora_communicator=self.list_external_corpora_communicator,
            tokenizer=self.raw_tokenizer_wrapper.tokenizer,
            config=CorpusControllerConfig(
                speculative_algorithm=self.server_args.speculative_algorithm or "",
                max_external_corpus_tokens=self.server_args.speculative_ngram_external_corpus_max_tokens,
            ),
            auto_create_handle_loop=self.auto_create_handle_loop,
        )

        # Output processor
        self.output_processor = OutputProcessor(
            rid_to_state=self.rid_to_state,
            tokenizer=self.raw_tokenizer_wrapper.tokenizer,
            request_metrics_recorder=self.request_metrics_recorder,
            request_log_manager=self.request_log_manager,
            lora_controller=self.lora_controller,
            send_to_scheduler=self.send_to_scheduler,
            config=OutputProcessorConfig(
                weight_version=self.server_args.weight_version,
                batch_notify_size=self.server_args.batch_notify_size,
                incremental_streaming_output=self.server_args.incremental_streaming_output,
                enable_metrics=self.enable_metrics,
                skip_tokenizer_init=self.server_args.skip_tokenizer_init,
                speculative_algorithm=self.server_args.speculative_algorithm or "",
                speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                dp_size=self.server_args.dp_size,
                enable_lora=self.server_args.enable_lora,
                served_model_name=self.server_args.served_model_name,
            ),
        )

        # Response emitter
        self.response_emitter = ResponseEmitter(
            rid_to_state=self.rid_to_state,
            pause_controller=self.pause_controller,
            lora_controller=self.lora_controller,
            request_log_manager=self.request_log_manager,
            request_metrics_recorder=self.request_metrics_recorder,
            config=ResponseEmitterConfig(
                incremental_streaming_output=self.server_args.incremental_streaming_output,
                enable_lora=self.server_args.enable_lora,
            ),
        )

        # Session controller
        self.session_controller = SessionController(
            send_to_scheduler=self.send_to_scheduler,
            dispatcher=self._result_dispatcher,
            config=SessionControllerConfig(
                enable_streaming_session=self.server_args.enable_streaming_session,
            ),
        )

        # Init request dispatcher
        self.init_request_dispatcher()

    def init_model_config(self):
        server_args = self.server_args
        model_config_class = getattr(self, "model_config_class", ModelConfig)

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = model_config_class.from_server_args(server_args)
        self.is_generation = self.model_config.is_generation
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id
        self.max_req_input_len = None  # Will be set later in engine.py
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.default_priority_value = server_args.default_priority_value
        speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        if speculative_algorithm.is_eagle():
            # In the current eagle implementation, we store the draft tokens in the output token slots,
            # so we need to reserve the space for the draft tokens.
            self.num_reserved_tokens = max(
                server_args.speculative_eagle_topk * server_args.speculative_num_steps,
                server_args.speculative_num_draft_tokens,
            )
        else:
            self.num_reserved_tokens = 0
        self.validate_total_tokens = True

    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        if self.server_args.tokenizer_worker_num == 1:
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )
        else:
            from sglang.srt.managers.multi_tokenizer_mixin import SenderWrapper

            # Use tokenizer_worker_ipc_name in multi-tokenizer mode
            send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False
            )

            # Make sure that each request carries the tokenizer_ipc_name for response routing
            self.send_to_scheduler = SenderWrapper(port_args, send_to_scheduler)

    def init_running_status(self):
        # Request states
        self.rid_to_state: Dict[str, ReqState] = {}
        self.event_loop = None
        self.asyncio_tasks = set()

        # Health check
        self.server_status = ServerStatus.Starting
        self.gracefully_exit = False
        self.last_receive_tstamp = real_time()

        # Subprocess liveness watchdog — set by Engine or http_server after construction
        self._subprocess_watchdog = None

    def init_weight_update(self):
        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()

    def init_disaggregation(self):
        # PD Disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        start_disagg_service(self.server_args)
        # Single-source counter for auto-assigning fake bootstrap_room.
        self.fake_bootstrap_room_counter = 0

    def init_metric_collector_watchdog(self):
        if self.server_args.gc_warning_threshold_secs > 0.0:
            configure_gc_warning(self.server_args.gc_warning_threshold_secs)
        self.soft_watchdog = Watchdog.create(
            debug_name="TokenizerManager",
            watchdog_timeout=self.server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_TOKENIZER.get(),
        )

    def init_request_dispatcher(self):
        self.sampling_params_class = SamplingParams
        self.signal_handler_class = SignalHandler

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()

        # Normalize the request
        obj.normalize_batch_and_arguments()
        self._set_default_priority(obj)

        if isinstance(obj, GenerateReqInput) and obj.routed_dp_rank is not None:
            dp_size = self.server_args.dp_size
            if dp_size <= 1 and obj.routed_dp_rank == 0:
                logger.warning(
                    f"routed_dp_rank={obj.routed_dp_rank} is ignored because dp_size={dp_size}"
                )
            elif obj.routed_dp_rank < 0 or obj.routed_dp_rank >= dp_size:
                raise ValueError(
                    f"routed_dp_rank={obj.routed_dp_rank} out of range [0, {dp_size})"
                )

        init_req(
            self.rid_to_state,
            obj=obj,
            request=request,
            enable_trace=self.server_args.enable_trace,
            disagg_mode=self.disaggregation_mode,
        )
        if self.server_args.language_only:
            self.multimodal_processor.maybe_dispatch_to_encoder(obj)
        if self.server_args.tokenizer_worker_num > 1:
            self._attach_multi_http_worker_info(obj)

        # Log the request
        self.request_log_manager.request_logger.log_received_request(
            obj, self.raw_tokenizer_wrapper.tokenizer, request
        )

        async with self.pause_controller.is_pause_cond:
            await self.pause_controller.is_pause_cond.wait_for(
                lambda: not self.pause_controller.is_pause
            )

        async with self.model_update_lock.reader_lock:
            await self.lora_controller._validate_and_resolve_lora(obj)

            # Tokenize the request and send it to the scheduler
            if obj.is_single:
                tokenized_obj = await self.request_preparer._tokenize_one_request(obj)
                self._send_one_request(tokenized_obj)
                async for response in self.response_emitter._wait_one_response(
                    obj, request
                ):
                    yield response
            else:
                async for response in self._handle_batch_request(obj, request):
                    yield response

    def _send_one_request(
        self,
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        tokenized_obj.time_stats.set_api_server_dispatch_time()
        tokenized_obj = wrap_shm_features(tokenized_obj)
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        tokenized_obj.time_stats.set_api_server_dispatch_finish_time()

    def _send_batch_request(
        self,
        tokenized_objs: List[
            Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
        ],
    ):
        """Send a batch of tokenized requests as a single batched request to the scheduler."""
        if isinstance(tokenized_objs[0], TokenizedGenerateReqInput):
            batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)
        else:
            batch_req = BatchTokenizedEmbeddingReqInput(batch=tokenized_objs)

        set_time_batch(tokenized_objs, "set_api_server_dispatch_time")
        self.send_to_scheduler.send_pyobj(batch_req)
        set_time_batch(tokenized_objs, "set_api_server_dispatch_finish_time")

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self.request_preparer._should_use_batch_tokenization(batch_size, obj):
                tokenized_objs = (
                    await self.request_preparer._batch_tokenize_and_process(
                        batch_size, obj
                    )
                )
                self._send_batch_request(tokenized_objs)

                # Set up generators for each request in the batch
                for i in range(batch_size):
                    tmp_obj = obj[i]
                    generators.append(
                        self.response_emitter._wait_one_response(tmp_obj, request)
                    )
                    rids.append(tmp_obj.rid)
            else:
                # Sequential tokenization and processing
                with (
                    input_blocker_guard_region(send_to_scheduler=self.send_to_scheduler)
                    if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
                    else nullcontext()
                ):
                    for i in range(batch_size):
                        tmp_obj = obj[i]
                        tokenized_obj = (
                            await self.request_preparer._tokenize_one_request(tmp_obj)
                        )
                        self._send_one_request(tokenized_obj)
                        generators.append(
                            self.response_emitter._wait_one_response(tmp_obj, request)
                        )
                        rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self.request_preparer._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                init_req(
                    self.rid_to_state,
                    obj=tmp_obj,
                    enable_trace=self.server_args.enable_trace,
                    disagg_mode=self.disaggregation_mode,
                )
                self._send_one_request(tokenized_obj)
                await self.response_emitter._wait_one_response(
                    tmp_obj, request
                ).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    init_req(
                        self.rid_to_state,
                        obj=tmp_obj,
                        enable_trace=self.server_args.enable_trace,
                        disagg_mode=self.disaggregation_mode,
                    )
                    tokenized_obj.time_stats = self.rid_to_state[tmp_obj.rid].time_stats
                    self._send_one_request(tokenized_obj)
                    generators.append(
                        self.response_emitter._wait_one_response(tmp_obj, request)
                    )
                    rids.append(tmp_obj.rid)

                self.rid_to_state[objs[i].rid].time_stats.set_finished_time()
                del self.rid_to_state[objs[i].rid]

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    def configure_logging(self, obj: ConfigureLoggingReq):
        self.request_log_manager.request_logger.configure(
            log_requests=obj.log_requests,
            log_requests_level=obj.log_requests_level,
            log_requests_format=obj.log_requests_format,
        )
        if obj.dump_requests_folder is not None:
            self.request_log_manager.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.request_log_manager.dump_requests_threshold = (
                obj.dump_requests_threshold
            )
        if obj.dump_requests_exclude_meta_keys is not None:
            self.request_log_manager.dump_requests_exclude_meta_keys = list(
                obj.dump_requests_exclude_meta_keys
            )
        if obj.crash_dump_folder is not None:
            self.request_log_manager.crash_dump_folder = obj.crash_dump_folder
        logging.info(f"Config logging: {obj=}")

    async def freeze_gc(self):
        """Send a freeze_gc message to the scheduler first, then freeze locally."""
        self.send_to_scheduler.send_pyobj(FreezeGCReq())
        freeze_gc("Tokenizer Manager")
        return None

    def auto_create_handle_loop(self):
        if self.event_loop is not None:
            return

        # Create and start the handle_loop task
        loop = get_or_create_event_loop()
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))
        )
        self.event_loop = loop

        # We only add signal handler when the tokenizer manager is in the main thread
        # due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = self.signal_handler_class(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(
                signal.SIGQUIT, signal_handler.running_phase_sigquit_handler
            )

        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
        )

    async def handle_loop(self):
        """The event loop that handles requests"""
        while True:
            with self.soft_watchdog.disable():
                recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            if isinstance(
                recv_obj,
                (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
            ):
                await self.output_processor.handle_batch_output(recv_obj)
            else:
                self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = real_time()
            self.soft_watchdog.feed()

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            remaining_rids = list(self.rid_to_state.keys())

            if self.server_status == ServerStatus.UnHealthy:
                # if health check failed, we should exit immediately
                logger.error(
                    "Signal SIGTERM received while health check failed. Force exiting."
                )
                self.request_log_manager.dump_requests_before_crash(
                    rid_to_state=self.rid_to_state
                )
                self.force_exit_handler()
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                # if force shutdown flag set, exit immediately
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting."
                )
                self.force_exit_handler()
                break

            logger.info(
                f"Gracefully exiting... Remaining number of requests {remain_num_req}. Remaining requests {remaining_rids=}."
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                self.request_log_manager.dump_requests_before_crash(
                    rid_to_state=self.rid_to_state
                )
                break

        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    def force_exit_handler(self):
        """Put some custom force exit logic here."""
        pass

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        self.send_to_scheduler.send_pyobj(ranks)

    def _set_default_priority(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        """Set the default priority value."""
        if (
            self.enable_priority_scheduling
            and obj.priority is None
            and self.default_priority_value is not None
        ):
            obj.priority = self.default_priority_value


class ServerStatus(Enum):
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(func.__self__, TokenizerManager):
            func.__self__.request_log_manager.dump_requests_before_crash(
                rid_to_state=func.__self__.rid_to_state,
            )
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


def _get_processor_wrapper(server_args):
    try:
        processor = get_processor(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            use_fast=not server_args.disable_fast_image_processor,
            tokenizer_backend=server_args.tokenizer_backend,
        )
    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:
            logger.info(
                f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
            )
            processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=True,
                tokenizer_backend=server_args.tokenizer_backend,
            )
        else:
            raise e
    return processor


def _determine_tensor_transport_mode(server_args: ServerArgs) -> TensorTransportMode:
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


class SignalHandler:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def sigterm_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True

    def running_phase_sigquit_handler(self, signum=None, frame=None):
        logger.error(
            f"SIGQUIT received. {signum=}, {frame=}. It usually means one child failed."
        )
        # Stop subprocess watchdog before killing processes to prevent false-positive
        # crash detection during normal shutdown
        if self.tokenizer_manager._subprocess_watchdog is not None:
            self.tokenizer_manager._subprocess_watchdog.stop()
        self.tokenizer_manager.request_log_manager.dump_requests_before_crash(
            rid_to_state=self.tokenizer_manager.rid_to_state,
        )
        kill_process_tree(os.getpid())


# Note: request abort handling logic
# We should handle all of the following cases correctly.
#
# | entrypoint | is_streaming | status          | abort engine    | cancel asyncio task   | rid_to_state                |
# | ---------- | ------------ | --------------- | --------------- | --------------------- | --------------------------- |
# | http       | yes          | validation      | background task | fast api              | del in _handle_abort_req    |
# | http       | yes          | waiting queue   | background task | fast api              | del in _handle_abort_req    |
# | http       | yes          | running         | background task | fast api              | del in _handle_batch_output |
# | http       | no           | validation      | http exception  | http exception        | del in _handle_abort_req    |
# | http       | no           | waiting queue   | type 1          | type 1 exception      | del in _handle_abort_req    |
# | http       | no           | running         | type 3          | type 3 exception      | del in _handle_batch_output |
#
