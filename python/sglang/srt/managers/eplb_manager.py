import logging
import time
from typing import TYPE_CHECKING, List

import torch.cuda
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._eplb_rebalance_layers_per_chunk = self._server_args.eplb_rebalance_layers_per_chunk

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            <= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be less than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._server_args.eplb_rebalance_num_iterations} iterations."
        )

    def on_forward_pass_end(self, forward_pass_id: int):
        if forward_pass_id % self._server_args.eplb_rebalance_num_iterations == 0:
            self.rebalance()

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._eplb_rebalance_layers_per_chunk is None

        if enable_timing:
            torch.cuda.synchronize()
            time_start = time.time()

        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )["logical_count"]
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

        if TODO:
            yield

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )
            if chunk_index != len(update_layer_ids_chunks) - 1:
                yield

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.cuda.synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(list(self._model_runner.model.routed_experts_weights_of_layer.keys()))

        chunk_size = self._eplb_rebalance_layers_per_chunk
        if chunk_size is None:
            chunk_size = 1000000

        return TODO
