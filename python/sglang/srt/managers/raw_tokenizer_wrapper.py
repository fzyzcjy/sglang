from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
import os  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from enum import Enum  # noqa: F401
from typing import Any, Optional  # noqa: F401

from sglang.srt.configs.model_config import ModelConfig  # noqa: F401
from sglang.srt.environ import envs  # noqa: F401
from sglang.srt.managers.async_dynamic_batch_tokenizer import (  # noqa: F401
    AsyncDynamicbatchTokenizer,
)
from sglang.srt.managers.multimodal_processor import (  # noqa: F401
    get_mm_processor,
    import_processors,
)
from sglang.srt.server_args import ServerArgs  # noqa: F401
from sglang.srt.utils.hf_transformers_utils import (  # noqa: F401
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)

logger = logging.getLogger(__name__)


class InputFormat(Enum):
    """Input format types for tokenization handling."""

    SINGLE_STRING = 1
    BATCH_STRINGS = 2
    CROSS_ENCODER_PAIRS = 3


def _get_processor_wrapper(server_args: ServerArgs):
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


def _determine_tensor_transport_mode(server_args: ServerArgs):
    is_cross_node = server_args.dist_init_addr
    if is_cross_node:
        return "default"
    else:
        return "cuda_ipc"


@dataclass(frozen=True, slots=True, kw_only=True)
class RawTokenizerWrapper:
    """Owns tokenizer / processor / mm_processor / async_dynamic_batch_tokenizer."""

    tokenizer: Optional[Any]
    processor: Optional[Any]
    mm_processor: Optional[Any]
    async_dynamic_batch_tokenizer: Optional[AsyncDynamicbatchTokenizer]

    @classmethod
    def from_server_args(
        cls,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
    ) -> "RawTokenizerWrapper":
        if model_config.is_multimodal:
            import_processors("sglang.srt.multimodal.processors")
            if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
                import_processors(mm_process_pkg, overwrite=True)
            _processor = _get_processor_wrapper(server_args)
            transport_mode = _determine_tensor_transport_mode(server_args)
            mm_processor = get_mm_processor(
                model_config.hf_config,
                server_args,
                _processor,
                transport_mode,
                model_config=model_config,
            )
            if server_args.skip_tokenizer_init:
                tokenizer = processor = None
            else:
                processor = _processor
                tokenizer = get_tokenizer_from_processor(processor)
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            mm_processor = processor = None
            if server_args.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
        if (
            server_args.enable_dynamic_batch_tokenizer
            and not server_args.skip_tokenizer_init
        ):
            async_dynamic_batch_tokenizer = AsyncDynamicbatchTokenizer(
                tokenizer,
                max_batch_size=server_args.dynamic_batch_tokenizer_batch_size,
                batch_wait_timeout_s=server_args.dynamic_batch_tokenizer_batch_timeout,
            )
        else:
            async_dynamic_batch_tokenizer = None
        return cls(
            tokenizer=tokenizer,
            processor=processor,
            mm_processor=mm_processor,
            async_dynamic_batch_tokenizer=async_dynamic_batch_tokenizer,
        )
