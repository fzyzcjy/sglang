from __future__ import annotations

import logging
import os
import sys

from sglang.srt.utils import kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


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
        # Late import: TokenizerManager imports this module, so resolving the
        # class name eagerly at module top would form a cycle.
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        if hasattr(func, "__self__") and isinstance(func.__self__, TokenizerManager):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)
