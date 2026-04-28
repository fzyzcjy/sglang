# SPDX-License-Identifier: Apache-2.0
# Monkey-patch dynamo Python handlers from sglang side.
# Called from sglang.srt.entrypoints.engine.Engine.__init__ when
# SGLANG_HACK_PRINT_REQ_LIFECYCLE=1.
#
# Why monkey-patch instead of editing dynamo source:
#   - dynamo lives in a separate repo and ships as wheel; modifying source
#     requires rebuilding container.
#   - sglang's sglang.Engine is constructed BEFORE dynamo handlers are
#     instantiated (init_llm.py: `engine = sgl.Engine(...); handler = DecodeWorkerHandler(engine, ...)`),
#     so patching the handler class in Engine.__init__ is in time for all
#     subsequent instances.
#
# Effect: emit Path A entry (dyn_*_generate_enter), first/last chunk markers,
# and Path B exit (dyn_*_generate_exit) — covering the dynamo Python side
# without touching dynamo source.
import logging
from functools import wraps

from sglang.srt.utils.req_lifecycle import lc

logger = logging.getLogger(__name__)


def _extract_rid(request):
    if not isinstance(request, dict):
        return "?"
    return request.get("trace_id") or request.get("request_id") or "?"


def _patch_decode():
    try:
        from dynamo.sglang.request_handlers.llm.decode_handler import (
            DecodeWorkerHandler,
        )
    except ImportError:
        return False

    orig = DecodeWorkerHandler.generate

    @wraps(orig)
    async def patched(self, request, context):
        dyn_rid = _extract_rid(request)
        prompt_len = (
            len(request.get("token_ids") or [])
            if isinstance(request, dict)
            else 0
        )
        lc(dyn_rid, "dyn_decode_generate_enter", prompt_len=prompt_len)
        sgl_rid = None
        chunk_idx = 0
        try:
            async for out in orig(self, request, context):
                if chunk_idx == 0:
                    if isinstance(out, dict):
                        sgl_rid = (out.get("meta_info") or {}).get("id")
                    lc(sgl_rid or dyn_rid, "dyn_decode_first_chunk", dyn_rid=dyn_rid)
                chunk_idx += 1
                yield out
            lc(sgl_rid or dyn_rid, "dyn_decode_last_chunk", total_chunks=chunk_idx, dyn_rid=dyn_rid)
        finally:
            # Path B last gap: covers PyO3 boundary on the way out.
            lc(sgl_rid or dyn_rid, "dyn_decode_generate_exit",
               total_chunks=chunk_idx, dyn_rid=dyn_rid)

    DecodeWorkerHandler.generate = patched
    return True


def _patch_prefill():
    try:
        from dynamo.sglang.request_handlers.llm.prefill_handler import (
            PrefillWorkerHandler,
        )
    except ImportError:
        return False

    orig = PrefillWorkerHandler.generate

    @wraps(orig)
    async def patched(self, request, context):
        rid = _extract_rid(request)
        lc(rid, "dyn_prefill_generate_enter")
        first = True
        try:
            async for out in orig(self, request, context):
                if first:
                    room = None
                    if isinstance(out, dict):
                        bi = out.get("bootstrap_info") or {}
                        room = bi.get("bootstrap_room")
                    lc(rid, "dyn_prefill_first_chunk", bootstrap_room=room)
                    first = False
                yield out
            lc(rid, "dyn_prefill_last_chunk")
        finally:
            lc(rid, "dyn_prefill_generate_exit")

    PrefillWorkerHandler.generate = patched
    return True


def patch_dynamo_lifecycle():
    """Best-effort monkey-patch of dynamo handlers; safe to call when dynamo
    is not present."""
    n = 0
    try:
        if _patch_decode():
            n += 1
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[REQ_LIFECYCLE] patch decode failed: {exc!r}")
    try:
        if _patch_prefill():
            n += 1
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[REQ_LIFECYCLE] patch prefill failed: {exc!r}")
    if n:
        logger.info(
            f"[REQ_LIFECYCLE] dynamo handlers patched ({n} class(es))"
        )
    else:
        logger.info(
            "[REQ_LIFECYCLE] no dynamo handlers found, skipping patch"
        )
    return n
