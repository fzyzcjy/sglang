# SPDX-License-Identifier: Apache-2.0
# Hack: per-request lifecycle prints for diagnosing dynamo+sglang stall.
# Gated by env SGLANG_HACK_PRINT_REQ_LIFECYCLE=1. Zero overhead when off.
#
# Format (one line per event, scalar metadata only):
#   [REQ_LIFECYCLE] rid=<id> ts_ns=<int> event=<name> [k=v]...
#
# Post-process: grep + sort by ts_ns; join cross-process events by rid or
# bootstrap_room.
import os
import sys
import time

_ON = os.environ.get("SGLANG_HACK_PRINT_REQ_LIFECYCLE", "0") not in (
    "0",
    "",
    "false",
    "False",
)


def lc(rid, event, **kvs):
    if not _ON:
        return
    parts = [
        f"[REQ_LIFECYCLE] rid={rid} ts_ns={time.time_ns()} event={event}"
    ]
    for k, v in kvs.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), file=sys.stdout)


def is_on():
    return _ON
