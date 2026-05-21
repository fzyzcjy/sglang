"""Default kwargs for spinning up an Engine in mock-model + canary mode.

Mock-model mode is testing-only; the default-filling logic lives here so the
main code (server_args) does not have to know about it.
"""

from __future__ import annotations

import os
from typing import Any


def mock_model_engine_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return Engine() kwargs that wire up mock-model + canary together.

    Defaults:
        load_format = "dummy"            (no real weights loaded)
        sampling_backend = "token_oracle" (gate for install_token_oracle_from_env)
        kv_canary = "raise"              (mock-model without canary is mostly pointless)
        cuda_graph_max_bs / max_running_requests / context_length / max_total_tokens
                                     keep canary capacities under the CUDA grid-safe ceiling

    Also sets ``SGLANG_KV_CANARY_INPUT_CHECK=1`` in the current process env so
    the canary's input-id verification path turns on when the engine starts.
    This is a side effect because input-check is mock-model-only and is no
    longer a server arg; the env var is the only injection path.

    ``SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE=1`` is also set so server_args
    accepts the test-only ``token_oracle`` sampling backend (the choice is
    env-gated to keep it out of production ``--sampling-backend --help``).

    ``SGLANG_KV_CANARY_INPUT_CHECK`` is enabled only when the effective
    sampling backend is ``token_oracle``. Tests that feed arbitrary input ids
    should use a normal sampling backend because the oracle expects its own
    deterministic ``(rid, position)`` token stream. When ``speculative_algorithm``
    is set, the default sampling backend switches to ``pytorch`` because the
    oracle can't predict draft-position tokens.

    Caller-supplied overrides win.
    """
    is_spec = "speculative_algorithm" in overrides
    default_sampling_backend = "pytorch" if is_spec else "token_oracle"
    effective_sampling_backend = overrides.get(
        "sampling_backend", default_sampling_backend
    )
    os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = (
        "1" if effective_sampling_backend == "token_oracle" else "0"
    )
    os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1"

    defaults: dict[str, Any] = {
        "load_format": "dummy",
        "sampling_backend": default_sampling_backend,
        "kv_canary": "raise",
        "cuda_graph_max_bs": 8,
        "max_running_requests": 32,
        "context_length": 2048,
        "max_total_tokens": 16384,
    }
    defaults.update(overrides)
    return defaults


def shutdown_engine_ignoring_zombie_reap(engine: Any) -> None:
    try:
        engine.shutdown()
    except RuntimeError as exc:
        if "not reaped within" not in str(exc):
            raise
