"""Patch installed sglang with a per-module row-count (GEMM M) probe.

Answers empirically: for every Linear-ish module in the model, how many rows
does it actually see per forward, compared with this DP rank's real token count?
A module whose rows/real_tokens ratio approaches dp_size is duplicated across
DP ranks; a ratio of 1 means it runs only on local tokens.

Env:
  SGLANG_DBG_GEMM_M=1     enable the probe
  SGLANG_DBG_DP_PAD=...   "" | heuristic | max (padding-mode override)
  SGLANG_DBG_DP_LOG=1     per-step [DPPAD] / [PCG] lines

Run: python patch_gemm_probe.py /sgl-workspace/sglang/python/sglang
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- 1. padding-mode override (same as patch_dpdbg.py) ---

DP_MODE_ANCHOR = """        if is_extend_in_batch and dp_size > 1:
            # Hybrid-SSM models materialize idle ranks via the MAX_LEN
            # fabricated-row conversion; other models keep mainline SUM_LEN.
            if get_flags().dp.max_len_with_idle and min(global_num_tokens) == 0:
                return DpPaddingMode.MAX_LEN
            return DpPaddingMode.SUM_LEN
"""

DP_MODE_REPLACEMENT = """        import os as _dbg_os

        _dbg_mode = _dbg_os.environ.get("SGLANG_DBG_DP_PAD", "")
        if _dbg_mode == "max" and dp_size > 1:
            return DpPaddingMode.MAX_LEN
        if _dbg_mode != "heuristic" and is_extend_in_batch and dp_size > 1:
            # Hybrid-SSM models materialize idle ranks via the MAX_LEN
            # fabricated-row conversion; other models keep mainline SUM_LEN.
            if get_flags().dp.max_len_with_idle and min(global_num_tokens) == 0:
                return DpPaddingMode.MAX_LEN
            return DpPaddingMode.SUM_LEN
"""

# --- 2. the probe itself, appended to dp_attention.py (imported by everything) ---

PROBE_TAIL = '''

# ---------------------------------------------------------------- gemm-M probe
import os as _probe_os

_PROBE_ON = _probe_os.environ.get("SGLANG_DBG_GEMM_M") == "1"
_PROBE_ROWS = {}
_PROBE_CALLS = {}
_PROBE_REAL_TOKENS = 0
_PROBE_STEPS = 0
_PROBE_LOGGER = __import__("logging").getLogger("gemmprobe")
_PROBE_INSTALLED = False


def probe_note_step(real_tokens: int) -> None:
    """Called once per forward from prepare_mlp_sync_batch with this rank's
    REAL (pre-padding) local token count."""
    global _PROBE_REAL_TOKENS, _PROBE_STEPS
    if not _PROBE_ON:
        return
    _PROBE_REAL_TOKENS += int(real_tokens)
    _PROBE_STEPS += 1
    if _PROBE_STEPS % 200 == 0:
        probe_dump()


def probe_dump() -> None:
    if not _PROBE_ON or _PROBE_REAL_TOKENS == 0:
        return
    import torch as _t

    rank = get_attention_dp_rank()
    lines = [
        f"[GEMMPROBE] rank={rank} steps={_PROBE_STEPS} "
        f"real_local_tokens={_PROBE_REAL_TOKENS}"
    ]
    for name in sorted(_PROBE_ROWS, key=lambda k: -_PROBE_ROWS[k]):
        rows = _PROBE_ROWS[name]
        lines.append(
            f"[GEMMPROBE] rank={rank} module={name} rows={rows} "
            f"calls={_PROBE_CALLS[name]} ratio={rows / _PROBE_REAL_TOKENS:.3f}"
        )
    _PROBE_LOGGER.warning("\\n".join(lines))


def _probe_key(module) -> str:
    """Collapse per-layer instances into one bucket keyed by role."""
    cls = type(module).__name__
    name = getattr(module, "_probe_name", None)
    return f"{name}|{cls}" if name else cls


def _probe_hook(module, args):
    if not args:
        return
    x = args[0]
    if not hasattr(x, "shape") or x.ndim < 2:
        return
    key = _probe_key(module)
    _PROBE_ROWS[key] = _PROBE_ROWS.get(key, 0) + int(x.shape[0])
    _PROBE_CALLS[key] = _PROBE_CALLS.get(key, 0) + 1


_PROBE_CLASS_HINTS = (
    "Linear",
    "Attention",
    "MLP",
    "MoE",
    "Experts",
    "Gate",
    "RMSNorm",
)


def probe_install(model) -> None:
    """Register forward-pre-hooks on every interesting submodule. Called from
    ModelRunner once the model is loaded."""
    global _PROBE_INSTALLED
    if not _PROBE_ON or _PROBE_INSTALLED:
        return
    _PROBE_INSTALLED = True
    count = 0
    for name, sub in model.named_modules():
        cls = type(sub).__name__
        if not any(hint in cls for hint in _PROBE_CLASS_HINTS):
            continue
        # Bucket layer 0 separately from the rest so per-layer noise is visible,
        # and strip the numeric layer index otherwise.
        parts = [p for p in name.split(".") if not p.isdigit()]
        sub._probe_name = ".".join(parts[-3:]) if parts else name
        sub.register_forward_pre_hook(_probe_hook)
        count += 1
    _PROBE_LOGGER.warning("[GEMMPROBE] installed hooks on %d modules", count)
'''

# --- 3. call probe_note_step + [DPPAD] logging from prepare_mlp_sync_batch ---

FBI_ANCHOR = """        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
"""

FBI_REPLACEMENT = """        import os as _dbg_os

        from sglang.srt.layers.dp_attention import probe_note_step

        _dbg_raw = list(self.original_global_num_tokens_cpu or [])
        _dbg_rank = get_parallel().attn_dp_rank
        probe_note_step(
            _dbg_raw[_dbg_rank] if _dbg_rank < len(_dbg_raw) else 0
        )
        if (
            _dbg_os.environ.get("SGLANG_DBG_DP_LOG") == "1"
            and self.is_extend_in_batch
            and max(global_num_tokens) >= 256
        ):
            global _DBG_DP_COUNT
            _DBG_DP_COUNT += 1
            if _DBG_DP_COUNT <= 3000:
                _DBG_DP_LOGGER.warning(
                    "[DPPAD] n=%d rank=%d mode=%s raw=%s final=%s local=%d buffer=%d mode_fwd=%s",
                    _DBG_DP_COUNT,
                    _dbg_rank,
                    "MAX_LEN" if dp_padding_mode.is_max_len() else "SUM_LEN",
                    _dbg_raw,
                    global_num_tokens,
                    num_tokens,
                    buffer_len,
                    self.forward_mode,
                )

        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
"""

FBI_TAIL = """

_DBG_DP_COUNT = 0
_DBG_DP_LOGGER = __import__("logging").getLogger("dppad")
"""

# --- 4. install hooks after model load ---

MR_ANCHOR = """        self.model_config = model_config
"""

MR_HOOK_ANCHOR = "        # Load the model\n"

# --- 5. log what the prefill CUDA graph actually pads to ---

PCG_ANCHOR = """        num_tokens = len(forward_batch.input_ids)
        static_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        self.raw_num_tokens = num_tokens
"""

PCG_REPLACEMENT = """        num_tokens = len(forward_batch.input_ids)
        static_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        self.raw_num_tokens = num_tokens

        import os as _dbg_os

        if _dbg_os.environ.get("SGLANG_DBG_DP_LOG") == "1":
            logger.warning(
                "[PCG] local_rows_in=%d bucket=%d mode=%s global_num_tokens=%s buffer=%s",
                num_tokens,
                static_num_tokens,
                forward_batch.dp_padding_mode,
                forward_batch.global_num_tokens_cpu,
                forward_batch.global_dp_buffer_len,
            )
"""


def apply(path: Path, anchor: str, replacement: str) -> None:
    text = path.read_text()
    if replacement in text:
        print(f"  already patched: {path.name}")
        return
    count = text.count(anchor)
    assert count == 1, f"anchor found {count}x in {path}"
    path.write_text(text.replace(anchor, replacement))
    print(f"  patched: {path.name}")


def append(path: Path, tail: str) -> None:
    text = path.read_text()
    if tail in text:
        print(f"  tail already present: {path.name}")
        return
    path.write_text(text + tail)
    print(f"  appended: {path.name}")


def main() -> None:
    root = Path(sys.argv[1])

    dp = root / "srt/layers/dp_attention.py"
    apply(dp, DP_MODE_ANCHOR, DP_MODE_REPLACEMENT)
    append(dp, PROBE_TAIL)

    fbi = root / "srt/model_executor/forward_batch_info.py"
    apply(fbi, FBI_ANCHOR, FBI_REPLACEMENT)
    append(fbi, FBI_TAIL)

    mr = root / "srt/model_executor/model_runner.py"
    text = mr.read_text()
    marker = "from sglang.srt.layers.dp_attention import probe_install"
    if marker not in text:
        anchor = "        self.load_model()\n"
        assert text.count(anchor) == 1, f"model_runner anchor count {text.count(anchor)}"
        text = text.replace(
            anchor,
            "        self.load_model()\n"
            "        from sglang.srt.layers.dp_attention import probe_install\n\n"
            "        probe_install(self.model)\n",
        )
        mr.write_text(text)
        print("  patched: model_runner.py")
    else:
        print("  already patched: model_runner.py")

    pcg = root / "srt/model_executor/runner/prefill_cuda_graph_runner.py"
    apply(pcg, PCG_ANCHOR, PCG_REPLACEMENT)

    print("done")


if __name__ == "__main__":
    main()
