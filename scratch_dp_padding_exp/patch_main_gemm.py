"""Round-4 instrumentation for main: [STEP] logging plus a per-module GEMM M probe.

Used only with --cuda-graph-backend-prefill disabled. Under a captured prefill
graph the Python hooks do not run during replay anyway, and the round-3
[STEP]/[PCG] logging was observed to make breakable capture fail, so this patch
deliberately omits the [PCG] hook.

Answers two questions with measurements rather than code reading:
  1. how many rows actually reach the attention projections under MAX_LEN vs
     SUM_LEN on main (via [STEP] local_rows and the q_proj/kv_a_proj row counts)
  2. whether MLA's ReplicatedLinear down-projections are recomputed once per
     attn-TP rank when attn_tp_size > 1

Env: SGLANG_DBG_DP_PAD = "" | heuristic | max, SGLANG_DBG_DP_LOG=1,
     SGLANG_DBG_GEMM_M=1

Run: python patch_main_gemm.py <path-to>/python/sglang
"""

from __future__ import annotations

import sys
from pathlib import Path

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

PROBE_TAIL = '''

# --------------------------------------------------- dp padding / gemm-M probe
import os as _probe_os

_PROBE_ON = _probe_os.environ.get("SGLANG_DBG_GEMM_M") == "1"
_PROBE_ROWS = {}
_PROBE_CALLS = {}
_PROBE_REAL = 0
_PROBE_PADDED = 0
_PROBE_STEPS = 0
_PROBE_LOGGER = __import__("logging").getLogger("gemmprobe")
_PROBE_INSTALLED = False


def probe_note_step(real_tokens: int, padded_rows: int) -> None:
    """One call per forward: this rank's real (pre-padding) local token count and
    the row count _pad_inputs_to_size padded the local tensors to."""
    global _PROBE_REAL, _PROBE_PADDED, _PROBE_STEPS
    if not _PROBE_ON:
        return
    _PROBE_REAL += int(real_tokens)
    _PROBE_PADDED += int(padded_rows)
    _PROBE_STEPS += 1
    if _PROBE_STEPS % 300 == 0:
        probe_dump()


def probe_dump() -> None:
    if not _PROBE_ON or _PROBE_REAL == 0:
        return
    rank = get_attention_dp_rank()
    waste = 100.0 * (_PROBE_PADDED - _PROBE_REAL) / _PROBE_REAL
    lines = [
        f"[PROBE] rank={rank} steps={_PROBE_STEPS} real={_PROBE_REAL} "
        f"padded_local_rows={_PROBE_PADDED} local_pad_waste={waste:+.2f}%"
    ]
    for name in sorted(_PROBE_ROWS, key=lambda k: -_PROBE_ROWS[k]):
        rows = _PROBE_ROWS[name]
        lines.append(
            f"[PROBE] rank={rank} module={name} rows={rows} "
            f"calls={_PROBE_CALLS[name]} avg_m={rows / _PROBE_CALLS[name]:.1f} "
            f"rows_per_real={rows / _PROBE_REAL:.3f}"
        )
    _PROBE_LOGGER.warning("\\n".join(lines))


def _probe_hook(module, args):
    if not args:
        return
    x = args[0]
    if not hasattr(x, "shape") or getattr(x, "ndim", 0) < 2:
        return
    key = getattr(module, "_probe_name", None) or type(module).__name__
    _PROBE_ROWS[key] = _PROBE_ROWS.get(key, 0) + int(x.shape[0])
    _PROBE_CALLS[key] = _PROBE_CALLS.get(key, 0) + 1


_PROBE_CLASS_HINTS = ("Linear", "Attention", "MLP", "MoE", "Experts", "Gate")


def probe_install(model) -> None:
    global _PROBE_INSTALLED
    if not _PROBE_ON or _PROBE_INSTALLED:
        return
    _PROBE_INSTALLED = True
    count = 0
    for name, sub in model.named_modules():
        cls = type(sub).__name__
        if not any(hint in cls for hint in _PROBE_CLASS_HINTS):
            continue
        parts = [p for p in name.split(".") if not p.isdigit()]
        sub._probe_name = f"{'.'.join(parts[-3:])}|{cls}"
        sub.register_forward_pre_hook(_probe_hook)
        count += 1
    _PROBE_LOGGER.warning("[PROBE] installed hooks on %d modules", count)
'''

FBI_ANCHOR = """        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
"""

FBI_REPLACEMENT = """        import os as _dbg_os

        from sglang.srt.layers.dp_attention import probe_note_step

        _dbg_raw = list(self.original_global_num_tokens_cpu or [])
        _dbg_rank = get_parallel().attn_dp_rank
        _dbg_real = _dbg_raw[_dbg_rank] if _dbg_rank < len(_dbg_raw) else 0
        if self.is_extend_in_batch:
            probe_note_step(_dbg_real, num_tokens)
        if _dbg_os.environ.get("SGLANG_DBG_DP_LOG") == "1" and self.is_extend_in_batch:
            global _DBG_STEP_N
            _DBG_STEP_N += 1
            if _DBG_STEP_N <= 4000:
                _DBG_STEP_LOGGER.warning(
                    "[STEP] n=%d rank=%d mode=%s raw=%s local_rows=%d buffer=%d fwd=%s",
                    _DBG_STEP_N,
                    _dbg_rank,
                    "MAX_LEN" if dp_padding_mode.is_max_len() else "SUM_LEN",
                    _dbg_raw,
                    num_tokens,
                    buffer_len,
                    self.forward_mode,
                )

        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
"""

FBI_TAIL = """

_DBG_STEP_N = 0
_DBG_STEP_LOGGER = __import__("logging").getLogger("dppad")
"""

MR_ANCHOR = "        self.load_model()\n"
MR_REPLACEMENT = (
    "        self.load_model()\n"
    "        from sglang.srt.layers.dp_attention import probe_install\n\n"
    "        probe_install(self.model)\n"
)


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

    apply(root / "srt/model_executor/model_runner.py", MR_ANCHOR, MR_REPLACEMENT)

    print("done")


if __name__ == "__main__":
    main()
