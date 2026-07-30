"""Patch the installed sglang in-place with DP-padding-mode instrumentation.

Adds:
  * env `SGLANG_DBG_DP_PAD` = "" (stock) | "heuristic" (drop the forced-SUM_LEN
    condition from PR #10414) | "max" (always MAX_LEN when dp_size > 1)
  * env `SGLANG_DBG_DP_LOG` = "1" to log the per-forward padding decision
  * a log of the row count entering the attention block of layer 0

Run: python patch_dpdbg.py /sgl-workspace/sglang/python/sglang
"""

from __future__ import annotations

import sys
from pathlib import Path

DP_ATTENTION_ANCHOR = """        if is_extend_in_batch and dp_size > 1:
            # Hybrid-SSM models materialize idle ranks via the MAX_LEN
            # fabricated-row conversion; other models keep mainline SUM_LEN.
            if get_flags().dp.max_len_with_idle and min(global_num_tokens) == 0:
                return DpPaddingMode.MAX_LEN
            return DpPaddingMode.SUM_LEN
"""

DP_ATTENTION_REPLACEMENT = """        import os as _dbg_os

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

FBI_ANCHOR = """        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
"""

FBI_REPLACEMENT = """        import os as _dbg_os

        if (
            _dbg_os.environ.get("SGLANG_DBG_DP_LOG") == "1"
            and self.is_extend_in_batch
            and max(global_num_tokens) >= 256
        ):
            global _DBG_DP_COUNT
            _DBG_DP_COUNT += 1
            if _DBG_DP_COUNT <= 3000:
                _DBG_DP_LOGGER.warning(
                    "[DPPAD] n=%d rank=%d mode=%s raw=%s final=%s local=%d buffer=%d mode_fwd=%s extend_in_batch=%s",
                    _DBG_DP_COUNT,
                    get_parallel().attn_dp_rank,
                    "MAX_LEN" if dp_padding_mode.is_max_len() else "SUM_LEN",
                    list(self.original_global_num_tokens_cpu or []),
                    global_num_tokens,
                    num_tokens,
                    buffer_len,
                    self.forward_mode,
                    self.is_extend_in_batch,
                )

        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
"""

FBI_TAIL = """

_DBG_DP_COUNT = 0
_DBG_DP_LOGGER = __import__("logging").getLogger("dppad")
"""

QWEN3_ANCHOR = """        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )"""

QWEN3_REPLACEMENT = """        import os as _dbg_os

        if (
            self.layer_id == 0
            and _dbg_os.environ.get("SGLANG_DBG_DP_LOG") == "1"
            and forward_batch.is_extend_in_batch
        ):
            global _DBG_ATTN_COUNT
            _DBG_ATTN_COUNT += 1
            if _DBG_ATTN_COUNT <= 3000:
                logger.warning(
                    "[DPATTN] n=%d attn_rows=%d real_extend_tokens=%s mode=%s",
                    _DBG_ATTN_COUNT,
                    hidden_states.shape[0],
                    forward_batch.extend_seq_lens_cpu,
                    forward_batch.dp_padding_mode,
                )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )"""

QWEN3_TAIL = """

_DBG_ATTN_COUNT = 0
"""


def apply(path: Path, anchor: str, replacement: str) -> None:
    text = path.read_text()
    if replacement in text:
        print(f"  already patched: {path.name}")
        return
    count = text.count(anchor)
    assert count == 1, f"anchor found {count}x in {path}"
    path.write_text(text.replace(anchor, replacement))
    print(f"  patched: {path}")


def append(path: Path, tail: str) -> None:
    text = path.read_text()
    if tail in text:
        print(f"  tail already present: {path.name}")
        return
    path.write_text(text + tail)
    print(f"  appended tail: {path}")


def main() -> None:
    root = Path(sys.argv[1])

    apply(root / "srt/layers/dp_attention.py", DP_ATTENTION_ANCHOR, DP_ATTENTION_REPLACEMENT)

    fbi = root / "srt/model_executor/forward_batch_info.py"
    apply(fbi, FBI_ANCHOR, FBI_REPLACEMENT)
    append(fbi, FBI_TAIL)

    qwen = root / "srt/models/qwen3_moe.py"
    apply(qwen, QWEN3_ANCHOR, QWEN3_REPLACEMENT)
    append(qwen, QWEN3_TAIL)

    print("done")


if __name__ == "__main__":
    main()
