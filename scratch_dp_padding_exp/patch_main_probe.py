"""Round-3 instrumentation, written against upstream main (3c9efaf3e1).

Goal: count the rows that actually reach the attention projections, including
prefill-CUDA-graph bucket padding, so MAX_LEN and SUM_LEN can be compared on
wasted attention work rather than on reasoning about the code.

Two log families, joined by a shared per-forward counter `n`:

  [STEP] n=.. rank=.. mode=.. raw=.. final=.. local_rows=.. buffer=.. fwd=..
         bcg_ok=..
      emitted from prepare_mlp_sync_batch. `local_rows` is what
      _pad_inputs_to_size pads the local tensors to; `raw` is the real
      pre-padding per-rank token counts.

  [PCG] n=.. local_rows_in=.. bucket=.. mode=.. global_num_tokens=.. buffer=..
      emitted from prefill_cuda_graph_runner.load_batch, i.e. only when the
      prefill CUDA graph is actually used. `bucket` is the captured shape the
      local tensors get padded up to, so it -- not `local_rows` -- is the real
      attention row count for graphed steps.

Rows reaching attention for a step = bucket if a [PCG] line exists for that n,
else local_rows. Waste = that minus raw[rank].

Env: SGLANG_DBG_DP_PAD = "" | heuristic | max, SGLANG_DBG_DP_LOG=1.

Run: python patch_main_probe.py <path-to>/python/sglang
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

STEP_COUNTER_TAIL = '''

# ------------------------------------------------------- dp padding step probe
_DBG_STEP = [0]


def dbg_current_step() -> int:
    return _DBG_STEP[0]
'''

FBI_ANCHOR = """        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
"""

FBI_REPLACEMENT = """        import os as _dbg_os

        from sglang.srt.layers.dp_attention import _DBG_STEP

        _DBG_STEP[0] += 1
        if _dbg_os.environ.get("SGLANG_DBG_DP_LOG") == "1" and self.is_extend_in_batch:
            _DBG_STEP_LOGGER.warning(
                "[STEP] n=%d rank=%d mode=%s raw=%s final=%s local_rows=%d "
                "buffer=%d fwd=%s bcg_ok=%s",
                _DBG_STEP[0],
                get_parallel().attn_dp_rank,
                "MAX_LEN" if dp_padding_mode.is_max_len() else "SUM_LEN",
                list(self.original_global_num_tokens_cpu or []),
                global_num_tokens,
                num_tokens,
                buffer_len,
                self.forward_mode,
                self.can_run_dp_breakable_cuda_graph,
            )

        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
"""

FBI_TAIL = """

_DBG_STEP_LOGGER = __import__("logging").getLogger("dppad")
"""

PCG_ANCHOR = """        static_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
"""

PCG_REPLACEMENT = """        static_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)

        import os as _dbg_os

        if _dbg_os.environ.get("SGLANG_DBG_DP_LOG") == "1":
            from sglang.srt.layers.dp_attention import dbg_current_step

            logger.warning(
                "[PCG] n=%d local_rows_in=%d bucket=%d mode=%s "
                "global_num_tokens=%s buffer=%s",
                dbg_current_step(),
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
    append(dp, STEP_COUNTER_TAIL)

    fbi = root / "srt/model_executor/forward_batch_info.py"
    apply(fbi, FBI_ANCHOR, FBI_REPLACEMENT)
    append(fbi, FBI_TAIL)

    pcg = root / "srt/model_executor/runner/prefill_cuda_graph_runner.py"
    apply(pcg, PCG_ANCHOR, PCG_REPLACEMENT)

    print("done")


if __name__ == "__main__":
    main()
