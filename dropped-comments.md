# Dropped comments — kv_canary cleanup pass (2026-05-20)

Format: `path:LINE_RANGE — first 80 chars of dropped text…`

## python/sglang/srt/kv_canary/

- python/sglang/srt/kv_canary/api.py:33-56 — install_canary docstring with numbered Steps 1..7 list narrating function body, plus "caller is responsible" sentence
- python/sglang/srt/kv_canary/endpoint.py:76-78 — launch_per_forward docstring restating "Call canary_verify_step then canary_write_step against this endpoint's canary_buf..."
- python/sglang/srt/kv_canary/endpoint.py:132-134 — launch_sweep docstring restating "Call only canary_verify_step against this endpoint's canary_buf..."
