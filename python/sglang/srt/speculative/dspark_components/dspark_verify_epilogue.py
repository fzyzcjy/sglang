from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.speculative.dspark_components.kernels.scatter_compact_to_strided import (
    scatter_compact_to_strided_into,
)


class DsparkVerifyEpilogue:
    """Compact->strided post-verify scatter, captured inside the token-keyed
    verify graph right after the target forward. The worker reads the static out
    buffers post-replay instead of launching the eager scatter; downstream accept
    / adjustments / inject are unchanged. RNG-free, so it applies to every
    compact replay regardless of sampling composition.

    Capture invariants:
    - verify_lens comes from the epilogue's OWN static buffer (worker fills it
      pre-replay); the capture-time uniform layout's tensors are capture-local,
      i.e. dead addresses at replay. The scatter bounds every read by
      verify_lens (<= stride each), so a stale tail cannot index out of the
      compact rows.
    - Out buffers are lazily allocated on the first warmup call (capture_one
      warms up twice before capturing), never in the graph memory pool.
    """

    def __init__(self, *, max_bs: int, verify_num_draft_tokens: int, device) -> None:
        self.max_bs = int(max_bs)
        self.stride = int(verify_num_draft_tokens)
        # Filled by the worker pre-replay; zero-init keeps a pre-fill idle
        # replay bounded.
        self.verify_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        # [max_bs*stride, dim], allocated on the first warmup call.
        self.strided_logits: Optional[torch.Tensor] = None
        self.strided_hidden: Optional[torch.Tensor] = None

    def fill_verify_lens(self, verify_lens: torch.Tensor) -> None:
        # The zeroed tail keeps padded-tier rows out of the scatter.
        bs = verify_lens.shape[0]
        self.verify_lens_buf[:bs].copy_(verify_lens)
        if bs < self.max_bs:
            self.verify_lens_buf[bs:].zero_()

    def _ensure_out(self, name: str, compact: torch.Tensor) -> torch.Tensor:
        buf = getattr(self, name)
        if (
            buf is None
            or buf.dtype != compact.dtype
            or buf.shape[1] != compact.shape[1]
        ):
            assert not torch.cuda.is_current_stream_capturing(), (
                "DsparkVerifyEpilogue output buffers must be allocated during "
                "warmup, not inside graph capture (pool memory is unreadable "
                "post-replay)."
            )
            buf = torch.empty(
                (self.max_bs * self.stride, compact.shape[1]),
                dtype=compact.dtype,
                device=compact.device,
            )
            setattr(self, name, buf)
        return buf

    def __call__(
        self,
        *,
        compact_logits: torch.Tensor,
        compact_hidden: torch.Tensor,
        bs: int,
    ) -> None:
        # bs is the padded capture-tier bs; the worker re-slices to the real
        # bs post-replay.
        logits_out = self._ensure_out("strided_logits", compact_logits)
        hidden_out = self._ensure_out("strided_hidden", compact_hidden)
        verify_lens = self.verify_lens_buf[:bs]
        scatter_compact_to_strided_into(
            compact=compact_logits,
            verify_lens=verify_lens,
            out=logits_out[: bs * self.stride],
            stride=self.stride,
            fill_value=0.0,
        )
        scatter_compact_to_strided_into(
            compact=compact_hidden,
            verify_lens=verify_lens,
            out=hidden_out[: bs * self.stride],
            stride=self.stride,
            fill_value=0.0,
        )
