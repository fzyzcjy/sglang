from __future__ import annotations

import bisect
import logging
from enum import Enum
from typing import Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class RaggedVerifyMode(str, Enum):
    # Named by what the verify forward computes:
    #   STATIC     — uniform gamma+1 block per request (default).
    #   CAP_ACCEPT — full block, caps accept at per-request ell_r (lossless harness).
    #   COMPACT    — only total = sum(1+ell_r) tokens (real-N, throughput gain).
    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"


def read_ragged_verify_mode() -> RaggedVerifyMode:
    value = envs.SGLANG_RAGGED_VERIFY_MODE.get()
    for mode in RaggedVerifyMode:
        if value == mode.value:
            return mode
    raise ValueError(
        f"invalid SGLANG_RAGGED_VERIFY_MODE={value!r}; expected one of "
        f"{', '.join(repr(m.value) for m in RaggedVerifyMode)}"
    )


def ragged_verify_compact_enabled() -> bool:
    return read_ragged_verify_mode() == RaggedVerifyMode.COMPACT


def round_up_grid(total: int, grid: Sequence[int]) -> int:
    if not grid:
        raise ValueError("round_up_grid requires a non-empty grid")
    if total > grid[-1]:
        raise ValueError(
            f"total {total} exceeds max grid tier {grid[-1]}; "
            "the caller must reject this batch before selecting a graph tier"
        )
    index = bisect.bisect_left(grid, total)
    return grid[index]


def build_graph_num_tokens_grid(
    *,
    capture_bs: Sequence[int],
    num_tokens_per_bs: int,
    token_grid: Sequence[int],
) -> list[int]:
    if num_tokens_per_bs < 1:
        raise ValueError(f"num_tokens_per_bs must be >= 1, got {num_tokens_per_bs}")
    uniform_tiers = {bs * num_tokens_per_bs for bs in capture_bs}
    merged = uniform_tiers.union(token_grid)
    if any(tier < 1 for tier in merged):
        raise ValueError(
            f"graph num-tokens grid tiers must be >= 1, got {sorted(merged)}"
        )
    return sorted(merged)


class RaggedVerifyLayout(msgspec.Struct, frozen=True):
    verify_lens: torch.Tensor
    verify_lens_cpu: list[int]
    total_verify_tokens: int
    extend_start_loc: torch.Tensor
    qo_indptr_device: torch.Tensor
    graph_num_tokens: int
    qo_indptr_host: Optional[torch.Tensor] = None
    kv_indptr_host: Optional[torch.Tensor] = None
    kv_lens_host: Optional[torch.Tensor] = None
    max_q_len: Optional[int] = None
    max_kv_len: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.verify_lens_cpu:
            raise ValueError("RaggedVerifyLayout requires at least one request")
        if min(self.verify_lens_cpu) < 1:
            raise ValueError(
                f"every request must verify the anchor (verify_len >= 1), got "
                f"{self.verify_lens_cpu}"
            )
        if self.total_verify_tokens != sum(self.verify_lens_cpu):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} != "
                f"sum(verify_lens_cpu) {sum(self.verify_lens_cpu)}"
            )
        if not (self.total_verify_tokens <= self.graph_num_tokens):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} exceeds "
                f"graph_num_tokens {self.graph_num_tokens}"
            )

    @property
    def bs(self) -> int:
        return len(self.verify_lens_cpu)

    @classmethod
    def from_verify_lens(
        cls,
        *,
        verify_lens_cpu: Sequence[int],
        device: torch.device,
        grid: Sequence[int],
        graph_num_tokens_floor: int = 0,
    ) -> RaggedVerifyLayout:
        verify_lens_list = [int(v) for v in verify_lens_cpu]
        total_verify_tokens = sum(verify_lens_list)
        # The token-keyed graph's bs-axis tensors are captured at a specific
        # capture_bs whose uniform full block is graph_num_tokens. A batch whose
        # real total rounds up to a smaller tier than bs * num_draft would select
        # a graph with too few request slots, so the bucket is floored to the
        # bs-derived full block (graph_num_tokens_floor) before rounding.
        bucket_input = max(total_verify_tokens, graph_num_tokens_floor)
        graph_num_tokens = round_up_grid(total=bucket_input, grid=grid)

        verify_lens = torch.tensor(verify_lens_list, dtype=torch.int32, device=device)
        cumsum = torch.cumsum(verify_lens, dim=0).to(torch.int32)
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        qo_indptr_device = torch.cat([zero, cumsum])
        extend_start_loc = qo_indptr_device[:-1].clone()

        return cls(
            verify_lens=verify_lens,
            verify_lens_cpu=verify_lens_list,
            total_verify_tokens=total_verify_tokens,
            extend_start_loc=extend_start_loc,
            qo_indptr_device=qo_indptr_device,
            graph_num_tokens=graph_num_tokens,
        )

    @classmethod
    def uniform(
        cls,
        *,
        bs: int,
        num_draft_tokens: int,
        device: torch.device,
        grid: Sequence[int],
    ) -> RaggedVerifyLayout:
        return cls.from_verify_lens(
            verify_lens_cpu=[num_draft_tokens] * bs,
            device=device,
            grid=grid,
        )

    def padded_to_bucket(self, *, num_draft_tokens: int) -> RaggedVerifyLayout:
        """Pad the layout to ``graph_num_tokens // num_draft_tokens`` requests
        whose verify_lens sum to exactly ``graph_num_tokens`` (the padded-token
        contract, §2.5).

        The token-keyed graph is captured at a fixed capture_bs and a frozen
        ``repeat_interleave(output_size=graph_num_tokens)``, so replay must feed a
        layout with that many request slots and that many total tokens. The
        shortfall ``graph_num_tokens - total_verify_tokens`` lands entirely in the
        token range ``[total_verify_tokens, graph_num_tokens)`` (after every real
        request), which the runner discards:

        - the appended synthetic requests (``padded_bs - bs`` of them) take a full
          ``num_draft_tokens`` block each and reference reserved req-pool slot 0;
        - any leftover is added to the very last entry's verify_len, so even when
          ``padded_bs == bs`` (no synthetic request, e.g. raw_bs already a
          capture_bs but the batch is ragged) the last real request's window is
          extended into the discarded tail.

        The worker's compact scatter keeps using the real (pre-pad) verify_lens,
        so the extended last request only adds discarded tail rows.
        """
        padded_bs = self.graph_num_tokens // num_draft_tokens
        shortfall = self.graph_num_tokens - self.total_verify_tokens
        if padded_bs == self.bs and shortfall == 0:
            return self

        assert padded_bs >= self.bs, (
            f"padded_bs {padded_bs} < bs {self.bs}: graph_num_tokens "
            f"{self.graph_num_tokens} cannot hold this batch's requests"
        )
        num_pad_reqs = padded_bs - self.bs
        padded_verify_lens_cpu = [
            *self.verify_lens_cpu,
            *([num_draft_tokens] * num_pad_reqs),
        ]
        leftover = self.graph_num_tokens - sum(padded_verify_lens_cpu)
        assert leftover >= 0, (
            f"negative leftover {leftover} padding to bucket "
            f"{self.graph_num_tokens} (bs={self.bs}, padded_bs={padded_bs})"
        )
        padded_verify_lens_cpu[-1] += leftover
        assert sum(padded_verify_lens_cpu) == self.graph_num_tokens
        assert min(padded_verify_lens_cpu) >= 1

        device = self.verify_lens.device
        verify_lens = torch.tensor(
            padded_verify_lens_cpu, dtype=torch.int32, device=device
        )
        cumsum = torch.cumsum(verify_lens, dim=0).to(torch.int32)
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        qo_indptr_device = torch.cat([zero, cumsum])
        extend_start_loc = qo_indptr_device[:-1].clone()

        return RaggedVerifyLayout(
            verify_lens=verify_lens,
            verify_lens_cpu=padded_verify_lens_cpu,
            total_verify_tokens=self.graph_num_tokens,
            extend_start_loc=extend_start_loc,
            qo_indptr_device=qo_indptr_device,
            graph_num_tokens=self.graph_num_tokens,
        )
