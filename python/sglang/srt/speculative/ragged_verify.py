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
    OFF = "off"
    CUTOFF_ONLY = "cutoff-only"
    FULL = "full"


_RAGGED_VERIFY_OFF_ALIASES = frozenset({"", "off"})


def read_ragged_verify_mode() -> RaggedVerifyMode:
    value = envs.SGLANG_RAGGED_VERIFY.get()
    if value in _RAGGED_VERIFY_OFF_ALIASES:
        return RaggedVerifyMode.OFF
    if value == RaggedVerifyMode.CUTOFF_ONLY.value:
        return RaggedVerifyMode.CUTOFF_ONLY
    if value == RaggedVerifyMode.FULL.value:
        return RaggedVerifyMode.FULL
    raise ValueError(
        "invalid SGLANG_RAGGED_VERIFY="
        f"{value!r}; expected one of 'off', 'cutoff-only', 'full'"
    )


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
    ) -> RaggedVerifyLayout:
        verify_lens_list = [int(v) for v in verify_lens_cpu]
        total_verify_tokens = sum(verify_lens_list)
        graph_num_tokens = round_up_grid(total=total_verify_tokens, grid=grid)

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
