# T1 block-forward parity fixture (GPU). NOT a test module (leading-underscore so
# CI's `test_*` glob skips it). Builds the PRODUCTION dsv4 draft block-forward and
# the EXTERNAL SoT oracle on identical inputs, plus the negative `force_causal_indices`
# seam. Imported by test_dsv4_block_forward_sot_parity.py.
#
# Design contract for the model+backend impl agents (this fixture is the executable
# spec the T1 guardrail runs against):
#
#  * The backend exposes a NON-CAUSAL full-block index builder
#    ``DeepseekV4AttnBackend.get_dspark_swa_page_indices`` that, for the TARGET_VERIFY
#    draft block, hands every gamma query row the SAME ``[whole window ++ whole block]``
#    index set (no causal triangle), padded to a multiple of PAGE_INDEX_ALIGNED_SIZE
#    with -1 in the dead slots. This is the landing point of plan R1.
#  * The model exposes ``forward(input_embeds, positions, forward_batch) ->
#    DSparkV4DraftOutput`` with ``.draft_hidden`` / ``.base_logits`` (hc-collapsed
#    dsv4 logits) / optional ``.x_post_hc``.
#  * ``force_causal_indices()`` is a context manager that monkeypatches the production
#    builder to delegate to the CAUSAL ``get_swa_page_indices`` triangle, so the
#    negative test can prove the guardrail catches a causal regression.
#
# Until those land, ``build_dsv4_block_forward_harness`` raises HarnessUnavailable and
# the T1 test skips cleanly (it is the un-skipped successor to the old
# test_dsv4_worker_parity GPU stub).

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

import torch


class HarnessUnavailable(RuntimeError):
    """Raised when the production block-forward contract is not yet wired."""


@dataclass
class _ProductionBlockOutput:
    base_logits: torch.Tensor
    draft_hidden: torch.Tensor


class Dsv4BlockForwardHarness:
    """Drives the production dsv4 draft block-forward through the real backend.

    The production path runs the real ``DeepseekV4AttnBackend`` with the non-causal
    full-block index builder (so the actual GPU metadata builder executes - a pure
    CPU numerics oracle would pass even with causal indices and is useless here).
    ``run_sot`` runs the vendored external SoT oracle on the SAME inputs (target
    hidden, anchor, KV, positions, start_pos); the two base_logits must agree within
    the fp8 tolerance.
    """

    def __init__(
        self,
        *,
        seed: int,
        exercise_non_causality: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self._exercise_non_causality = exercise_non_causality
        self._build()

    @property
    def exercises_non_causality(self) -> bool:
        """True when the fixture places attention weight on LATER block positions.

        A query row at block position i must non-causally attend a position j > i
        whose KV differs materially, so a causal-triangle regression changes the
        output. The builder sets this by constructing distinct per-position block KV
        and (under exercise_non_causality) verifying the SoT softmax assigns
        non-trivial mass to the upper triangle.
        """
        return self._exercises_non_causality

    @property
    def stage(self):
        """The production DSpark stage whose attention weights the SoT oracle shares."""
        return self._stage

    def run_production(self) -> _ProductionBlockOutput:
        """Run the production block-forward (real backend) and return base_logits."""
        return self._run_production()

    def run_sot(self, oracle) -> torch.Tensor:
        """Run the external SoT oracle on the same inputs; return base_logits."""
        return self._run_sot(oracle)

    @contextlib.contextmanager
    def force_causal_indices(self) -> Iterator[None]:
        """Monkeypatch the production builder to the causal triangle (negative test)."""
        with self._force_causal_indices():
            yield

    # ------------------------------------------------------------------
    # Wiring against the production contract. Filled in once the model +
    # backend agents land DSparkV4DraftOutput + get_dspark_swa_page_indices.
    # ------------------------------------------------------------------

    def _build(self) -> None:
        raise HarnessUnavailable(
            "dsv4 block-forward harness not wired: requires the production "
            "DeepseekV4ForCausalLMDSpark.forward(forward_batch) -> DSparkV4DraftOutput "
            "and DeepseekV4AttnBackend.get_dspark_swa_page_indices (model+backend "
            "agents own them). The T1 guardrail skips until they land."
        )

    def _run_production(self) -> _ProductionBlockOutput:  # pragma: no cover
        raise HarnessUnavailable("harness not wired")

    def _run_sot(self, oracle) -> torch.Tensor:  # pragma: no cover
        raise HarnessUnavailable("harness not wired")

    @contextlib.contextmanager
    def _force_causal_indices(self) -> Iterator[None]:  # pragma: no cover
        raise HarnessUnavailable("harness not wired")
        yield


def build_dsv4_block_forward_harness(
    *,
    seed: int = 0,
    exercise_non_causality: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dsv4BlockForwardHarness:
    """Build the T1 production-vs-SoT block-forward harness (GPU).

    Raises HarnessUnavailable until the production contract is wired; the T1 test
    treats that as a clean skip.
    """
    return Dsv4BlockForwardHarness(
        seed=seed,
        exercise_non_causality=exercise_non_causality,
        device=torch.device(device),
        dtype=dtype,
    )
