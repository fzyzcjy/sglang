# T1 block-forward parity fixture (GPU). NOT a test module (leading-underscore so
# CI's `test_*` glob skips it). Builds the PRODUCTION dsv4 draft block-forward and
# the EXTERNAL SoT oracle on identical inputs, plus the negative `force_causal_indices`
# seam. Imported by test_dsv4_block_forward_sot_parity.py.
#
# Design contract for the model+backend impl agents (this fixture is the executable
# spec the T1 guardrail runs against), per the FINAL agent-A contract:
#
#  * The backend exposes a NON-CAUSAL full-block index builder
#    ``DeepseekV4AttnBackend.get_dspark_swa_page_indices(*, seq_lens_casual,
#    req_pool_indices_repeated, out_loc, block_size) -> (swa_page_indices [num_q, K],
#    swa_topk_lengths [num_q])`` plus ``init_forward_metadata_dspark_draft_block``
#    (gamma-token, need_compress=False, SWA-only), gated by the ``is_dspark_draft``
#    flag. Per request it builds ``cat([arange(min(W, prefix+1)), W + arange(gamma)])``
#    shared across all gamma rows (no causal triangle), translated to SWA, padded to
#    64, invalid slots -1. This is the landing point of plan R1.
#  * The model exposes ``forward(input_ids, positions, forward_batch,
#    input_embeds=None) -> DSparkV4DraftOutput`` with ``.base_logits [bs*gamma,
#    org_vocab]`` (hc-collapsed + norm + lm_head + TP all_gather + org_vocab crop),
#    ``.draft_hidden [bs*gamma, hc, d]``, and ``.x_post_hc [bs*gamma, d] | None``.
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
from typing import Iterator, Sequence

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

    Parameters drive the three GPU-tier consumers:
      * T1 (SoT parity): ``exercise_non_causality`` constructs distinct per-position
        block KV so a causal regression diverges.
      * T3 (dynamic batch): ``prefix_lens`` / ``accept_lens`` build a mixed-length
        batch; ``batch_size`` / ``row_base_logits`` expose per-row slices for the
        run-alone independence check.
      * T4 (TP parity): ``num_heads`` / ``tp_size`` build the q-pad-gap config
        (n_local_heads not in {64,128} at TP=2).
    """

    def __init__(
        self,
        *,
        seed: int,
        exercise_non_causality: bool,
        device: torch.device,
        dtype: torch.dtype,
        prefix_lens: Sequence[int] | None,
        accept_lens: Sequence[int] | None,
        num_heads: int | None,
        tp_size: int,
    ) -> None:
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self._exercise_non_causality = exercise_non_causality
        self.prefix_lens = tuple(prefix_lens) if prefix_lens is not None else (8,)
        self.accept_lens = tuple(accept_lens) if accept_lens is not None else (1,)
        self.batch_size = len(self.prefix_lens)
        self.num_heads = num_heads
        self.tp_size = tp_size
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

    def row_base_logits(self, output: _ProductionBlockOutput, row: int) -> torch.Tensor:
        """Slice the [bs*gamma, vocab] base_logits down to one request's gamma rows."""
        return self._row_base_logits(output, row)

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

    def _row_base_logits(
        self, output: _ProductionBlockOutput, row: int
    ) -> torch.Tensor:  # pragma: no cover
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
    prefix_lens: Sequence[int] | None = None,
    accept_lens: Sequence[int] | None = None,
    num_heads: int | None = None,
    tp_size: int = 1,
) -> Dsv4BlockForwardHarness:
    """Build the production-vs-SoT block-forward harness (GPU) for T1/T3/T4.

    Raises HarnessUnavailable until the production contract is wired; the tests treat
    that as a clean skip.
    """
    return Dsv4BlockForwardHarness(
        seed=seed,
        exercise_non_causality=exercise_non_causality,
        device=torch.device(device),
        dtype=dtype,
        prefix_lens=prefix_lens,
        accept_lens=accept_lens,
        num_heads=num_heads,
        tp_size=tp_size,
    )
