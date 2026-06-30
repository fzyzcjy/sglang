import os
import types
import unittest

import torch

from sglang.srt.models.deepseek_v4_dspark import DSparkV4MarkovHead
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GAMMA = 3
_VOCAB = 17
_MARKOV_RANK = 8


def _ensure_dist_initialized() -> None:
    """Single-rank gloo distributed env + TP/PP/EP groups for the CPU markov head.

    DSparkV4MarkovHead.markov_w1 is a VocabParallelEmbedding whose forward calls
    get_tp_group(); even at tp=1 that asserts the model-parallel group exists, so
    the CPU test must initialize it before any markov-head forward runs.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29643")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


def _make_worker(*, draft_model, gamma: int = _GAMMA) -> DSparkWorkerV2:
    """A worker stub holding only the fields _sample_draft_block / accept read."""
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.gamma = gamma
    worker.verify_num_draft_tokens = gamma + 1
    worker.device = _DEVICE
    worker.draft_model = draft_model
    return worker


def _all_greedy_sampling_info(bs: int) -> types.SimpleNamespace:
    """top_k == 1 for every row -> the all-greedy (no-RNG) draft path."""
    return types.SimpleNamespace(
        top_ks=torch.ones(bs, dtype=torch.int64),
        temperatures=torch.ones(bs, 1, dtype=torch.float32),
        is_all_greedy=True,
        is_any_greedy=True,
    )


def _all_sampling_sampling_info(
    bs: int, *, temperature: float = 1.0
) -> types.SimpleNamespace:
    """top_k > 1 for every row -> the all-sampling (one multinomial per step) path."""
    return types.SimpleNamespace(
        top_ks=torch.full((bs,), 5, dtype=torch.int64),
        temperatures=torch.full((bs, 1), temperature, dtype=torch.float32),
        is_all_greedy=False,
        is_any_greedy=False,
    )


def _draft_probs_from_block(
    block_result, *, temperatures: torch.Tensor
) -> torch.Tensor:
    """Reproduce the accept-path draft_probs (dspark_worker_v2._accept_draft_tokens).

    draft_probs[:, k] = softmax(corrected_logits[:, k] / temperature). This IS the
    distribution q(s_k) the step-k token was sampled from (the Markov-corrected,
    temperature-scaled step logits), so this helper mirrors the production formula
    exactly.
    """
    return torch.softmax(
        block_result.corrected_logits.float() / temperatures[:, None, None],
        dim=-1,
    )


class _Dsv4DraftProbsContractMixin:
    """Shared assertions for the draft_probs losslessness contract (dsv4 + dense).

    Concrete subclasses provide a markov head via ``_build_markov_head``.
    """

    def _build_markov_head(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _draft_model(self, head):
        return types.SimpleNamespace(markov_head=head)

    def test_draft_probs_row_k_is_q_of_s_k(self) -> None:
        """draft_probs[:, k] equals q(s_k): softmax of the step-k corrected logits."""
        torch.manual_seed(1)
        bs = 2
        head = self._build_markov_head()
        worker = _make_worker(draft_model=self._draft_model(head))
        base_logits = torch.randn(bs, _GAMMA, _VOCAB)
        anchor = torch.randint(0, _VOCAB, (bs,), dtype=torch.int64)
        sampling_info = _all_greedy_sampling_info(bs)

        block = worker._sample_draft_block(
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=None,
            sampling_info=sampling_info,
        )
        draft_probs = _draft_probs_from_block(block, temperatures=block.temperatures)

        self.assertEqual(draft_probs.shape, (bs, _GAMMA, _VOCAB))
        for k in range(_GAMMA):
            expected = torch.softmax(
                block.corrected_logits[:, k].float() / block.temperatures[:, None],
                dim=-1,
            )
            torch.testing.assert_close(draft_probs[:, k], expected)

    def test_draft_probs_row_zero_uses_anchor_as_prev(self) -> None:
        """Row 0 is q(s_0) computed with the anchor as the previous token (no anchor row)."""
        torch.manual_seed(2)
        bs = 1
        head = self._build_markov_head()
        worker = _make_worker(draft_model=self._draft_model(head))
        base_logits = torch.randn(bs, _GAMMA, _VOCAB)
        anchor = torch.tensor([7], dtype=torch.int64)
        sampling_info = _all_greedy_sampling_info(bs)

        block = worker._sample_draft_block(
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=None,
            sampling_info=sampling_info,
        )

        # The row-0 corrected logits must equal base_logits[:, 0] biased by the
        # markov step using the ANCHOR token. A spurious extra anchor row would
        # shift everything by one and break this.
        with torch.no_grad():
            expected_row0 = head.apply_step_logits(
                base_logits[:, 0],
                token_ids=anchor,
                hidden_states=None,
            )
        torch.testing.assert_close(block.corrected_logits[:, 0], expected_row0)
        self.assertEqual(block.corrected_logits.shape[1], _GAMMA)
        self.assertEqual(block.draft_tokens.shape, (bs, _GAMMA))

    def test_greedy_block_draws_no_random_numbers(self) -> None:
        """All-greedy draft sampling must NOT advance the RNG (matches off/a+b path)."""
        torch.manual_seed(3)
        bs = 2
        head = self._build_markov_head()
        worker = _make_worker(draft_model=self._draft_model(head))
        base_logits = torch.randn(bs, _GAMMA, _VOCAB)
        anchor = torch.randint(0, _VOCAB, (bs,), dtype=torch.int64)
        sampling_info = _all_greedy_sampling_info(bs)

        rng_before = torch.get_rng_state()
        worker._sample_draft_block(
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=None,
            sampling_info=sampling_info,
        )
        rng_after = torch.get_rng_state()
        self.assertTrue(
            torch.equal(rng_before, rng_after),
            "all-greedy draft sampling drew random numbers; the RNG stream would "
            "diverge from the off/cutoff path and break byte-identical losslessness.",
        )

    def test_sampling_block_draws_one_multinomial_per_step(self) -> None:
        """All-sampling draft draws exactly gamma multinomials (one per step)."""
        torch.manual_seed(4)
        bs = 2
        head = self._build_markov_head()
        worker = _make_worker(draft_model=self._draft_model(head))
        base_logits = torch.randn(bs, _GAMMA, _VOCAB)
        anchor = torch.randint(0, _VOCAB, (bs,), dtype=torch.int64)
        sampling_info = _all_sampling_sampling_info(bs)

        multinomial_calls = {"n": 0}
        real_multinomial = torch.multinomial

        def _counting_multinomial(*args, **kwargs):
            multinomial_calls["n"] += 1
            return real_multinomial(*args, **kwargs)

        torch.multinomial = _counting_multinomial
        try:
            worker._sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=anchor,
                draft_hidden=None,
                sampling_info=sampling_info,
            )
        finally:
            torch.multinomial = real_multinomial

        self.assertEqual(
            multinomial_calls["n"],
            _GAMMA,
            "all-sampling draft must draw exactly one multinomial per step (gamma "
            "total) so the RNG draw count matches the a+b path.",
        )

    def test_draft_probs_normalized_per_step(self) -> None:
        """Each q(s_k) row is a proper distribution summing to 1 over the vocab."""
        torch.manual_seed(5)
        bs = 3
        head = self._build_markov_head()
        worker = _make_worker(draft_model=self._draft_model(head))
        base_logits = torch.randn(bs, _GAMMA, _VOCAB)
        anchor = torch.randint(0, _VOCAB, (bs,), dtype=torch.int64)
        sampling_info = _all_sampling_sampling_info(bs, temperature=0.7)

        block = worker._sample_draft_block(
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=None,
            sampling_info=sampling_info,
        )
        draft_probs = _draft_probs_from_block(block, temperatures=block.temperatures)
        sums = draft_probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums))


class TestDsv4WorkerDraftProbs(_Dsv4DraftProbsContractMixin, CustomTestCase):
    """draft_probs losslessness contract for the dsv4 (DSparkV4MarkovHead) head."""

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def _build_markov_head(self):
        torch.manual_seed(100)
        head = DSparkV4MarkovHead(vocab_size=_VOCAB, markov_rank=_MARKOV_RANK)
        head.eval()
        return head


class TestDenseWorkerDraftProbs(_Dsv4DraftProbsContractMixin, CustomTestCase):
    """draft_probs losslessness contract for the dense (VanillaMarkov) head."""

    def _build_markov_head(self):
        torch.manual_seed(200)
        head = VanillaMarkov(vocab_size=_VOCAB, markov_rank=_MARKOV_RANK)
        head.eval()
        return head


if __name__ == "__main__":
    unittest.main()
