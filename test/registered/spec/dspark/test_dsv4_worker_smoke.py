import types
import unittest

import torch

from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GAMMA = 3
_HIDDEN = 8


def _make_dense_draft_model() -> types.SimpleNamespace:
    """A dense draft model exposes none of the V4 capabilities."""
    return types.SimpleNamespace(confidence_head=None)


class _FakeV4DraftModel:
    """A draft model that exposes the V4 capability surface (forward_spec etc.)."""

    def __init__(self, *, confidence: torch.Tensor) -> None:
        self.confidence_head = object()
        self._confidence = confidence
        self.forward_spec_calls: list[dict] = []
        self.inject_calls: list[dict] = []
        self.attached: dict = {}

    def forward_spec(self, input_ids, main_hidden, start_pos=0, sampler=None):
        self.forward_spec_calls.append(
            {
                "input_ids": input_ids,
                "main_hidden": main_hidden,
                "start_pos": start_pos,
                "sampler": sampler,
            }
        )
        bs = input_ids.shape[0]
        sampled = torch.zeros((bs, _GAMMA), dtype=torch.long)
        for step in range(_GAMMA):
            step_logits = torch.zeros((bs, 16), dtype=torch.float32)
            step_logits[:, step + 1] = 1.0
            sampled[:, step] = sampler(step_logits, step)
        output_ids = torch.cat([input_ids.view(-1, 1), sampled], dim=1)
        corrected_logits = torch.zeros((bs, _GAMMA, 16), dtype=torch.float32)
        return output_ids, corrected_logits

    def inject_target_hidden(self, *, main_hidden, start_pos, is_prefill):
        self.inject_calls.append(
            {
                "main_hidden": main_hidden,
                "start_pos": start_pos,
                "is_prefill": is_prefill,
            }
        )
        return main_hidden

    def last_confidence(self):
        return self._confidence

    def attach_shared_modules(self, *, embed_tokens, lm_head):
        self.attached = {"embed_tokens": embed_tokens, "lm_head": lm_head}


def _make_worker(*, draft_model) -> DSparkWorkerV2:
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.gamma = _GAMMA
    worker.verify_num_draft_tokens = _GAMMA + 1
    worker.device = _DEVICE
    worker.draft_model = draft_model
    worker._draft_owns_block_forward = hasattr(draft_model, "forward_spec")
    worker._draft_owns_kv_injection = hasattr(draft_model, "inject_target_hidden")
    worker._draft_owns_confidence = hasattr(draft_model, "last_confidence")
    worker._verify_backend_self_adds_seq_lens_cache = None
    return worker


class TestDsv4CapabilityDetection(CustomTestCase):
    def test_v4_draft_model_sets_all_capability_flags(self) -> None:
        """A draft model exposing the V4 surface flips every capability flag True."""
        worker = _make_worker(
            draft_model=_FakeV4DraftModel(confidence=torch.full((2, _GAMMA), 0.5))
        )
        self.assertTrue(worker._draft_owns_block_forward)
        self.assertTrue(worker._draft_owns_kv_injection)
        self.assertTrue(worker._draft_owns_confidence)

    def test_dense_draft_model_leaves_all_capability_flags_false(self) -> None:
        """A dense draft model exposes no V4 capabilities, so all flags stay False."""
        worker = _make_worker(draft_model=_make_dense_draft_model())
        self.assertFalse(worker._draft_owns_block_forward)
        self.assertFalse(worker._draft_owns_kv_injection)
        self.assertFalse(worker._draft_owns_confidence)


class TestDsv4VerifyBackendSelfAdd(CustomTestCase):
    def _worker_with_backend(self, backend) -> DSparkWorkerV2:
        worker = _make_worker(draft_model=_make_dense_draft_model())
        worker._target_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(attn_backend=backend)
        )
        return worker

    def test_v4_family_verify_backend_self_adds(self) -> None:
        """A backend owning the raw target-verify metadata builder self-adds seq lens."""
        backend = types.SimpleNamespace(
            make_forward_metadata_from_raw_verify=lambda *a, **k: None
        )
        worker = self._worker_with_backend(backend)
        self.assertTrue(worker._verify_backend_self_adds_seq_lens())

    def test_dense_verify_backend_does_not_self_add(self) -> None:
        """A dense backend lacks the raw verify builder, so the worker keeps the pre-add."""
        backend = types.SimpleNamespace()
        worker = self._worker_with_backend(backend)
        self.assertFalse(worker._verify_backend_self_adds_seq_lens())

    def test_self_add_capability_is_cached(self) -> None:
        """The resolved self-add capability is cached after first resolution."""
        backend = types.SimpleNamespace(
            make_forward_metadata_from_raw_verify=lambda *a, **k: None
        )
        worker = self._worker_with_backend(backend)
        worker._verify_backend_self_adds_seq_lens()
        worker._target_worker.model_runner.attn_backend = types.SimpleNamespace()
        self.assertTrue(worker._verify_backend_self_adds_seq_lens())


class TestDsv4DraftForwardDispatch(CustomTestCase):
    def test_propose_routes_to_forward_spec_for_v4(self) -> None:
        """The worker delegates the draft block to forward_spec when the model owns it."""
        draft_model = _FakeV4DraftModel(confidence=torch.full((2, _GAMMA), 0.5))
        worker = _make_worker(draft_model=draft_model)
        bs = 2
        anchor = torch.tensor([4, 5], dtype=torch.int64)
        main_hidden = torch.randn(bs, _HIDDEN)
        draft_input = types.SimpleNamespace(
            bonus_tokens=anchor, hidden_states=main_hidden
        )
        batch = types.SimpleNamespace(seq_lens=torch.tensor([10, 10]))
        sampling_info = types.SimpleNamespace(
            top_ks=torch.tensor([1, 1]), temperatures=torch.tensor([[1.0], [1.0]])
        )

        proposal = worker._propose_draft_block_via_model(
            batch=batch,
            draft_input=draft_input,
            bs=bs,
            device=_DEVICE,
            sampling_info=sampling_info,
        )

        self.assertEqual(len(draft_model.forward_spec_calls), 1)
        call = draft_model.forward_spec_calls[0]
        self.assertEqual(call["start_pos"], 10)
        self.assertTrue(torch.equal(call["input_ids"], anchor))
        self.assertEqual(proposal.draft_block_ids.shape, (bs, _GAMMA))
        self.assertEqual(proposal.draft_block.draft_tokens.shape, (bs, _GAMMA))
        # anchor at column 0, greedy sampler picks token (step + 1) at each step.
        self.assertEqual(proposal.draft_block_ids[:, 0].tolist(), anchor.tolist())
        self.assertEqual(proposal.draft_block.draft_tokens[0].tolist(), [1, 2, 3])

    def test_forward_spec_none_during_decode_raises(self) -> None:
        """forward_spec returning None during decode is an error (block must run)."""
        draft_model = _FakeV4DraftModel(confidence=torch.full((1, _GAMMA), 0.5))
        draft_model.forward_spec = lambda *a, **k: None
        worker = _make_worker(draft_model=draft_model)
        draft_input = types.SimpleNamespace(
            bonus_tokens=torch.tensor([4], dtype=torch.int64),
            hidden_states=torch.randn(1, _HIDDEN),
        )
        batch = types.SimpleNamespace(seq_lens=torch.tensor([10]))
        sampling_info = types.SimpleNamespace(
            top_ks=torch.tensor([1]), temperatures=torch.tensor([[1.0]])
        )
        with self.assertRaises(RuntimeError):
            worker._propose_draft_block_via_model(
                batch=batch,
                draft_input=draft_input,
                bs=1,
                device=_DEVICE,
                sampling_info=sampling_info,
            )


class TestDsv4InjectDispatch(CustomTestCase):
    def test_inject_delegates_to_model_for_v4(self) -> None:
        """The worker delegates the whole KV injection to inject_target_hidden for V4."""
        draft_model = _FakeV4DraftModel(confidence=torch.full((2, _GAMMA), 0.5))
        worker = _make_worker(draft_model=draft_model)
        worker.model_runner = types.SimpleNamespace(device=_DEVICE)
        target_hidden = torch.randn(2, _HIDDEN)
        worker._inject_target_hidden_to_draft_kv(
            target_hidden=target_hidden,
            cache_loc=torch.tensor([0, 1]),
            positions=torch.tensor([10, 10]),
            start_pos=9,
            is_prefill=False,
        )
        self.assertEqual(len(draft_model.inject_calls), 1)
        call = draft_model.inject_calls[0]
        self.assertEqual(call["start_pos"], 9)
        self.assertFalse(call["is_prefill"])
        self.assertTrue(torch.equal(call["main_hidden"], target_hidden))

    def test_inject_skips_empty_hidden_for_v4(self) -> None:
        """An empty target hidden is a no-op (no delegation call)."""
        draft_model = _FakeV4DraftModel(confidence=torch.full((2, _GAMMA), 0.5))
        worker = _make_worker(draft_model=draft_model)
        worker.model_runner = types.SimpleNamespace(device=_DEVICE)
        worker._inject_target_hidden_to_draft_kv(
            target_hidden=torch.empty((0, _HIDDEN)),
            cache_loc=torch.empty((0,), dtype=torch.int64),
            positions=torch.empty((0,), dtype=torch.int64),
            start_pos=0,
            is_prefill=True,
        )
        self.assertEqual(len(draft_model.inject_calls), 0)


class TestDsv4ConfidenceRelay(CustomTestCase):
    def test_relays_model_confidence_for_v4(self) -> None:
        """The relay forwards last_confidence() and never computes the dense tap."""
        confidence = torch.full((2, _GAMMA), 0.4)
        draft_model = _FakeV4DraftModel(confidence=confidence)
        worker = _make_worker(draft_model=draft_model)
        stashed = {}

        def _capture(*, req_pool_indices, confidence):
            stashed["req"] = req_pool_indices
            stashed["confidence"] = confidence

        worker._stash_confidence = _capture

        def _fail(**kwargs):
            raise AssertionError("dense _compute_confidence must not run for V4")

        worker._compute_confidence = _fail

        worker._relay_confidence(
            req_pool_indices=torch.tensor([0, 1]),
            draft_hidden=None,
            anchor_tokens=torch.tensor([4, 5]),
            draft_tokens=torch.zeros((2, _GAMMA), dtype=torch.long),
        )
        self.assertTrue(torch.equal(stashed["confidence"], confidence))

    def test_falls_back_to_dense_compute_when_model_confidence_none(self) -> None:
        """When last_confidence() is None the relay uses the dense _compute_confidence."""
        draft_model = _FakeV4DraftModel(confidence=None)
        worker = _make_worker(draft_model=draft_model)
        stashed = {}
        computed = torch.full((2, _GAMMA), 0.9)

        worker._stash_confidence = (
            lambda *, req_pool_indices, confidence: stashed.update(
                confidence=confidence
            )
        )
        worker._compute_confidence = (
            lambda *, draft_hidden, anchor_tokens, draft_tokens: computed
        )

        worker._relay_confidence(
            req_pool_indices=torch.tensor([0, 1]),
            draft_hidden=torch.randn(2, _GAMMA, _HIDDEN),
            anchor_tokens=torch.tensor([4, 5]),
            draft_tokens=torch.zeros((2, _GAMMA), dtype=torch.long),
        )
        self.assertTrue(torch.equal(stashed["confidence"], computed))


class TestDsv4HiddenGeometry(CustomTestCase):
    def test_select_committed_hidden_picks_bonus_position(self) -> None:
        """The committed hidden per request is gathered at the correct_len index."""
        worker = _make_worker(
            draft_model=_FakeV4DraftModel(confidence=torch.full((2, _GAMMA), 0.5))
        )
        bs = 2
        window = _GAMMA + 1
        hidden = torch.arange(bs * window * _HIDDEN, dtype=torch.float32).view(
            bs * window, _HIDDEN
        )
        correct_len = torch.tensor([1, 3], dtype=torch.int32)
        committed = worker._select_committed_hidden(
            hidden=hidden, correct_len=correct_len, bs=bs
        )
        hidden_2d = hidden.view(bs, window, _HIDDEN)
        self.assertTrue(torch.equal(committed[0], hidden_2d[0, 1]))
        self.assertTrue(torch.equal(committed[1], hidden_2d[1, 3]))

    def test_select_prefill_last_hidden_picks_extend_boundaries(self) -> None:
        """The first-decode anchor hidden is gathered at each request's extend boundary."""
        worker = _make_worker(
            draft_model=_FakeV4DraftModel(confidence=torch.full((2, _GAMMA), 0.5))
        )
        total = 7
        hidden = torch.arange(total * _HIDDEN, dtype=torch.float32).view(total, _HIDDEN)
        last = worker._select_prefill_last_hidden(hidden=hidden, extend_lens=[3, 4])
        self.assertTrue(torch.equal(last[0], hidden[2]))
        self.assertTrue(torch.equal(last[1], hidden[6]))


if __name__ == "__main__":
    unittest.main()
