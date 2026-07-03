import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_info_dumper import (
    DecodeStepObservation,
    DsparkInfoDumper,
    InfoComponent,
    resolve_components,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_dumper(components, **kwargs):
    clock = FakeClock()
    dumper = DsparkInfoDumper(
        components=set(components),
        gamma=5,
        verify_num_draft_tokens=6,
        tp_rank=0,
        device=torch.device("cpu"),
        mode_value="static",
        clock=clock,
        **kwargs,
    )
    return dumper, clock


def make_obs(*, forward_ct, bs=4, num_verify_tokens=24):
    return DecodeStepObservation(
        forward_ct=forward_ct,
        bs=bs,
        mode="static",
        budget=100,
        lag_steps=0,
        num_verify_tokens=num_verify_tokens,
        verify_lens=torch.full((bs,), 6, dtype=torch.int32),
        confidence=torch.full((bs, 5), 0.9),
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        prefix_lens=torch.full((bs,), 128, dtype=torch.int64),
        draft_tokens=torch.zeros((bs, 5), dtype=torch.int64),
        bonus_tokens=torch.zeros((bs,), dtype=torch.int64),
        correct_len=torch.full((bs,), 3, dtype=torch.int32),
        cap_trim_lens=torch.zeros((bs,), dtype=torch.int32),
        commit_lens=torch.full((bs,), 4, dtype=torch.int32),
        rids=[f"r{i}" for i in range(bs)],
    )


class TestResolveComponents(CustomTestCase):
    def test_empty_disables(self):
        """No tokens means the dumper is disabled."""
        self.assertEqual(resolve_components(()), set())

    def test_all_expands_to_every_component(self):
        """The 'all' token selects every known component."""
        self.assertEqual(resolve_components(("all",)), set(InfoComponent))

    def test_subset_and_whitespace_are_kept(self):
        """A comma list keeps exactly the named components, stripped."""
        self.assertEqual(
            resolve_components((" core ", "reqs")),
            {InfoComponent.CORE, InfoComponent.REQS},
        )

    def test_unknown_component_raises(self):
        """An unknown component name is a configuration error."""
        with self.assertRaises(ValueError):
            resolve_components(("core", "bogus"))


class TestCoreAndCpuTiming(CustomTestCase):
    def test_disabled_dumper_records_nothing(self):
        """With no components enabled, observe is a no-op and dump returns None."""
        dumper, clock = make_dumper(set())
        dumper.begin_step()
        dumper.observe_decode_step(make_obs(forward_ct=1))
        self.assertIsNone(dumper.dump())

    def test_non_root_rank_is_disabled(self):
        """Only tp_rank 0 records, to avoid duplicated per-rank dumps."""
        clock = FakeClock()
        dumper = DsparkInfoDumper(
            components={"core"},
            gamma=5,
            verify_num_draft_tokens=6,
            tp_rank=1,
            device=torch.device("cpu"),
            mode_value="static",
            clock=clock,
        )
        self.assertFalse(dumper.enabled)
        dumper.observe_decode_step(make_obs(forward_ct=1))
        self.assertIsNone(dumper.dump())

    def test_one_record_per_step_including_the_last(self):
        """dump() flushes the pending step, so N observes yield N records."""
        dumper, clock = make_dumper({"core", "step_cpu_time"})
        for forward_ct in range(1, 4):
            dumper.observe_decode_step(make_obs(forward_ct=forward_ct))
            clock.advance(0.01)
        records = dumper.dump()["records"]
        self.assertEqual([r["forward_ct"] for r in records], [1, 2, 3])

    def test_step_cpu_ms_is_attributed_to_the_step_it_measures(self):
        """The gap before observe(S) is step S's cpu period, not step S-1's."""
        dumper, clock = make_dumper({"core", "step_cpu_time"})
        dumper.observe_decode_step(make_obs(forward_ct=1))
        clock.advance(0.02)
        dumper.observe_decode_step(make_obs(forward_ct=2))
        records = dumper.dump()["records"]
        first = next(r for r in records if r["forward_ct"] == 1)
        second = next(r for r in records if r["forward_ct"] == 2)
        self.assertNotIn("step_cpu_ms", first)
        self.assertAlmostEqual(second["step_cpu_ms"], 20.0, places=3)

    def test_core_fields_present(self):
        """The core component carries bs / mode / budget / verify-token counts."""
        dumper, _ = make_dumper({"core"})
        dumper.observe_decode_step(make_obs(forward_ct=7, bs=3, num_verify_tokens=18))
        record = dumper.dump()["records"][0]
        self.assertEqual(record["bs"], 3)
        self.assertEqual(record["num_running_reqs"], 3)
        self.assertEqual(record["num_verify_tokens"], 18)
        self.assertEqual(record["mode"], "static")

    def test_core_only_omits_timing_fields(self):
        """Selecting only core drops step_cpu_ms from the record."""
        dumper, clock = make_dumper({"core"})
        dumper.observe_decode_step(make_obs(forward_ct=1))
        clock.advance(0.01)
        dumper.observe_decode_step(make_obs(forward_ct=2))
        for record in dumper.dump()["records"]:
            self.assertNotIn("step_cpu_ms", record)

    def test_non_decode_step_resets_cpu_pairing(self):
        """A prefill/idle step drains pending and breaks the cpu-time gap."""
        dumper, clock = make_dumper({"core", "step_cpu_time"})
        dumper.observe_decode_step(make_obs(forward_ct=1))
        clock.advance(0.02)
        dumper.note_non_decode_step()
        clock.advance(0.02)
        dumper.observe_decode_step(make_obs(forward_ct=3))
        records = dumper.dump()["records"]
        self.assertEqual([r["forward_ct"] for r in records], [1, 3])
        for record in records:
            self.assertNotIn("step_cpu_ms", record)

    def test_oversized_gap_nulls_cpu_ms_but_keeps_record(self):
        """A stall above the threshold nulls step_cpu_ms without dropping the step."""
        dumper, clock = make_dumper({"core", "step_cpu_time"}, max_step_cpu_seconds=0.5)
        dumper.observe_decode_step(make_obs(forward_ct=1))
        clock.advance(0.6)
        dumper.observe_decode_step(make_obs(forward_ct=2))
        records = dumper.dump()["records"]
        self.assertEqual([r["forward_ct"] for r in records], [1, 2])
        second = next(r for r in records if r["forward_ct"] == 2)
        self.assertNotIn("step_cpu_ms", second)

    def test_ring_buffer_evicts_oldest(self):
        """max_records bounds the deque, evicting the oldest records."""
        dumper, clock = make_dumper({"core"}, max_records=3)
        for forward_ct in range(1, 8):
            dumper.observe_decode_step(make_obs(forward_ct=forward_ct))
            clock.advance(0.01)
        records = dumper.dump()["records"]
        self.assertEqual([r["forward_ct"] for r in records], [5, 6, 7])

    def test_dump_is_repeatable(self):
        """A second dump returns the same records without re-draining."""
        dumper, clock = make_dumper({"core"})
        dumper.observe_decode_step(make_obs(forward_ct=1))
        clock.advance(0.01)
        dumper.observe_decode_step(make_obs(forward_ct=2))
        self.assertEqual(dumper.dump(), dumper.dump())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for d2h staging")
class TestReqsAndGpuTiming(CustomTestCase):
    def _cuda_obs(self, *, forward_ct, bs=4):
        obs = make_obs(forward_ct=forward_ct, bs=bs)
        return DecodeStepObservation(
            forward_ct=obs.forward_ct,
            bs=obs.bs,
            mode=obs.mode,
            budget=obs.budget,
            lag_steps=obs.lag_steps,
            num_verify_tokens=obs.num_verify_tokens,
            verify_lens=obs.verify_lens.cuda(),
            confidence=obs.confidence.cuda(),
            req_pool_indices=obs.req_pool_indices.cuda(),
            prefix_lens=obs.prefix_lens.cuda(),
            draft_tokens=obs.draft_tokens.cuda(),
            bonus_tokens=obs.bonus_tokens.cuda(),
            correct_len=obs.correct_len.cuda(),
            cap_trim_lens=obs.cap_trim_lens.cuda(),
            commit_lens=obs.commit_lens.cuda(),
            rids=obs.rids,
        )

    def _make(self, components):
        return DsparkInfoDumper(
            components=set(components),
            gamma=5,
            verify_num_draft_tokens=6,
            tp_rank=0,
            device=torch.device("cuda"),
            mode_value="static",
        )

    def test_reqs_component_stages_per_request_detail(self):
        """The reqs component drains per-request tensors into the record."""
        dumper = self._make({"core", "reqs"})
        dumper.observe_decode_step(self._cuda_obs(forward_ct=1, bs=3))
        dumper.observe_decode_step(self._cuda_obs(forward_ct=2, bs=3))
        record = next(r for r in dumper.dump()["records"] if r["forward_ct"] == 1)
        self.assertEqual(len(record["reqs"]), 3)
        req = record["reqs"][0]
        self.assertEqual(req["rid"], "r0")
        self.assertEqual(req["verify_len"], 6)
        self.assertEqual(req["acc_len"], 4)
        self.assertEqual(req["correct_drafts"], 3)
        self.assertEqual(len(req["survival"]), 5)

    def test_gpu_timing_populates_segment_fields(self):
        """Whole-step / draft / target-verify GPU ms are read at drain."""
        dumper = self._make(
            {"step_gpu_time", "draft_gpu_time", "target_verify_gpu_time"}
        )
        for forward_ct in (1, 2):
            dumper.begin_step()
            with dumper.segment("draft"):
                torch.zeros(1024, device="cuda").sum()
            with dumper.segment("target_verify"):
                torch.zeros(1024, device="cuda").sum()
            dumper.observe_decode_step(self._cuda_obs(forward_ct=forward_ct))
        record = next(r for r in dumper.dump()["records"] if r["forward_ct"] == 1)
        self.assertGreaterEqual(record["step_gpu_ms"], 0.0)
        self.assertGreaterEqual(record["draft_gpu_ms"], 0.0)
        self.assertGreaterEqual(record["target_verify_gpu_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
