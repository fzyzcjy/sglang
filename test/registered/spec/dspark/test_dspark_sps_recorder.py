import unittest

from sglang.srt.speculative.dspark_components.dspark_sps_recorder import (
    SpsDataRecorder,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_recorder(**kwargs) -> tuple[SpsDataRecorder, FakeClock]:
    clock = FakeClock()
    recorder = SpsDataRecorder(clock=clock, **kwargs)
    return recorder, clock


class TestSpsDataRecorderPairing(CustomTestCase):
    def test_first_step_emits_no_record(self):
        """A single observed step has no successor stamp, so nothing is emitted."""
        recorder, _ = make_recorder()
        recorder.observe_decode_step(
            forward_ct=1, num_running_reqs=4, num_verify_tokens=32
        )
        self.assertEqual(recorder.dump_records(), [])

    def test_dt_is_attributed_to_the_previous_step(self):
        """The gap between two stamps becomes the previous step's step_time."""
        recorder, clock = make_recorder()
        recorder.observe_decode_step(
            forward_ct=1, num_running_reqs=4, num_verify_tokens=32
        )
        clock.advance(0.02)
        recorder.observe_decode_step(
            forward_ct=2, num_running_reqs=5, num_verify_tokens=40
        )
        self.assertEqual(recorder.dump_records(), [[1, 4, 32, 0.02]])

    def test_consecutive_steps_emit_one_record_per_gap(self):
        """N observed steps emit N-1 records, each keyed by the earlier step."""
        recorder, clock = make_recorder()
        for forward_ct in range(1, 5):
            recorder.observe_decode_step(
                forward_ct=forward_ct, num_running_reqs=2, num_verify_tokens=16
            )
            clock.advance(0.01)
        records = recorder.dump_records()
        self.assertEqual([record[0] for record in records], [1, 2, 3])

    def test_non_decode_step_breaks_the_pairing(self):
        """A prefill/idle step between two decode steps suppresses the cross-gap record."""
        recorder, clock = make_recorder()
        recorder.observe_decode_step(
            forward_ct=1, num_running_reqs=4, num_verify_tokens=32
        )
        clock.advance(0.02)
        recorder.note_non_decode_step()
        clock.advance(0.02)
        recorder.observe_decode_step(
            forward_ct=3, num_running_reqs=4, num_verify_tokens=32
        )
        self.assertEqual(recorder.dump_records(), [])

    def test_oversized_gap_is_dropped(self):
        """A gap above max_step_interval is treated as a stall and not recorded."""
        recorder, clock = make_recorder(max_step_interval=0.5)
        recorder.observe_decode_step(
            forward_ct=1, num_running_reqs=4, num_verify_tokens=32
        )
        clock.advance(0.6)
        recorder.observe_decode_step(
            forward_ct=2, num_running_reqs=4, num_verify_tokens=32
        )
        clock.advance(0.1)
        recorder.observe_decode_step(
            forward_ct=3, num_running_reqs=4, num_verify_tokens=32
        )
        self.assertEqual([record[0] for record in recorder.dump_records()], [2])


class TestSpsDataRecorderBuffer(CustomTestCase):
    def test_ring_buffer_keeps_only_the_newest_records(self):
        """When more records than max_records arrive, the oldest are evicted."""
        recorder, clock = make_recorder(max_records=3)
        for forward_ct in range(1, 8):
            recorder.observe_decode_step(
                forward_ct=forward_ct, num_running_reqs=1, num_verify_tokens=8
            )
            clock.advance(0.01)
        self.assertEqual([record[0] for record in recorder.dump_records()], [4, 5, 6])

    def test_dump_is_non_destructive(self):
        """Dumping twice returns the same records; reading never drains the ring."""
        recorder, clock = make_recorder()
        recorder.observe_decode_step(
            forward_ct=1, num_running_reqs=4, num_verify_tokens=32
        )
        clock.advance(0.01)
        recorder.observe_decode_step(
            forward_ct=2, num_running_reqs=4, num_verify_tokens=32
        )
        self.assertEqual(recorder.dump_records(), recorder.dump_records())

    def test_rejects_non_positive_max_records(self):
        """max_records below 1 is a configuration error."""
        with self.assertRaises(ValueError):
            SpsDataRecorder(max_records=0)


if __name__ == "__main__":
    unittest.main()
