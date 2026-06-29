import unittest

from sglang.srt.speculative.dspark_sps_table import SpsCostTable, profile_sps_table
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[8, 16, 32, 64],
        sample_steps_per_sec=[1000.0, 950.0, 500.0, 480.0],
        max_batch_tokens=128,
    )


class TestSpsCostTableInvariants(CustomTestCase):
    def test_rejects_non_increasing_batch_tokens(self):
        """__post_init__ rejects non strictly-increasing sample_batch_tokens."""
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 8, 16],
                sample_steps_per_sec=[1.0, 2.0, 3.0],
                max_batch_tokens=16,
            )

    def test_rejects_unsorted_batch_tokens(self):
        """__post_init__ rejects unsorted sample_batch_tokens."""
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[16, 8],
                sample_steps_per_sec=[1.0, 2.0],
                max_batch_tokens=16,
            )

    def test_rejects_length_mismatch(self):
        """__post_init__ rejects mismatched probe/SPS list lengths."""
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 16],
                sample_steps_per_sec=[1.0],
                max_batch_tokens=16,
            )

    def test_rejects_empty_table(self):
        """__post_init__ rejects an empty probe list."""
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[],
                sample_steps_per_sec=[],
                max_batch_tokens=0,
            )

    def test_rejects_max_below_largest_probe(self):
        """__post_init__ rejects max_batch_tokens below the largest probe."""
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 16],
                sample_steps_per_sec=[1.0, 2.0],
                max_batch_tokens=15,
            )


class TestSpsCostTableLookup(CustomTestCase):
    def test_lookup_exact_probe_returns_that_sps(self):
        """lookup at an exact captured probe returns that probe's SPS."""
        table = _make_table()
        self.assertEqual(table.lookup(8), 1000.0)
        self.assertEqual(table.lookup(16), 950.0)
        self.assertEqual(table.lookup(32), 500.0)
        self.assertEqual(table.lookup(64), 480.0)

    def test_lookup_floors_to_lower_captured_probe(self):
        """lookup floors B down to the largest captured probe (no interpolation)."""
        table = _make_table()
        self.assertEqual(table.lookup(31), 950.0)
        self.assertEqual(table.lookup(63), 500.0)

    def test_lookup_does_not_interpolate_across_cliff(self):
        """lookup preserves the hardware cliff rather than linearly interpolating."""
        table = _make_table()
        midpoint = table.lookup((16 + 32) // 2)
        self.assertEqual(midpoint, 950.0)
        self.assertNotEqual(midpoint, (950.0 + 500.0) / 2)

    def test_lookup_below_first_probe_clamps_to_first(self):
        """lookup below the smallest probe clamps to the first SPS."""
        table = _make_table()
        self.assertEqual(table.lookup(1), 1000.0)
        self.assertEqual(table.lookup(7), 1000.0)

    def test_lookup_above_last_probe_clamps_to_last(self):
        """lookup above the largest probe clamps to the last SPS."""
        table = _make_table()
        self.assertEqual(table.lookup(65), 480.0)
        self.assertEqual(table.lookup(10_000), 480.0)


class TestSpsCostTableJsonRoundTrip(CustomTestCase):
    def test_json_round_trip_preserves_table(self):
        """to_json followed by from_json reproduces an equal table."""
        table = _make_table()
        restored = SpsCostTable.from_json(table.to_json())
        self.assertEqual(restored.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(restored.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(restored.max_batch_tokens, table.max_batch_tokens)

    def test_json_round_trip_preserves_lookup_behavior(self):
        """A round-tripped table looks up identically to the original."""
        table = _make_table()
        restored = SpsCostTable.from_json(table.to_json())
        for batch_tokens in (1, 8, 31, 64, 200):
            self.assertEqual(restored.lookup(batch_tokens), table.lookup(batch_tokens))


class TestProfileSpsTable(CustomTestCase):
    def test_profile_builds_increasing_batch_tokens(self):
        """profile_sps_table emits probes at B = num_requests * (1 + gamma)."""

        def fake_timer(worker: object, num_requests: int, gamma: int) -> float:
            return 0.001 * num_requests

        table = profile_sps_table(
            target_worker=object(),
            probe_request_counts=[1, 2, 4],
            gamma=7,
            iters=3,
            time_uniform_verify_step=fake_timer,
        )
        self.assertEqual(table.sample_batch_tokens, [8, 16, 32])
        self.assertEqual(table.max_batch_tokens, 32)
        self.assertEqual(len(table.sample_steps_per_sec), 3)

    def test_profile_rejects_invalid_gamma(self):
        """profile_sps_table raises for gamma < 1."""
        with self.assertRaises(ValueError):
            profile_sps_table(
                target_worker=object(),
                probe_request_counts=[1],
                gamma=0,
                iters=1,
                time_uniform_verify_step=lambda w, r, g: 0.001,
            )

    def test_profile_rejects_invalid_iters(self):
        """profile_sps_table raises for iters < 1."""
        with self.assertRaises(ValueError):
            profile_sps_table(
                target_worker=object(),
                probe_request_counts=[1],
                gamma=7,
                iters=0,
                time_uniform_verify_step=lambda w, r, g: 0.001,
            )

    def test_profile_takes_median_across_iters(self):
        """profile_sps_table uses the median duration across iters per probe."""
        durations = iter([0.01, 0.001, 0.02])

        def jittery_timer(worker: object, num_requests: int, gamma: int) -> float:
            return next(durations)

        table = profile_sps_table(
            target_worker=object(),
            probe_request_counts=[1],
            gamma=7,
            iters=3,
            time_uniform_verify_step=jittery_timer,
        )
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 1.0 / 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
