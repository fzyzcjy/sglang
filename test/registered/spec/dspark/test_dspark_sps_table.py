import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    SpsCostTable,
    load_sps_table_from_path,
    profile_sps_table,
)
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


class TestLoadSpsTableFromPath(CustomTestCase):
    def test_load_from_path_round_trips_table_and_lookup(self):
        """load_sps_table_from_path reads back a table written to a JSON file."""
        table = _make_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sps.json"
            path.write_text(table.to_json(), encoding="utf-8")
            loaded = load_sps_table_from_path(str(path))
        self.assertEqual(loaded.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(loaded.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(loaded.max_batch_tokens, table.max_batch_tokens)
        for batch_tokens in (1, 8, 31, 64, 200):
            self.assertEqual(loaded.lookup(batch_tokens), table.lookup(batch_tokens))


class TestFlatTableLookupIsConstant(CustomTestCase):
    def test_flat_table_lookup_is_one_for_any_batch(self):
        """The inert flat default (SPS=1.0) returns 1.0 for any B, so Theta = tau
        and the budget degenerates to verify-all-up-to-max."""
        flat = SpsCostTable(
            sample_batch_tokens=[1],
            sample_steps_per_sec=[1.0],
            max_batch_tokens=4096,
        )
        for batch_tokens in (0, 1, 2, 17, 256, 100_000):
            self.assertEqual(flat.lookup(batch_tokens), 1.0)


class TestProfileSpsTable(CustomTestCase):
    def test_profile_sorts_out_of_order_probes(self):
        """profile_sps_table sorts probes by batch_tokens into a valid table."""
        table = profile_sps_table(
            probes=[(32, 500.0), (8, 1000.0), (16, 950.0)],
        )
        self.assertEqual(table.sample_batch_tokens, [8, 16, 32])
        self.assertEqual(table.sample_steps_per_sec, [1000.0, 950.0, 500.0])

    def test_profile_passes_steps_per_sec_through_unchanged(self):
        """profile_sps_table stores each probe's steps_per_sec verbatim."""
        table = profile_sps_table(probes=[(4, 1234.5), (8, 678.25)])
        self.assertEqual(table.sample_steps_per_sec, [1234.5, 678.25])

    def test_profile_rejects_duplicate_batch_tokens(self):
        """profile_sps_table rejects duplicate batch_tokens (caller medians first)."""
        with self.assertRaises(ValueError):
            profile_sps_table(probes=[(8, 1000.0), (8, 900.0)])

    def test_profile_rejects_empty_probes(self):
        """profile_sps_table raises when given no probes."""
        with self.assertRaises(ValueError):
            profile_sps_table(probes=[])

    def test_profile_max_batch_tokens_defaults_to_largest_probe(self):
        """profile_sps_table defaults max_batch_tokens to the largest batch_tokens."""
        table = profile_sps_table(probes=[(8, 1000.0), (64, 480.0), (16, 950.0)])
        self.assertEqual(table.max_batch_tokens, 64)

    def test_profile_honors_explicit_max_batch_tokens(self):
        """profile_sps_table uses an explicit max_batch_tokens clamp bound."""
        table = profile_sps_table(
            probes=[(8, 1000.0), (16, 950.0)], max_batch_tokens=256
        )
        self.assertEqual(table.max_batch_tokens, 256)


def _make_bench_result(*, batch_size: int, output_throughput: float):
    """Build a BenchOneCaseResult fake with only the conversion-relevant fields set."""
    from sglang.benchmark.one_batch_server import BenchOneCaseResult

    output_len = 1024
    return BenchOneCaseResult(
        run_name="test",
        batch_size=batch_size,
        input_len=512,
        output_len=output_len,
        latency=1.0,
        input_throughput=1.0,
        output_throughput=output_throughput,
        overall_throughput=1.0,
        last_ttft=0.1,
        last_gen_throughput=output_throughput,
        acc_length=-1.0,
    )


class TestProfilerConversion(CustomTestCase):
    def _table_from_results(self, results):
        """Build + write + self-check a table from fake bench results, then reload it."""
        from sglang.benchmark import dspark_sps_profiler

        outcome = dspark_sps_profiler.build_sps_table(
            results=results, max_batch_tokens=None
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "sps.json"
            out_path.write_text(outcome.table.to_json(), encoding="utf-8")
            dspark_sps_profiler.run_self_check(out_path=out_path)
            return load_sps_table_from_path(str(out_path))

    def test_conversion_sets_batch_tokens_and_steps_per_sec(self):
        """batch_tokens = batch_size and steps_per_sec = output_throughput / batch_size."""
        table = self._table_from_results(
            [
                _make_bench_result(batch_size=2, output_throughput=1000.0),
                _make_bench_result(batch_size=4, output_throughput=1600.0),
                _make_bench_result(batch_size=8, output_throughput=2400.0),
            ]
        )
        self.assertEqual(table.sample_batch_tokens, [2, 4, 8])
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 500.0, places=6)
        self.assertAlmostEqual(table.sample_steps_per_sec[1], 400.0, places=6)
        self.assertAlmostEqual(table.sample_steps_per_sec[2], 300.0, places=6)

    def test_conversion_medians_across_repeats(self):
        """Repeats of the same batch size are medianed per batch_tokens."""
        # bs=4 yields steps_per_sec 250, 200, 300 across three repeats -> median 250.
        table = self._table_from_results(
            [
                _make_bench_result(batch_size=4, output_throughput=1000.0),
                _make_bench_result(batch_size=4, output_throughput=800.0),
                _make_bench_result(batch_size=4, output_throughput=1200.0),
            ]
        )
        self.assertEqual(table.sample_batch_tokens, [4])
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 250.0, places=6)

    def test_conversion_keeps_non_monotone_samples_without_crashing(self):
        """A non-monotone steps_per_sec sweep warns in self-check but does not crash."""
        # bs=8 has a higher steps_per_sec than bs=4 (non-monotone rise > 10%).
        table = self._table_from_results(
            [
                _make_bench_result(batch_size=4, output_throughput=1000.0),
                _make_bench_result(batch_size=8, output_throughput=8000.0),
            ]
        )
        self.assertEqual(table.sample_batch_tokens, [4, 8])
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 250.0, places=6)
        self.assertAlmostEqual(table.sample_steps_per_sec[1], 1000.0, places=6)

    def test_conversion_skips_degenerate_output_throughput(self):
        """Cases with output_throughput <= 0 are dropped, not turned into bad probes."""
        table = self._table_from_results(
            [
                _make_bench_result(batch_size=4, output_throughput=0.0),
                _make_bench_result(batch_size=8, output_throughput=2400.0),
            ]
        )
        self.assertEqual(table.sample_batch_tokens, [8])
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 300.0, places=6)


def _build_sps_cost_table_for(*, sps_table_path):
    from sglang.srt.speculative.dspark_components.dspark_scheduler import (
        build_sps_cost_table,
    )

    server_args = SimpleNamespace(
        speculative_dspark_sps_table_path=sps_table_path,
        max_running_requests=4,
    )
    return build_sps_cost_table(server_args=server_args, verify_num_draft_tokens=5)


class TestBuildSpsCostTableContract(CustomTestCase):
    def test_no_table_path_raises(self):
        """An enabled scheduler with no table path (None or "") raises ValueError naming the profiler and 'const'."""
        for sps_table_path in (None, ""):
            with self.assertRaises(ValueError) as cm:
                _build_sps_cost_table_for(sps_table_path=sps_table_path)
            message = str(cm.exception)
            self.assertIn("dspark_sps_profiler", message)
            self.assertIn("const", message)

    def test_const_sentinel_returns_flat_table(self):
        """The literal 'const' sentinel opts into a flat constant-SPS table (SPS=1.0)."""
        table = _build_sps_cost_table_for(sps_table_path="const")
        self.assertEqual(table.sample_steps_per_sec, [1.0])

    def test_real_path_loads_table(self):
        """A real table path loads the pre-profiled table back from its JSON file."""
        table = _make_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sps.json"
            path.write_text(table.to_json(), encoding="utf-8")
            loaded = _build_sps_cost_table_for(sps_table_path=str(path))
        self.assertEqual(loaded.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(loaded.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(loaded.max_batch_tokens, table.max_batch_tokens)


if __name__ == "__main__":
    unittest.main()
