import unittest

from sglang.benchmark.dspark_sps_profiler import (
    ServerContext,
    SpsRow,
    align_batch_sizes_to_dp,
    build_request_count_sweep,
    build_table_from_rounds,
    postprocess_round,
    resolve_cuda_graph_max_bs,
    validate_sweep_against_server,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_rows(
    *,
    num_rows: int = 30,
    num_running_reqs: int = 4,
    num_verify_tokens: int = 32,
    step_time: float = 0.01,
    first_forward_ct: int = 0,
) -> list[SpsRow]:
    return [
        SpsRow(
            forward_ct=first_forward_ct + index,
            num_running_reqs=num_running_reqs,
            num_verify_tokens=num_verify_tokens,
            step_time=step_time,
        )
        for index in range(num_rows)
    ]


def make_context(**overrides) -> ServerContext:
    values = dict(
        base_url="http://localhost:30000",
        tokenizer_path="dummy",
        tp_size=4,
        dp_size=1,
        verify_num_draft_tokens=8,
        cuda_graph_max_bs=128,
        skip_max_running_requests_threshold=float("inf"),
        skip_token_capacity_threshold=float("inf"),
    )
    values.update(overrides)
    return ServerContext(**values)


class TestPostprocessRound(CustomTestCase):
    def test_single_rank_round_builds_probe_from_median_step_time(self):
        """A steady single-rank round yields batch_tokens=bs*gamma1 and sps=1/median(dt)."""
        outcome = postprocess_round(
            rank_rows=[make_rows(step_time=0.01)],
            batch_size=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            client_result={},
        )
        self.assertEqual(outcome.batch_tokens, 32)
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)
        self.assertEqual(outcome.match_fraction, 1.0)

    def test_round_warmup_steps_are_dropped_from_timing(self):
        """The first aligned steps are excluded so ramp-in noise never enters the median."""
        slow_head = make_rows(num_rows=8, step_time=0.5, first_forward_ct=0)
        steady_tail = make_rows(num_rows=20, step_time=0.01, first_forward_ct=8)
        outcome = postprocess_round(
            rank_rows=[slow_head + steady_tail],
            batch_size=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            client_result={},
        )
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)

    def test_off_target_batch_rows_are_filtered_out(self):
        """Ramp-up rows at a smaller running batch are excluded from the probe."""
        ramp = make_rows(num_rows=10, num_running_reqs=2, num_verify_tokens=16)
        steady = make_rows(num_rows=30, first_forward_ct=10, step_time=0.02)
        outcome = postprocess_round(
            rank_rows=[ramp + steady],
            batch_size=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            client_result={},
        )
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertAlmostEqual(outcome.match_fraction, 0.75)

    def test_round_that_never_stabilizes_raises(self):
        """A round where almost no step hits the target batch is rejected."""
        rows = make_rows(num_rows=50, num_running_reqs=3, num_verify_tokens=24)
        rows += make_rows(num_rows=2, first_forward_ct=50)
        with self.assertRaisesRegex(RuntimeError, "never stabilized"):
            postprocess_round(
                rank_rows=[rows],
                batch_size=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                client_result={},
            )

    def test_batch_size_must_be_a_multiple_of_dp_size(self):
        """A per-system batch size not divisible by dp_size is a caller bug."""
        with self.assertRaises(ValueError):
            postprocess_round(
                rank_rows=[make_rows(), make_rows()],
                batch_size=5,
                dp_size=2,
                verify_num_draft_tokens=8,
                client_result={},
            )


class TestPostprocessRoundCrossRank(CustomTestCase):
    def test_two_uniform_ranks_average_their_step_times(self):
        """With dp=2 the per-step timing averages the two ranks' step times."""
        outcome = postprocess_round(
            rank_rows=[make_rows(step_time=0.01), make_rows(step_time=0.03)],
            batch_size=8,
            dp_size=2,
            verify_num_draft_tokens=8,
            client_result={},
        )
        self.assertEqual(outcome.batch_size_per_rank, 4)
        self.assertEqual(outcome.batch_tokens, 32)
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertEqual(len(outcome.per_rank_median_step_time), 2)
        self.assertAlmostEqual(outcome.per_rank_median_step_time[0], 0.01)
        self.assertAlmostEqual(outcome.per_rank_median_step_time[1], 0.03)

    def test_rank_with_no_new_records_raises(self):
        """A DP rank that produced no decode-step records fails the round loudly."""
        with self.assertRaisesRegex(RuntimeError, "no new decode-step records"):
            postprocess_round(
                rank_rows=[make_rows(), []],
                batch_size=8,
                dp_size=2,
                verify_num_draft_tokens=8,
                client_result={},
            )

    def test_disjoint_forward_ct_ranges_raise(self):
        """Ranks whose forward_ct counters never overlap are misaligned and rejected."""
        with self.assertRaisesRegex(RuntimeError, "no common forward_ct"):
            postprocess_round(
                rank_rows=[
                    make_rows(first_forward_ct=0),
                    make_rows(first_forward_ct=1000),
                ],
                batch_size=8,
                dp_size=2,
                verify_num_draft_tokens=8,
                client_result={},
            )

    def test_cross_rank_verify_token_mismatch_raises(self):
        """Ranks at the target batch but with different verify token counts are rejected."""
        with self.assertRaisesRegex(RuntimeError, "num_verify_tokens"):
            postprocess_round(
                rank_rows=[make_rows(), make_rows(num_verify_tokens=40)],
                batch_size=8,
                dp_size=2,
                verify_num_draft_tokens=8,
                client_result={},
            )

    def test_rank_count_mismatch_raises(self):
        """Getting records from fewer ranks than dp_size is rejected."""
        with self.assertRaisesRegex(RuntimeError, "DP ranks"):
            postprocess_round(
                rank_rows=[make_rows()],
                batch_size=8,
                dp_size=2,
                verify_num_draft_tokens=8,
                client_result={},
            )


class TestTableAssembly(CustomTestCase):
    def test_repeats_take_the_median_per_batch_tokens(self):
        """Multiple rounds at the same batch_tokens collapse to their median sps."""
        rounds = [
            postprocess_round(
                rank_rows=[make_rows(step_time=step_time)],
                batch_size=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                client_result={},
            )
            for step_time in (0.01, 0.02, 0.04)
        ]
        table = build_table_from_rounds(rounds=rounds, max_batch_tokens=None)
        self.assertEqual(table.sample_batch_tokens, [32])
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 50.0)

    def test_probes_are_sorted_by_batch_tokens(self):
        """Rounds swept out of order still produce a sorted probe grid."""
        rounds = [
            postprocess_round(
                rank_rows=[
                    make_rows(
                        num_running_reqs=batch_size,
                        num_verify_tokens=batch_size * 8,
                    )
                ],
                batch_size=batch_size,
                dp_size=1,
                verify_num_draft_tokens=8,
                client_result={},
            )
            for batch_size in (8, 2, 4)
        ]
        table = build_table_from_rounds(rounds=rounds, max_batch_tokens=None)
        self.assertEqual(table.sample_batch_tokens, [16, 32, 64])


class TestSweepHelpers(CustomTestCase):
    def test_align_rounds_up_to_dp_multiples_and_dedupes(self):
        """Batch sizes are rounded up to dp_size multiples and deduplicated."""
        self.assertEqual(
            align_batch_sizes_to_dp(batch_sizes=[1, 2, 4, 5, 8], dp_size=4),
            [4, 8],
        )

    def test_align_is_identity_for_dp1(self):
        """With dp_size=1 the sweep passes through unchanged (sorted, deduped)."""
        self.assertEqual(
            align_batch_sizes_to_dp(batch_sizes=[8, 1, 2], dp_size=1), [1, 2, 8]
        )

    def test_request_count_sweep_tapers_and_hits_the_max(self):
        """The default sweep starts at powers of two and always includes the max."""
        sweep = build_request_count_sweep(100)
        self.assertEqual(sweep[:4], [1, 2, 4, 8])
        self.assertEqual(sweep[-1], 100)
        self.assertIn(64, sweep)

    def test_request_count_sweep_rejects_non_positive_max(self):
        """A non-positive maximum request count raises."""
        with self.assertRaises(ValueError):
            build_request_count_sweep(0)

    def test_sweep_beyond_captured_cuda_graphs_raises(self):
        """A sweep exceeding the captured decode cuda-graph max bs is rejected."""
        with self.assertRaisesRegex(ValueError, "cuda graphs"):
            validate_sweep_against_server(
                context=make_context(cuda_graph_max_bs=64),
                batch_sizes=[8, 128],
            )

    def test_sweep_within_captured_cuda_graphs_passes(self):
        """A sweep whose per-rank max fits the captured graphs is accepted."""
        validate_sweep_against_server(
            context=make_context(cuda_graph_max_bs=64, dp_size=2),
            batch_sizes=[8, 128],
        )

    def test_resolve_cuda_graph_max_bs_prefers_captured_list(self):
        """The captured bs list wins over the max_bs field when both exist."""
        internal_state = {
            "cuda_graph_config": {"decode": {"bs": [1, 2, 160], "max_bs": 128}}
        }
        self.assertEqual(resolve_cuda_graph_max_bs(internal_state=internal_state), 160)

    def test_resolve_cuda_graph_max_bs_handles_missing_config(self):
        """A server without a parseable cuda_graph_config resolves to None."""
        self.assertIsNone(resolve_cuda_graph_max_bs(internal_state={}))


if __name__ == "__main__":
    unittest.main()
