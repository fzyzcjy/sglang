"""Tests for arbitrary (non-standard) dim layouts.

Tensor dim_specs in real usage can be any valid identifiers, not just the
canonical "b s h d".  Examples: "whatever_1 b s whatever_2", "x y(tp) z",
"expert_idx token hidden(tp)", etc.  These tests verify that the unshard
pipeline handles such layouts correctly — both planning and round-trip
execution.
"""

import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    AxisInfo,
    ConcatParams,
    PickParams,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis, parse_dims
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


def _name_tensors(
    tensors: list[torch.Tensor], dim_names: list[str]
) -> list[torch.Tensor]:
    return [t.refine_names(*dim_names) for t in tensors]


class TestArbitraryDimPlanner:
    """Planner produces correct dim_name for non-standard dim names."""

    def test_single_dim_sharded(self) -> None:
        """Single dim 'foo(tp)' → concat on dim_name='foo'."""
        dim_specs = parse_dims("foo(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=2)} for i in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 1
        assert isinstance(plans[0].params, ConcatParams)
        assert plans[0].params.dim_name == "foo"

    def test_sharded_at_dim0(self) -> None:
        """'hidden(tp) feature' → concat on dim_name='hidden' (first position)."""
        dim_specs = parse_dims("hidden(tp) feature")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 1
        assert isinstance(plans[0].params, ConcatParams)
        assert plans[0].params.dim_name == "hidden"

    def test_sharded_at_last_dim(self) -> None:
        """'a b c d last(tp)' → concat on dim_name='last'."""
        dim_specs = parse_dims("a b c d last(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=2)} for i in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 1
        assert isinstance(plans[0].params, ConcatParams)
        assert plans[0].params.dim_name == "last"

    def test_sharded_in_middle_of_many_dims(self) -> None:
        """'whatever_1 b s(cp) whatever_2 h(tp) whatever_3' → CP dim_name='s', TP dim_name='h'."""
        dim_specs = parse_dims("whatever_1 b s(cp) whatever_2 h(tp) whatever_3")
        parallel_infos = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=cp, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=tp, axis_size=2),
            }
            for cp in range(2)
            for tp in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        concat_plans = [p for p in plans if isinstance(p.params, ConcatParams)]
        assert len(concat_plans) == 2

        cp_plan = next(p for p in concat_plans if p.axis == ParallelAxis.CP)
        tp_plan = next(p for p in concat_plans if p.axis == ParallelAxis.TP)
        assert cp_plan.params.dim_name == "s"
        assert tp_plan.params.dim_name == "h"

    def test_five_plain_dims_one_sharded(self) -> None:
        """'x0 x1 x2 x3(tp) x4' → concat on dim_name='x3'."""
        dim_specs = parse_dims("x0 x1 x2 x3(tp) x4")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=3)} for i in range(3)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 1
        assert isinstance(plans[0].params, ConcatParams)
        assert plans[0].params.dim_name == "x3"

    def test_underscored_names(self) -> None:
        """Dim names with underscores: 'batch_size seq_len(cp) num_heads(tp) head_dim'."""
        dim_specs = parse_dims("batch_size seq_len(cp) num_heads(tp) head_dim")
        parallel_infos = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=cp, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=tp, axis_size=2),
            }
            for cp in range(2)
            for tp in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        concat_plans = [p for p in plans if isinstance(p.params, ConcatParams)]
        cp_plan = next(p for p in concat_plans if p.axis == ParallelAxis.CP)
        tp_plan = next(p for p in concat_plans if p.axis == ParallelAxis.TP)
        assert cp_plan.params.dim_name == "seq_len"
        assert tp_plan.params.dim_name == "num_heads"

    def test_mixed_sharded_replicated_arbitrary_names(self) -> None:
        """'alpha beta(ep) gamma delta(tp)' with CP replicated → Pick(CP) + Concat(EP) + Concat(TP)."""
        dim_specs = parse_dims("alpha beta(ep) gamma delta(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp in range(2):
            for ep in range(2):
                for tp in range(2):
                    parallel_infos.append(
                        {
                            ParallelAxis.CP: AxisInfo(axis_rank=cp, axis_size=2),
                            ParallelAxis.EP: AxisInfo(axis_rank=ep, axis_size=2),
                            ParallelAxis.TP: AxisInfo(axis_rank=tp, axis_size=2),
                        }
                    )
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        pick_plans = [p for p in plans if isinstance(p.params, PickParams)]
        concat_plans = [p for p in plans if isinstance(p.params, ConcatParams)]
        assert len(pick_plans) == 1
        assert pick_plans[0].axis == ParallelAxis.CP

        assert len(concat_plans) == 2
        ep_plan = next(p for p in concat_plans if p.axis == ParallelAxis.EP)
        tp_plan = next(p for p in concat_plans if p.axis == ParallelAxis.TP)
        assert ep_plan.params.dim_name == "beta"
        assert tp_plan.params.dim_name == "delta"


class TestArbitraryDimExecutor:
    """E2E round-trip: shard a tensor → unshard → verify reconstruction."""

    def test_single_arbitrary_dim_tp2(self) -> None:
        """Single dim 'neurons(tp)' TP=2: shard on dim=0 and reconstruct."""
        torch.manual_seed(42)
        full_tensor = torch.randn(8)
        shards = _name_tensors(list(full_tensor.chunk(2, dim=0)), ["neurons"])

        dim_specs = parse_dims("neurons(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=2)} for i in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        with warning_sink.context() as warnings:
            result = execute_unsharder_plan(plans[0], shards)
        assert len(result) == 1
        assert torch.allclose(result[0].rename(None), full_tensor)
        assert warnings == []

    def test_sharded_at_dim0_5d(self) -> None:
        """'hidden(tp) a b c d' TP=2: shard on dim=0 of a 5-D tensor."""
        torch.manual_seed(42)
        full_tensor = torch.randn(8, 2, 3, 4, 5)
        dim_names: list[str] = ["hidden", "a", "b", "c", "d"]
        shards = _name_tensors(list(full_tensor.chunk(2, dim=0)), dim_names)

        dim_specs = parse_dims("hidden(tp) a b c d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=2)} for i in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        with warning_sink.context() as warnings:
            current: list[torch.Tensor] = shards
            for plan in plans:
                current = execute_unsharder_plan(plan, current)
        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)
        assert warnings == []

    def test_sharded_at_last_dim_5d(self) -> None:
        """'a b c d last(tp)' TP=4: shard on dim=4 (last) of a 5-D tensor."""
        torch.manual_seed(42)
        full_tensor = torch.randn(2, 3, 4, 5, 8)
        dim_names: list[str] = ["a", "b", "c", "d", "last"]
        shards = _name_tensors(list(full_tensor.chunk(4, dim=4)), dim_names)

        dim_specs = parse_dims("a b c d last(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        with warning_sink.context() as warnings:
            current: list[torch.Tensor] = shards
            for plan in plans:
                current = execute_unsharder_plan(plan, current)
        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)
        assert warnings == []

    def test_whatever_1_b_s_whatever_2_cp_tp(self) -> None:
        """'whatever_1 b s(cp) whatever_2(tp)' CP=2 TP=2: shards at dim=2 and dim=3."""
        torch.manual_seed(42)
        full_tensor = torch.randn(3, 4, 8, 12)
        dim_names: list[str] = ["whatever_1", "b", "s", "whatever_2"]

        cp_chunks = list(full_tensor.chunk(2, dim=2))
        shard_map: dict[tuple[int, int], torch.Tensor] = {}
        for cp_rank in range(2):
            tp_chunks = list(cp_chunks[cp_rank].chunk(2, dim=3))
            for tp_rank in range(2):
                shard_map[(cp_rank, tp_rank)] = tp_chunks[tp_rank]

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp_rank in range(2):
            for tp_rank in range(2):
                tensors.append(shard_map[(cp_rank, tp_rank)])
                parallel_infos.append(
                    {
                        ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                        ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                    }
                )

        dim_specs = parse_dims("whatever_1 b s(cp) whatever_2(tp)")
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current = _name_tensors(tensors, dim_names)
        with warning_sink.context() as warnings:
            for plan in plans:
                current = execute_unsharder_plan(plan, current)
        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)
        assert warnings == []

    def test_six_dims_sharded_at_positions_1_and_4(self) -> None:
        """'prefix expert(ep) mid0 mid1 head(tp) suffix' EP=2 TP=3."""
        torch.manual_seed(42)
        full_tensor = torch.randn(2, 6, 3, 4, 9, 5)
        dim_names: list[str] = ["prefix", "expert", "mid0", "mid1", "head", "suffix"]

        ep_chunks = list(full_tensor.chunk(2, dim=1))
        shard_map: dict[tuple[int, int], torch.Tensor] = {}
        for ep_rank in range(2):
            tp_chunks = list(ep_chunks[ep_rank].chunk(3, dim=4))
            for tp_rank in range(3):
                shard_map[(ep_rank, tp_rank)] = tp_chunks[tp_rank]

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for ep_rank in range(2):
            for tp_rank in range(3):
                tensors.append(shard_map[(ep_rank, tp_rank)])
                parallel_infos.append(
                    {
                        ParallelAxis.EP: AxisInfo(axis_rank=ep_rank, axis_size=2),
                        ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=3),
                    }
                )

        dim_specs = parse_dims("prefix expert(ep) mid0 mid1 head(tp) suffix")
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current = _name_tensors(tensors, dim_names)
        with warning_sink.context() as warnings:
            for plan in plans:
                current = execute_unsharder_plan(plan, current)
        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)
        assert warnings == []

    def test_three_sharded_axes_arbitrary_names(self) -> None:
        """'dim_a(ep) dim_b(cp) dim_c dim_d(tp)' EP=2 CP=2 TP=2 with scrambled ranks."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 6, 8, 10)
        dim_names: list[str] = ["dim_a", "dim_b", "dim_c", "dim_d"]

        ep_chunks = list(full_tensor.chunk(2, dim=0))
        shard_map: dict[tuple[int, int, int], torch.Tensor] = {}
        for ep_rank in range(2):
            cp_chunks = list(ep_chunks[ep_rank].chunk(2, dim=1))
            for cp_rank in range(2):
                tp_chunks = list(cp_chunks[cp_rank].chunk(2, dim=3))
                for tp_rank in range(2):
                    shard_map[(ep_rank, cp_rank, tp_rank)] = tp_chunks[tp_rank]

        scrambled_assignment: list[tuple[int, int, int]] = [
            (1, 0, 1),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 0),
            (0, 0, 1),
            (1, 1, 1),
        ]

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for ep_rank, cp_rank, tp_rank in scrambled_assignment:
            tensors.append(shard_map[(ep_rank, cp_rank, tp_rank)])
            parallel_infos.append(
                {
                    ParallelAxis.EP: AxisInfo(axis_rank=ep_rank, axis_size=2),
                    ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                    ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                }
            )

        dim_specs = parse_dims("dim_a(ep) dim_b(cp) dim_c dim_d(tp)")
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 3

        current = _name_tensors(tensors, dim_names)
        with warning_sink.context() as warnings:
            for plan in plans:
                current = execute_unsharder_plan(plan, current)
        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)
        assert warnings == []

    def test_replicated_with_arbitrary_names(self) -> None:
        """'alpha beta gamma' (no sharded dims) with TP=2 CP=2 → all replicated."""
        torch.manual_seed(42)
        full_tensor = torch.randn(3, 4, 5)
        dim_names: list[str] = ["alpha", "beta", "gamma"]

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp in range(2):
            for tp in range(2):
                tensors.append(full_tensor.clone())
                parallel_infos.append(
                    {
                        ParallelAxis.CP: AxisInfo(axis_rank=cp, axis_size=2),
                        ParallelAxis.TP: AxisInfo(axis_rank=tp, axis_size=2),
                    }
                )

        dim_specs = parse_dims("alpha beta gamma")
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2
        assert all(isinstance(p.params, PickParams) for p in plans)

        current = _name_tensors(tensors, dim_names)
        with warning_sink.context() as warnings:
            for plan in plans:
                current = execute_unsharder_plan(plan, current)
        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)
        assert warnings == []

    def test_mixed_replicated_sharded_arbitrary_6d(self) -> None:
        """'a0 a1(ep) a2 a3(tp) a4 a5' with CP replicated, 6-D tensor."""
        torch.manual_seed(42)
        full_tensor = torch.randn(2, 4, 3, 8, 5, 6)
        dim_names: list[str] = ["a0", "a1", "a2", "a3", "a4", "a5"]

        ep_chunks = list(full_tensor.chunk(2, dim=1))
        shard_map: dict[tuple[int, int], torch.Tensor] = {}
        for ep_rank in range(2):
            tp_chunks = list(ep_chunks[ep_rank].chunk(2, dim=3))
            for tp_rank in range(2):
                shard_map[(ep_rank, tp_rank)] = tp_chunks[tp_rank]

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp in range(2):
            for ep in range(2):
                for tp in range(2):
                    tensors.append(shard_map[(ep, tp)].clone())
                    parallel_infos.append(
                        {
                            ParallelAxis.CP: AxisInfo(axis_rank=cp, axis_size=2),
                            ParallelAxis.EP: AxisInfo(axis_rank=ep, axis_size=2),
                            ParallelAxis.TP: AxisInfo(axis_rank=tp, axis_size=2),
                        }
                    )

        dim_specs = parse_dims("a0 a1(ep) a2 a3(tp) a4 a5")
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        pick_plans = [p for p in plans if isinstance(p.params, PickParams)]
        concat_plans = [p for p in plans if isinstance(p.params, ConcatParams)]
        assert len(pick_plans) == 1
        assert pick_plans[0].axis == ParallelAxis.CP
        assert len(concat_plans) == 2

        current = _name_tensors(tensors, dim_names)
        with warning_sink.context() as warnings:
            for plan in plans:
                current = execute_unsharder_plan(plan, current)
        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)
        assert warnings == []

    def test_1d_tensor_single_dim(self) -> None:
        """Simplest case: 1-D tensor 'vec(tp)' TP=4."""
        torch.manual_seed(42)
        full_tensor = torch.randn(16)
        shards = _name_tensors(list(full_tensor.chunk(4, dim=0)), ["vec"])

        dim_specs = parse_dims("vec(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        with warning_sink.context() as warnings:
            result = execute_unsharder_plan(plans[0], shards)
        assert len(result) == 1
        assert torch.allclose(result[0].rename(None), full_tensor)
        assert warnings == []

    def test_7d_tensor_sharded_at_dim5(self) -> None:
        """High-dimensional: 7-D tensor 'd0 d1 d2 d3 d4 d5(tp) d6' TP=2."""
        torch.manual_seed(42)
        full_tensor = torch.randn(2, 2, 2, 2, 2, 4, 2)
        dim_names: list[str] = ["d0", "d1", "d2", "d3", "d4", "d5", "d6"]
        shards = _name_tensors(list(full_tensor.chunk(2, dim=5)), dim_names)

        dim_specs = parse_dims("d0 d1 d2 d3 d4 d5(tp) d6")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=2)} for i in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 1
        assert isinstance(plans[0].params, ConcatParams)
        assert plans[0].params.dim_name == "d5"

        with warning_sink.context() as warnings:
            result = execute_unsharder_plan(plans[0], shards)
        assert len(result) == 1
        assert torch.allclose(result[0].rename(None), full_tensor)
        assert warnings == []

    def test_adjacent_sharded_dims(self) -> None:
        """Adjacent sharded dims: 'pre cp_dim(cp) tp_dim(tp) post' — dim_name='cp_dim' and 'tp_dim'."""
        torch.manual_seed(42)
        full_tensor = torch.randn(3, 8, 6, 5)
        dim_names: list[str] = ["pre", "cp_dim", "tp_dim", "post"]

        cp_chunks = list(full_tensor.chunk(2, dim=1))
        shard_map: dict[tuple[int, int], torch.Tensor] = {}
        for cp_rank in range(2):
            tp_chunks = list(cp_chunks[cp_rank].chunk(3, dim=2))
            for tp_rank in range(3):
                shard_map[(cp_rank, tp_rank)] = tp_chunks[tp_rank]

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp in range(2):
            for tp in range(3):
                tensors.append(shard_map[(cp, tp)])
                parallel_infos.append(
                    {
                        ParallelAxis.CP: AxisInfo(axis_rank=cp, axis_size=2),
                        ParallelAxis.TP: AxisInfo(axis_rank=tp, axis_size=3),
                    }
                )

        dim_specs = parse_dims("pre cp_dim(cp) tp_dim(tp) post")
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current = _name_tensors(tensors, dim_names)
        with warning_sink.context() as warnings:
            for plan in plans:
                current = execute_unsharder_plan(plan, current)
        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)
        assert warnings == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
