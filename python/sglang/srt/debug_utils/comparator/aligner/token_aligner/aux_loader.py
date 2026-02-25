from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Iterable, Tuple

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.aligner.entrypoint.executor import (
    execute_sub_plans,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.planner import (
    compute_per_step_sub_plans,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    PositionalSeqId,
    SeqId,
    SGLangSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerStepAux,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.output_types import GeneralWarning
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows

_BSHD_NOT_SUPPORTED_MSG: str = (
    "BSHD layout is not currently supported. "
    "Use aux_loader BSHD→THD conversion (planned)."
)


# ── plugin ABC ─────────────────────────────────────────────────────


class _AuxPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def tensor_names(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def non_tensor_names(self) -> frozenset[str]: ...

    @property
    def cp_sharded_names(self) -> frozenset[str]:
        return frozenset()

    @property
    def discriminating_names(self) -> frozenset[str]:
        """Field names unique to this framework (excluding shared names like input_ids)."""
        return frozenset()

    @abstractmethod
    def detect_layout(self, raw: dict[int, dict[str, object]]) -> str: ...

    @abstractmethod
    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: str, step: int
    ) -> TokenAlignerStepAux: ...

    @property
    def all_names(self) -> frozenset[str]:
        return self.tensor_names | self.non_tensor_names


# ── sglang plugin ─────────────────────────────────────────────────


class _SGLangPlugin(_AuxPlugin):
    @property
    def name(self) -> str:
        return "sglang"

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset({"input_ids", "positions", "seq_lens", "req_pool_indices"})

    @property
    def non_tensor_names(self) -> frozenset[str]:
        return frozenset({"rids"})

    @property
    def cp_sharded_names(self) -> frozenset[str]:
        return frozenset({"input_ids", "positions"})

    @property
    def discriminating_names(self) -> frozenset[str]:
        return frozenset({"seq_lens", "positions", "req_pool_indices", "rids"})

    def detect_layout(self, raw: dict[int, dict[str, object]]) -> str:
        return "thd"

    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: str, step: int
    ) -> TokenAlignerStepAux:
        input_ids = step_data["input_ids"]
        positions = step_data["positions"]
        seq_lens = step_data["seq_lens"]
        rids_raw = step_data.get("rids")

        assert isinstance(
            input_ids, torch.Tensor
        ), f"input_ids: expected Tensor, got {type(input_ids)}"
        assert isinstance(
            positions, torch.Tensor
        ), f"positions: expected Tensor, got {type(positions)}"
        assert isinstance(
            seq_lens, torch.Tensor
        ), f"seq_lens: expected Tensor, got {type(seq_lens)}"

        seq_lens_list: list[int] = seq_lens.tolist()
        num_seqs: int = len(seq_lens_list)

        seq_ids: list[SeqId]
        if rids_raw is not None and isinstance(rids_raw, (list, tuple)):
            seq_ids = [SGLangSeqId(rid=str(r)) for r in rids_raw]
        else:
            seq_ids = [PositionalSeqId(step=step, seq_index=i) for i in range(num_seqs)]

        return TokenAlignerStepAux(
            input_ids=input_ids.tolist(),
            positions=positions.tolist(),
            seq_lens=seq_lens_list,
            seq_ids=seq_ids,
        )


# ── megatron plugin ───────────────────────────────────────────────


class _MegatronPlugin(_AuxPlugin):
    @property
    def name(self) -> str:
        return "megatron"

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset({"input_ids", "position_ids", "cu_seqlens_q", "cu_seqlens_kv"})

    @property
    def non_tensor_names(self) -> frozenset[str]:
        return frozenset({"qkv_format"})

    @property
    def cp_sharded_names(self) -> frozenset[str]:
        return frozenset({"input_ids", "position_ids"})

    @property
    def discriminating_names(self) -> frozenset[str]:
        return frozenset({"cu_seqlens_q", "cu_seqlens_kv", "qkv_format"})

    def detect_layout(self, raw: dict[int, dict[str, object]]) -> str:
        for step_data in raw.values():
            if (qkv_format := step_data.get("qkv_format")) is not None:
                fmt = qkv_format if isinstance(qkv_format, str) else str(qkv_format)
                if "bshd" in fmt.lower():
                    raise NotImplementedError(_BSHD_NOT_SUPPORTED_MSG)
                return "thd"

            input_ids = step_data.get("input_ids")
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
                raise NotImplementedError(_BSHD_NOT_SUPPORTED_MSG)

        warning_sink.add(
            GeneralWarning(
                category="layout_detection_fallback",
                message=(
                    "Megatron layout detection: no qkv_format or 2D input_ids found, "
                    "falling back to thd"
                ),
            )
        )
        return "thd"

    def compute_step_aux(
        self, step_data: dict[str, object], *, layout: str, step: int
    ) -> TokenAlignerStepAux:
        input_ids: torch.Tensor = step_data["input_ids"]

        if (cu_seqlens_q := step_data.get("cu_seqlens_q")) is not None:
            seq_lens: torch.Tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        else:
            seq_lens = torch.tensor([input_ids.shape[0]], dtype=torch.long)

        if (position_ids := step_data.get("position_ids")) is not None:
            positions: torch.Tensor = position_ids
        else:
            positions = _infer_positions(seq_lens=seq_lens)

        seq_lens_list: list[int] = seq_lens.tolist()
        num_seqs: int = len(seq_lens_list)
        seq_ids: list[SeqId] = [
            PositionalSeqId(step=step, seq_index=seq_index)
            for seq_index in range(num_seqs)
        ]

        return TokenAlignerStepAux(
            input_ids=input_ids.tolist(),
            positions=positions.tolist(),
            seq_lens=seq_lens_list,
            seq_ids=seq_ids,
        )


# ── plugin registry ───────────────────────────────────────────────

_plugins: list[_AuxPlugin] = [_SGLangPlugin(), _MegatronPlugin()]

AUX_NAMES: frozenset[str] = frozenset().union(*(p.all_names for p in _plugins))


# ── framework-agnostic ────────────────────────────────────────────


def load_and_normalize_aux(
    dump_path: Path, df: pl.DataFrame
) -> Optional[TokenAlignerGlobalAux]:
    """Bootstrap: load, unshard, and normalize auxiliary tensors for one side."""
    plugin: Optional[_AuxPlugin] = _detect_plugin(df, dump_path=dump_path)
    if plugin is None:
        return None

    available_names: set[str] = set(df["name"].unique().to_list()) & plugin.all_names
    steps: list[int] = sorted(df["step"].unique().to_list())
    tensor_names: set[str] = available_names & plugin.tensor_names
    non_tensor_names: set[str] = available_names & plugin.non_tensor_names

    steps_data: dict[int, dict[str, object]] = {}
    for step in steps:
        step_data = dict(
            _load_step_data(
                step=step,
                tensor_names=tensor_names,
                non_tensor_names=non_tensor_names,
                df=df,
                dump_path=dump_path,
                plugin=plugin,
            )
        )
        if step_data:
            steps_data[step] = step_data

    layout: str = plugin.detect_layout(steps_data)

    step_auxs: dict[int, TokenAlignerStepAux] = {
        step: plugin.compute_step_aux(step_data, layout=layout, step=step)
        for step, step_data in steps_data.items()
    }

    return TokenAlignerGlobalAux(
        step_auxs=step_auxs, framework=plugin.name, layout=layout
    )


def has_aux_tensors(df: pl.DataFrame) -> bool:
    """Check if the DataFrame contains the minimum auxiliary tensors for alignment."""
    names: set[str] = set(df["name"].unique().to_list())
    has_input_ids: bool = "input_ids" in names
    has_seq_info: bool = ("seq_lens" in names) or ("cu_seqlens_q" in names)
    return has_input_ids and has_seq_info


def _detect_plugin(df: pl.DataFrame, dump_path: Path) -> Optional[_AuxPlugin]:
    names: set[str] = set(df["name"].unique().to_list())

    for plugin in _plugins:
        if names & plugin.discriminating_names:
            return plugin

    first_row: dict = df.row(0, named=True)
    value: ValueWithMeta = ValueWithMeta.load(dump_path / first_row["filename"])

    for plugin in _plugins:
        if f"{plugin.name}_parallel_info" in value.meta:
            return plugin

    return None


def _load_step_data(
    *,
    step: int,
    tensor_names: set[str],
    non_tensor_names: set[str],
    df: pl.DataFrame,
    dump_path: Path,
    plugin: _AuxPlugin,
) -> Iterable[Tuple[str, object]]:
    """Load all tensor and non-tensor aux values for a single step."""
    for name in non_tensor_names:
        value = _load_non_tensor_aux(name=name, step=step, df=df, dump_path=dump_path)
        if value is not None:
            yield name, value

    for name in tensor_names:
        tensor = _load_and_align_aux_tensor(
            name=name, step=step, df=df, dump_path=dump_path, plugin=plugin
        )
        if tensor is not None:
            yield name, tensor


def _load_non_tensor_aux(
    *, name: str, step: int, df: pl.DataFrame, dump_path: Path
) -> Optional[object]:
    """Load a non-tensor auxiliary value for a step, validating consistency across ranks."""
    rows = filter_rows(df, conditions={"name": name, "step": step})
    if not rows:
        return None

    loaded: list[ValueWithMeta] = [
        ValueWithMeta.load(dump_path / r["filename"]) for r in rows
    ]

    if len(loaded) > 1:
        first_value = loaded[0].value
        for i, item in enumerate(loaded[1:], start=1):
            if item.value != first_value:
                warning_sink.add(
                    GeneralWarning(
                        category=f"{name}_mismatch",
                        message=(
                            f"{name} mismatch across ranks: rank 0 has {first_value}, "
                            f"rank {i} has {item.value}"
                        ),
                    )
                )
                break

    return loaded[0].value


def _load_and_align_aux_tensor(
    *, name: str, step: int, df: pl.DataFrame, dump_path: Path, plugin: _AuxPlugin
) -> Optional[torch.Tensor]:
    """Load an auxiliary tensor for (name, step), align if needed."""
    rows = filter_rows(df, conditions={"name": name, "step": step})
    if not rows:
        return None

    loaded: list[ValueWithMeta] = [
        ValueWithMeta.load(dump_path / r["filename"]) for r in rows
    ]

    tensors: list[torch.Tensor] = [
        item.value for item in loaded if isinstance(item.value, torch.Tensor)
    ]
    if not tensors:
        return None

    if len(tensors) == 1:
        return tensors[0]

    metas: list[dict] = [item.meta for item in loaded]
    dims_str: Optional[str] = metas[0].get("dims") or _infer_aux_dims(
        name=name, plugin=plugin, metas=metas
    )

    if dims_str is not None:
        effective_metas: list[dict] = [{**m, "dims": dims_str} for m in metas]
        sub_plans = compute_per_step_sub_plans(metas=effective_metas)
        result = execute_sub_plans(tensors=tensors, plans=sub_plans)
        assert result is not None
        return result

    warning_sink.add(
        GeneralWarning(
            category="aux_no_dims",
            message=(
                f"aux tensor '{name}' has {len(tensors)} ranks "
                f"but no dims metadata, using rank 0 only"
            ),
        )
    )
    return tensors[0]


def _infer_aux_dims(
    *, name: str, plugin: _AuxPlugin, metas: list[dict]
) -> Optional[str]:
    """Infer dims for aux tensors lacking explicit dims metadata."""
    parallel_infos = [normalize_parallel_info(m) for m in metas]
    has_cp: bool = any(ParallelAxis.CP in info for info in parallel_infos)
    if not has_cp:
        return None

    if name in plugin.cp_sharded_names:
        raise NotImplementedError(
            f"Aux tensor '{name}' is CP-sharded but reorderer does not yet support "
            f"zigzag reordering on the 't' dimension. "
            f"Pass explicit dims= at dump time or wait for t-dim zigzag support."
        )

    return None


def _infer_positions(*, seq_lens: torch.Tensor) -> torch.Tensor:
    """Infer positions when position_ids is missing (THD only)."""
    return torch.cat([torch.arange(int(slen.item())) for slen in seq_lens])
