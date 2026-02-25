from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    ExternalSeqId,
    TokenAlignerStepAux,
    TokenAlignerGlobalAux,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows

_AUX_NAMES_BY_FRAMEWORK: dict[str, frozenset[str]] = {
    "sglang": frozenset(
        {"input_ids", "positions", "seq_lens", "req_pool_indices", "rids"}
    ),
    "megatron": frozenset(
        {"input_ids", "position_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"}
    ),
}
AUX_NAMES: frozenset[str] = frozenset().union(*_AUX_NAMES_BY_FRAMEWORK.values())


# ── framework-agnostic ──────────────────────────────────────────────


def load_and_normalize_aux(
    dump_path: Path, df: pl.DataFrame
) -> Optional[TokenAlignerGlobalAux]:
    """Bootstrap: load, unshard, and normalize auxiliary tensors for one side."""
    framework: Optional[str] = _detect_framework(df, dump_path=dump_path)
    if framework is None:
        return None

    aux_names: frozenset[str] = _AUX_NAMES_BY_FRAMEWORK[framework]

    available_names: set[str] = set(df["name"].unique().to_list()) & aux_names
    step_values: list[int] = sorted(df["step"].unique().to_list())

    raw: dict[int, dict[str, object]] = {}
    for step in step_values:
        step_data: dict[str, object] = {}
        for name in available_names:
            tensor = _load_and_unshard_aux_tensor(
                name=name, step=step, df=df, dump_path=dump_path
            )
            if tensor is not None:
                step_data[name] = tensor
        if step_data:
            raw[step] = step_data

    layout: str = _detect_layout(raw, framework)
    steps: dict[int, TokenAlignerStepAux] = {}
    for step, step_data in raw.items():
        steps[step] = _normalize_step(
            step_data=step_data, framework=framework, layout=layout, step=step
        )

    return TokenAlignerGlobalAux(steps=steps, framework=framework, layout=layout)


def has_aux_tensors(df: pl.DataFrame) -> bool:
    """Check if the DataFrame contains the minimum auxiliary tensors for alignment."""
    names: set[str] = set(df["name"].unique().to_list())
    has_input_ids: bool = "input_ids" in names
    has_seq_info: bool = ("seq_lens" in names) or ("cu_seqlens_q" in names)
    return has_input_ids and has_seq_info


def _detect_framework(df: pl.DataFrame, dump_path: Path) -> Optional[str]:
    """Detect framework from aux tensor names, with parallel_info as fallback.

    Primary: tensor names present in the data (cu_seqlens_q → megatron, seq_lens → sglang).
    Fallback: parallel_info key embedded in dump metadata.
    """
    names: set[str] = set(df["name"].unique().to_list())

    if names & {"cu_seqlens_q", "cu_seqlens_kv", "qkv_format"}:
        return "megatron"
    if names & {"seq_lens", "positions", "req_pool_indices", "rids"}:
        return "sglang"

    first_row: dict = df.row(0, named=True)
    value: ValueWithMeta = ValueWithMeta.load(dump_path / first_row["filename"])

    if "megatron_parallel_info" in value.meta:
        return "megatron"
    if "sglang_parallel_info" in value.meta:
        return "sglang"

    return None


def _detect_layout(raw: dict[int, dict[str, object]], framework: str) -> str:
    """Detect layout from loaded auxiliary tensors.

    Currently only THD is supported. BSHD detection raises NotImplementedError.
    """
    if framework == "megatron":
        return _detect_layout_megatron(raw)
    return "thd"


def _load_and_unshard_aux_tensor(
    *, name: str, step: int, df: pl.DataFrame, dump_path: Path
) -> Optional[object]:
    """Load an auxiliary tensor for (name, step), unshard if needed."""
    rows = filter_rows(df, conditions={"name": name, "step": step})
    if not rows:
        return None

    loaded: list[ValueWithMeta] = [
        ValueWithMeta.load(dump_path / r["filename"]) for r in rows
    ]

    if name == "rids":
        if len(loaded) > 1:
            first_value = loaded[0].value
            for i, item in enumerate(loaded[1:], start=1):
                if item.value != first_value:
                    warnings.warn(
                        f"rids mismatch across ranks: rank 0 has {first_value}, "
                        f"rank {i} has {item.value}"
                    )
                    break
        return loaded[0].value

    tensors: list[torch.Tensor] = [
        item.value for item in loaded if isinstance(item.value, torch.Tensor)
    ]
    if not tensors:
        return None

    if len(tensors) == 1:
        return tensors[0]

    metas: list[dict] = [item.meta for item in loaded]
    dims_str = metas[0].get("dims")
    if dims_str is not None:
        dim_specs = parse_dims(dims_str)
        parallel_infos = [normalize_parallel_info(m) for m in metas]
        plans = compute_unsharder_plan(
            dim_specs=dim_specs, parallel_infos=parallel_infos
        )

        current = tensors
        for plan in plans:
            current, _ = execute_unsharder_plan(plan, current)

        assert len(current) == 1
        return current[0]

    warnings.warn(
        f"aux tensor '{name}' has {len(tensors)} ranks but no dims metadata, "
        f"using rank 0 only"
    )
    return tensors[0]


def _normalize_step(
    *, step_data: dict[str, object], framework: str, layout: str, step: int
) -> TokenAlignerStepAux:
    """Normalize raw loaded data into StepAux."""
    if framework == "sglang":
        return _normalize_sglang(step_data, step=step)
    else:
        return _normalize_megatron(step_data, layout=layout, step=step)


# ── sglang ──────────────────────────────────────────────────────────


def _normalize_sglang(step_data: dict[str, object], *, step: int) -> TokenAlignerStepAux:
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

    num_seqs: int = int(seq_lens.shape[0])

    seq_ids: tuple[ExternalSeqId, ...]
    if rids_raw is not None and isinstance(rids_raw, (list, tuple)):
        seq_ids = tuple(str(r) for r in rids_raw)
    else:
        seq_ids = tuple((step, i) for i in range(num_seqs))

    return TokenAlignerStepAux(
        input_ids=input_ids,
        positions=positions,
        seq_lens=seq_lens,
        seq_ids=seq_ids,
    )


# ── megatron ────────────────────────────────────────────────────────


def _detect_layout_megatron(raw: dict[int, dict[str, object]]) -> str:
    """Detect layout for Megatron framework from loaded auxiliary tensors."""
    for step_data in raw.values():
        qkv_format = step_data.get("qkv_format")
        if qkv_format is not None:
            fmt = qkv_format if isinstance(qkv_format, str) else str(qkv_format)
            if "bshd" in fmt.lower():
                raise NotImplementedError(
                    "BSHD layout is not currently supported. "
                    "Use aux_loader BSHD→THD conversion (planned)."
                )
            return "thd"

        input_ids = step_data.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
            raise NotImplementedError(
                "BSHD layout is not currently supported. "
                "Use aux_loader BSHD→THD conversion (planned)."
            )

    warnings.warn(
        "Megatron layout detection: no qkv_format or 2D input_ids found, "
        "falling back to thd"
    )
    return "thd"


def _normalize_megatron(
    step_data: dict[str, object], *, layout: str, step: int
) -> TokenAlignerStepAux:
    input_ids: torch.Tensor = step_data["input_ids"]

    cu_seqlens_q = step_data.get("cu_seqlens_q")
    if cu_seqlens_q is not None:
        seq_lens: torch.Tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    else:
        seq_lens = torch.tensor([input_ids.shape[0]], dtype=torch.long)

    position_ids = step_data.get("position_ids")
    if position_ids is not None:
        positions: torch.Tensor = position_ids
    else:
        positions = _infer_positions(seq_lens=seq_lens)

    num_seqs: int = int(seq_lens.shape[0])
    seq_ids: tuple[ExternalSeqId, ...] = tuple(
        (step, seq_index) for seq_index in range(num_seqs)
    )

    return TokenAlignerStepAux(
        input_ids=input_ids,
        positions=positions,
        seq_lens=seq_lens,
        seq_ids=seq_ids,
    )


def _infer_positions(*, seq_lens: torch.Tensor) -> torch.Tensor:
    """Infer positions when position_ids is missing (THD only)."""
    return torch.cat([torch.arange(int(slen.item())) for slen in seq_lens])
