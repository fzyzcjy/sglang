from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

import polars as pl
from pydantic import ConfigDict, Discriminator, Field, TypeAdapter, model_validator
from rich.console import Group, RenderableType
from rich.markup import escape

from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
    format_comparison,
    format_replicated_checks,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
)
from sglang.srt.debug_utils.comparator.utils import Pair, _StrictBase

if TYPE_CHECKING:
    from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
        AlignerPlan,
    )
    from sglang.srt.debug_utils.comparator.report_sink import Verbosity


class BaseLog(_StrictBase):
    category: str
    message: str

    def to_text(self) -> str:
        return self.message


class ErrorLog(BaseLog):
    kind: Literal["error"] = "error"


class InfoLog(BaseLog):
    kind: Literal["info"] = "info"


AnyLog = Annotated[Union[ErrorLog, InfoLog], Discriminator("kind")]


def _split_logs(logs: list[BaseLog]) -> tuple[list[ErrorLog], list[InfoLog]]:
    errors: list[ErrorLog] = [log for log in logs if isinstance(log, ErrorLog)]
    infos: list[InfoLog] = [log for log in logs if isinstance(log, InfoLog)]
    return errors, infos


class ReplicatedCheckResult(_StrictBase):
    axis: str
    group_index: int
    compared_index: int
    baseline_index: int
    passed: bool
    atol: float
    diff: Optional[DiffInfo] = None


class BundleFileInfo(_StrictBase):
    """Per-file info within a bundle (one rank's raw tensor)."""

    shape: list[int]
    dtype: str
    rank: Optional[int] = None
    parallel_info: Optional[dict[str, str]] = None  # e.g. {"tp": "0/4", "ep": "1/2"}


class BundleSideInfo(_StrictBase):
    num_files: int
    files: list[BundleFileInfo]
    dims: Optional[str] = None  # e.g. "b s h(tp) d"


class ShapeSnapshot(_StrictBase):
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]


class StepShapeTrace(_StrictBase):
    step: int
    snapshots: list[ShapeSnapshot]  # snapshots[i] corresponds to sub_plans[i]


class SideShapeTrace(_StrictBase):
    step_traces: list[StepShapeTrace]


class _OutputRecord(_StrictBase):
    errors: list[ErrorLog] = Field(default_factory=list)
    infos: list[InfoLog] = Field(default_factory=list)

    @abstractmethod
    def _format_body(self) -> str: ...

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        return self._format_body()

    def to_rich(self, verbosity: Verbosity = "normal") -> RenderableType:
        body: RenderableType = self._format_rich_body(verbosity=verbosity)

        log_lines: list[str] = []
        if self.errors:
            log_lines.extend(f"  [red]✗ {e.to_text()}[/]" for e in self.errors)
        if self.infos:
            log_lines.extend(f"  [dim]ℹ {i.to_text()}[/]" for i in self.infos)

        if not log_lines:
            return body

        log_block: str = "\n".join(log_lines)
        if isinstance(body, str):
            return body + "\n" + log_block
        return Group(body, log_block)

    def to_text(self) -> str:
        body = self._format_body()
        if self.errors:
            body += "\n" + "\n".join(f"  ✗ {e.to_text()}" for e in self.errors)
        if self.infos:
            body += "\n" + "\n".join(f"  ℹ {i.to_text()}" for i in self.infos)
        return body


class RecordLocation(_StrictBase):
    step: Optional[int] = None


class _BaseComparisonRecord(_OutputRecord):
    location: RecordLocation = Field(default_factory=RecordLocation)

    def _format_location_prefix(self) -> str:
        if self.location.step is not None:
            return f"[step={self.location.step}] "
        return ""

    def _format_location_prefix_rich(self) -> str:
        if self.location.step is not None:
            return escape(f"[step={self.location.step}]") + " "
        return ""

    def _format_location_suffix(self) -> str:
        if self.location.step is not None:
            return f" (step={self.location.step})"
        return ""


class ConfigRecord(_OutputRecord):
    type: Literal["config"] = "config"
    config: dict[str, Any]

    def _format_body(self) -> str:
        return f"Config: {self.config}"

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        from rich.panel import Panel

        lines: list[str] = [f"  [bold]{k}[/] : {v}" for k, v in self.config.items()]
        return Panel("\n".join(lines), title="Comparator Config", border_style="cyan")


class SkipComparisonRecord(_BaseComparisonRecord):
    type: Literal["skip"] = "skip"
    name: str
    reason: str

    @property
    def category(self) -> str:
        if self.errors:
            return "failed"
        return "skipped"

    def _format_body(self) -> str:
        return f"Skip: {self.name}{self._format_location_suffix()} ({self.reason})"

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        suffix: str = self._format_location_suffix()
        return f"[dim]⊘ {escape(self.name)}{suffix} ── skipped ({escape(self.reason)})[/]"


class _TableRecord(_OutputRecord):
    label: str
    rows: list[dict[str, Any]]

    @abstractmethod
    def _table_title(self) -> str: ...

    def _format_body(self) -> str:
        from sglang.srt.debug_utils.comparator.display import _render_polars_as_text

        return _render_polars_as_text(
            pl.DataFrame(self.rows), title=self._table_title()
        )

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        from sglang.srt.debug_utils.comparator.display import (
            _render_polars_as_rich_table,
        )

        return _render_polars_as_rich_table(
            pl.DataFrame(self.rows), title=self._table_title()
        )


class RankInfoRecord(_TableRecord):
    type: Literal["rank_info"] = "rank_info"

    def _table_title(self) -> str:
        return f"{self.label} ranks"


class InputIdsRecord(_TableRecord):
    type: Literal["input_ids"] = "input_ids"

    def _table_title(self) -> str:
        return f"{self.label} input_ids & positions"


class TensorComparisonRecord(TensorComparisonInfo, _BaseComparisonRecord):
    model_config = ConfigDict(extra="forbid", defer_build=True)

    type: Literal["comparison"] = "comparison"
    aligner_plan: Optional[AlignerPlan] = None
    replicated_checks: list[ReplicatedCheckResult] = Field(default_factory=list)
    raw_bundle_info: Optional[Pair[BundleSideInfo]] = None
    shape_traces: Optional[Pair[SideShapeTrace]] = None

    @property
    def category(self) -> str:
        if self.errors:
            return "failed"
        if any(not check.passed for check in self.replicated_checks):
            return "failed"
        return "passed" if self.diff is not None and self.diff.passed else "failed"

    def _format_body(self) -> str:
        body: str = self._format_location_prefix() + format_comparison(self)
        if self.replicated_checks:
            body += "\n" + format_replicated_checks(self.replicated_checks)
        if self.aligner_plan is not None:
            body += "\n" + _format_aligner_plan(self.aligner_plan)
        return body

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
            format_comparison_rich,
        )

        return self._format_location_prefix_rich() + format_comparison_rich(
            record=self, verbosity=verbosity
        )


class NonTensorComparisonRecord(_BaseComparisonRecord):
    type: Literal["non_tensor"] = "non_tensor"
    name: str
    baseline_value: str
    target_value: str
    baseline_type: str
    target_type: str
    values_equal: bool

    @property
    def category(self) -> str:
        if self.errors:
            return "failed"
        return "passed" if self.values_equal else "failed"

    def _format_body(self) -> str:
        suffix: str = self._format_location_suffix()
        if self.values_equal:
            return f"NonTensor: {self.name}{suffix} = {self.baseline_value} ({self.baseline_type}) [equal]"
        return (
            f"NonTensor: {self.name}{suffix}\n"
            f"  baseline = {self.baseline_value} ({self.baseline_type})\n"
            f"  target   = {self.target_value} ({self.target_type})"
        )

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        suffix: str = self._format_location_suffix()
        name: str = escape(self.name)
        baseline_val: str = escape(self.baseline_value)
        target_val: str = escape(self.target_value)

        if self.values_equal:
            return (
                f"═ {name}{suffix} = {baseline_val} "
                f"({self.baseline_type}) [green]✓[/]"
            )
        return (
            f"═ [bold red]{name}{suffix}[/]\n"
            f"  baseline = {baseline_val} ({self.baseline_type})\n"
            f"  target   = {target_val} ({self.target_type})"
        )


class SummaryRecord(_OutputRecord):
    type: Literal["summary"] = "summary"
    total: int
    passed: int
    failed: int
    skipped: int

    @model_validator(mode="after")
    def _validate_totals(self) -> "SummaryRecord":
        expected: int = self.passed + self.failed + self.skipped
        if self.total != expected:
            raise ValueError(
                f"total={self.total} != passed({self.passed}) + failed({self.failed}) + skipped({self.skipped}) = {expected}"
            )
        return self

    def _format_body(self) -> str:
        return (
            f"Summary: {self.passed} passed, {self.failed} failed, "
            f"{self.skipped} skipped (total {self.total})"
        )

    def _format_rich_body(self, verbosity: Verbosity = "normal") -> RenderableType:
        from rich.panel import Panel

        text: str = (
            f"[bold green]{self.passed} passed[/] │ "
            f"[bold red]{self.failed} failed[/] │ "
            f"[yellow]{self.skipped} skipped[/] │ "
            f"{self.total} total"
        )
        return Panel(text, title="SUMMARY", border_style="bold")


class LogRecord(_OutputRecord):
    type: Literal["log"] = "log"

    def _format_body(self) -> str:
        return ""


def _format_aligner_plan(plan: AlignerPlan) -> str:
    lines: list[str] = ["Aligner Plan:"]

    for side_label, side_plans in [
        ("baseline", plan.per_step_plans.x),
        ("target", plan.per_step_plans.y),
    ]:
        if not side_plans:
            lines.append(f"  {side_label}: (no steps)")
            continue

        step_summaries: list[str] = []
        for step_plan in side_plans:
            sub_strs: list[str] = []
            for sub in step_plan.sub_plans:
                sub_strs.append(f"{sub.type}")
            summary: str = ", ".join(sub_strs) if sub_strs else "passthrough"
            step_summaries.append(f"step={step_plan.step}: {summary}")
        lines.append(f"  {side_label}: [{'; '.join(step_summaries)}]")

    if plan.token_aligner_plan is not None:
        num_tokens: int = len(plan.token_aligner_plan.locators.x.steps)
        lines.append(f"  token_aligner: {num_tokens} tokens aligned")

    if plan.axis_aligner_plan is not None:
        parts: list[str] = []
        if plan.axis_aligner_plan.pattern.x:
            parts.append(f"x: {plan.axis_aligner_plan.pattern.x}")
        if plan.axis_aligner_plan.pattern.y:
            parts.append(f"y: {plan.axis_aligner_plan.pattern.y}")
        lines.append(f"  axis_aligner: {', '.join(parts)}")

    return "\n".join(lines)


AnyRecord = Annotated[
    Union[
        ConfigRecord,
        RankInfoRecord,
        InputIdsRecord,
        SkipComparisonRecord,
        TensorComparisonRecord,
        NonTensorComparisonRecord,
        SummaryRecord,
        LogRecord,
    ],
    Discriminator("type"),
]


def _get_any_record_adapter() -> TypeAdapter:
    return TypeAdapter(AnyRecord)


def parse_record_json(json_str: str | bytes) -> AnyRecord:
    return _get_any_record_adapter().validate_json(json_str)
