"""Formatting functions for comparator output records.

Extracted from output_types.py to separate data-structure definitions
from rendering / formatting logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from rich.markup import escape
from rich.panel import Panel

from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
    format_comparison,
    format_replicated_checks,
)

if TYPE_CHECKING:
    from rich.console import RenderableType

    from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
        AlignerPlan,
    )
    from sglang.srt.debug_utils.comparator.output_types import (
        ConfigRecord,
        LogRecord,
        NonTensorComparisonRecord,
        SkipComparisonRecord,
        SummaryRecord,
        TensorComparisonRecord,
        _TableRecord,
    )

Verbosity = Literal["minimal", "normal", "verbose"]


# ── ConfigRecord ──────────────────────────────────────────────────────


def _format_config_body(record: ConfigRecord) -> str:
    return f"Config: {record.config}"


def _format_config_rich_body(
    record: ConfigRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    lines: list[str] = [f"  [bold]{k}[/] : {v}" for k, v in record.config.items()]
    return Panel("\n".join(lines), title="Comparator Config", border_style="cyan")


# ── SkipComparisonRecord ─────────────────────────────────────────────


def _format_skip_body(record: SkipComparisonRecord) -> str:
    return f"Skip: {record.name}{record._format_location_suffix()} ({record.reason})"


def _format_skip_rich_body(
    record: SkipComparisonRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    suffix: str = record._format_location_suffix()
    return (
        f"[dim]⊘ {escape(record.name)}{suffix} ── skipped ({escape(record.reason)})[/]"
    )


# ── _TableRecord ─────────────────────────────────────────────────────


def _format_table_body(record: _TableRecord) -> str:
    import polars as pl

    from sglang.srt.debug_utils.comparator.display import _render_polars_as_text

    return _render_polars_as_text(
        pl.DataFrame(record.rows), title=record._table_title()
    )


def _format_table_rich_body(
    record: _TableRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    import polars as pl

    from sglang.srt.debug_utils.comparator.display import (
        _render_polars_as_rich_table,
    )

    return _render_polars_as_rich_table(
        pl.DataFrame(record.rows), title=record._table_title()
    )


# ── TensorComparisonRecord ───────────────────────────────────────────


def _format_tensor_comparison_body(record: TensorComparisonRecord) -> str:
    body: str = record._format_location_prefix() + format_comparison(record)
    if record.replicated_checks:
        body += "\n" + format_replicated_checks(record.replicated_checks)
    if record.aligner_plan is not None:
        body += "\n" + _format_aligner_plan(record.aligner_plan)
    return body


def _format_tensor_comparison_rich_body(
    record: TensorComparisonRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    from sglang.srt.debug_utils.comparator.tensor_comparator.formatter import (
        format_comparison_rich,
    )

    return record._format_location_prefix_rich() + format_comparison_rich(
        record=record, verbosity=verbosity
    )


# ── NonTensorComparisonRecord ────────────────────────────────────────


def _format_non_tensor_body(record: NonTensorComparisonRecord) -> str:
    suffix: str = record._format_location_suffix()
    if record.values_equal:
        return f"NonTensor: {record.name}{suffix} = {record.baseline_value} ({record.baseline_type}) [equal]"
    return (
        f"NonTensor: {record.name}{suffix}\n"
        f"  baseline = {record.baseline_value} ({record.baseline_type})\n"
        f"  target   = {record.target_value} ({record.target_type})"
    )


def _format_non_tensor_rich_body(
    record: NonTensorComparisonRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    suffix: str = record._format_location_suffix()
    name: str = escape(record.name)
    baseline_val: str = escape(record.baseline_value)
    target_val: str = escape(record.target_value)

    if record.values_equal:
        return (
            f"═ {name}{suffix} = {baseline_val} "
            f"({record.baseline_type}) [green]✓[/]"
        )
    return (
        f"═ [bold red]{name}{suffix}[/]\n"
        f"  baseline = {baseline_val} ({record.baseline_type})\n"
        f"  target   = {target_val} ({record.target_type})"
    )


# ── SummaryRecord ────────────────────────────────────────────────────


def _format_summary_body(record: SummaryRecord) -> str:
    return (
        f"Summary: {record.passed} passed, {record.failed} failed, "
        f"{record.skipped} skipped (total {record.total})"
    )


def _format_summary_rich_body(
    record: SummaryRecord, verbosity: Verbosity = "normal"
) -> RenderableType:
    text: str = (
        f"[bold green]{record.passed} passed[/] │ "
        f"[bold red]{record.failed} failed[/] │ "
        f"[yellow]{record.skipped} skipped[/] │ "
        f"{record.total} total"
    )
    return Panel(text, title="SUMMARY", border_style="bold")


# ── LogRecord ────────────────────────────────────────────────────────


def _format_log_body(record: LogRecord) -> str:
    return ""


# ── Standalone helpers ───────────────────────────────────────────────


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
