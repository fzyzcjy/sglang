from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Literal, Optional

from rich.console import Console

from sglang.srt.debug_utils.comparator.output_types import _OutputRecord

_CONSOLE: Optional[Console] = None
_VERBOSITY: str = "normal"

Verbosity = Literal["minimal", "normal", "verbose"]


def _get_console() -> Console:
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console()
    return _CONSOLE


def get_verbosity() -> str:
    return _VERBOSITY


def _set_verbosity(verbosity: str) -> None:
    global _VERBOSITY
    _VERBOSITY = verbosity


def _print_to_stdout(record: _OutputRecord, *, output_format: str) -> None:
    if output_format == "json":
        print(record.model_dump_json())
    else:
        console: Console = _get_console()
        console.print(record.to_rich())
        console.print()  # blank line between records


class ReportSink:
    """Unified entry point for all record output."""

    def __init__(self) -> None:
        self._output_format: str = "text"
        self._report_file: Optional[IO[str]] = None
        self._report_path: Optional[Path] = None

    def configure(
        self,
        *,
        output_format: str = "text",
        report_path: Optional[Path] = None,
        verbosity: str = "normal",
    ) -> None:
        self._output_format = output_format
        _set_verbosity(verbosity)

        if report_path is not None:
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                self._report_file = open(report_path, "w", encoding="utf-8")
                self._report_path = report_path
            except OSError as exc:
                print(
                    f"Warning: cannot open report file {report_path}: {exc}",
                    file=sys.stderr,
                )

    def add(self, record: _OutputRecord) -> None:
        _print_to_stdout(record, output_format=self._output_format)

        if self._report_file is not None:
            self._report_file.write(record.model_dump_json())
            self._report_file.write("\n")
            self._report_file.flush()

    def close(self) -> None:
        if self._report_file is not None:
            self._report_file.close()
            self._report_file = None

    @property
    def report_path(self) -> Optional[Path]:
        return self._report_path

    def _reset(self) -> None:
        self.close()
        self._output_format = "text"
        _set_verbosity("normal")
        self._report_path = None


report_sink = ReportSink()
