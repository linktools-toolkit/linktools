#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The doctor modal.

Runs the project + Runtime checks through ``RuntimeClient.doctor`` and renders
the resulting :class:`DoctorReport`. Esc closes the overlay."""

from typing import TYPE_CHECKING

from rich.markup import escape
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from ...client import RuntimeClient


class DoctorModal(ModalScreen):
    """Project + Runtime validation report overlay."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Close")]

    DEFAULT_CSS = """
    DoctorModal { align: center middle; }
    DoctorModal > VerticalScroll {
        width: 80; height: 70%; border: round $primary;
        background: $surface; padding: 1 2;
    }
    DoctorModal Static { height: auto; }
    """

    def __init__(self, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client

    def compose(self) -> "ComposeResult":
        yield VerticalScroll(Static("[dim]running checks…[/dim]", id="doctor-body"))

    def on_mount(self) -> None:
        self.run_worker(self._load())

    async def _load(self) -> None:
        body = self.query_one("#doctor-body", Static)
        report = await self.client.doctor()
        lines: "list[str]" = ["[b]Doctor[/b]\n"]
        for check in report.checks:
            mark = "ok" if check.ok else "fail"
            color = "green" if check.ok else "red"
            detail = f": {escape(check.detail)}" if check.detail else ""
            lines.append(
                f"[{color}]{escape(mark)}[/{color}] {escape(check.label)}{detail}"
            )
        if not report.checks:
            lines.append("[dim](no checks)[/dim]")
        elif report.failed:
            lines.append(f"\n[red]{len(report.failed)} check(s) failed[/red]")
        else:
            lines.append("\n[green]all checks passed[/green]")
        body.update("\n".join(lines))
