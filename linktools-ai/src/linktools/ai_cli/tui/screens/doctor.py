#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The doctor screen.

Runs the project + Runtime checks through ``RuntimeClient.doctor`` and renders
the resulting ``DoctorReport`` (the screen never re-implements the checks). Esc
returns to chat."""

from typing import TYPE_CHECKING

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import RichLog

if TYPE_CHECKING:
    from linktools.ai_cli.client import RuntimeClient


class DoctorScreen(Screen):
    """Project + Runtime validation report."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client

    def compose(self) -> ComposeResult:
        yield RichLog(id="doctor-log", wrap=True, markup=True)

    def on_mount(self) -> None:
        self.run_worker(self._load())

    async def _load(self) -> None:
        log = self.query_one("#doctor-log", RichLog)
        report = await self.client.doctor()
        for check in report.checks:
            mark = "[green]ok[/green]" if check.ok else "[red]fail[/red]"
            detail = f": {escape(check.detail)}" if check.detail else ""
            log.write(f"[{mark}] {escape(check.label)}{detail}")
        failed = report.failed
        if not report.checks:
            log.write("[dim](no checks)[/dim]")
        elif failed:
            log.write(f"[red]{len(failed)} check(s) failed[/red]")
        else:
            log.write("[green]all checks passed[/green]")
