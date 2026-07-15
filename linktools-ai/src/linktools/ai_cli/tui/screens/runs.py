#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The runs screen.

Read-only listing of sessions / runs / pending approvals through
``RuntimeClient``. Esc returns to chat."""

from typing import TYPE_CHECKING

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import RichLog

if TYPE_CHECKING:
    from linktools.ai_cli.client import RuntimeClient


class RunsScreen(Screen):
    """Sessions / runs / approvals overview."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client

    def compose(self) -> ComposeResult:
        yield RichLog(id="runs-log", wrap=True, markup=True)

    def on_mount(self) -> None:
        self.run_worker(self._load())

    async def _load(self) -> None:
        log = self.query_one("#runs-log", RichLog)

        async def section(title: str, fetch, render) -> None:
            log.write(f"[b]{title}[/b]")
            items = await fetch()
            if not items:
                log.write("  [dim](none)[/dim]")
            for item in items:
                log.write(f"  {render(item)}")

        await section(
            "Sessions",
            self.client.list_sessions,
            lambda r: escape(
                f"{getattr(r, 'id', '?')} ({getattr(getattr(r, 'status', None), 'value', '?')})"
            ),
        )
        await section(
            "Runs",
            self.client.list_runs,
            lambda r: escape(
                f"{getattr(r, 'id', '?')} ({getattr(getattr(r, 'status', None), 'value', '?')})"
            ),
        )
        await section(
            "Pending approvals",
            self.client.list_approvals,
            lambda r: escape(
                f"{getattr(r, 'id', '?')} run={getattr(r, 'run_id', '?')} "
                f"tool={getattr(r, 'tool_name', '?')}"
            ),
        )
