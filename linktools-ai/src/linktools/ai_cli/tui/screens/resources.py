#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The resources screen.

Read-only listing of the project's agents / skills / MCP servers, fetched
through ``RuntimeClient`` (no registry access from the UI). Esc returns to
chat."""

from typing import TYPE_CHECKING

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import RichLog

if TYPE_CHECKING:
    from linktools.ai_cli.client import RuntimeClient


class ResourcesScreen(Screen):
    """Agents / skills / MCP overview."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client

    def compose(self) -> ComposeResult:
        yield RichLog(id="resources-log", wrap=True, markup=True)

    def on_mount(self) -> None:
        self.run_worker(self._load())

    async def _load(self) -> None:
        log = self.query_one("#resources-log", RichLog)

        async def section(title: str, fetch) -> None:
            log.write(f"[b]{title}[/b]")
            ids = await fetch()
            if not ids:
                log.write("  [dim](none)[/dim]")
            for item in ids:
                log.write(f"  - {escape(str(item))}")

        await section("Agents", self.client.list_agents)
        await section("Skills", self.client.list_skills)
        await section("MCP servers", self.client.list_mcp_servers)
