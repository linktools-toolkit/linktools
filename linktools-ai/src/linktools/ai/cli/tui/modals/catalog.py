#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The catalog modal.

A scrollable overlay listing the project's agents / skills / MCP servers
(fetched through ``RuntimeClient``). Esc/closes returns to the workspace."""

from typing import TYPE_CHECKING
from collections.abc import Awaitable, Callable, Iterable

from rich.markup import escape
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from ...client import RuntimeClient


class CatalogModal(ModalScreen):
    """Agents / skills / MCP overview overlay."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Close")]

    DEFAULT_CSS = """
    CatalogModal { align: center middle; }
    CatalogModal > VerticalScroll {
        width: 70; height: 70%; border: round $primary;
        background: $surface; padding: 1 2;
    }
    CatalogModal Static { height: auto; }
    """

    def __init__(self, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client

    def compose(self) -> "ComposeResult":
        yield VerticalScroll(Static("[b]Catalog[/b]\n", id="catalog-body"))

    def on_mount(self) -> None:
        self.run_worker(self._load())

    async def _load(self) -> None:
        body = self.query_one("#catalog-body", Static)
        lines: "list[str]" = ["[b]Catalog[/b]\n"]

        async def section(
            title: str, fetch: "Callable[[], Awaitable[Iterable[str]]]"
        ) -> None:
            lines.append(f"[b]{title}[/b]")
            ids = await fetch()
            if not ids:
                lines.append("  [dim](none)[/dim]")
            for item in ids:
                lines.append(f"  - {escape(str(item))}")
            lines.append("")

        await section("Agents", self.client.list_agents)
        await section("Skills", self.client.list_skills)
        await section("MCP servers", self.client.list_mcp_servers)
        body.update("\n".join(lines))
