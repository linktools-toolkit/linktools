#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The status bar.

A one-line dock at the bottom (above the composer) showing the active session
and the run status. Single source of truth for "what is the app doing right
now"; the workspace pushes updates to it. The keybinding hint lives in the
composer helper line, not here, to avoid duplication."""

from typing import TYPE_CHECKING

from rich.markup import escape
from textual.containers import Horizontal
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class StatusBar(Horizontal):
    """Left = session/model, right = status, far-right = keybinding hint."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $boost;
        border-top: solid $primary-background;
        layer: base;
    }
    StatusBar > Static { height: 1; padding: 0 1; }
    StatusBar #status-left { width: 1fr; }
    """

    def compose(self) -> "ComposeResult":
        yield Static("", id="status-left")

    @property
    def left_text(self) -> str:
        """The current status text (for tests / assertions)."""
        from rich.text import Text

        content = getattr(
            self.query_one("#status-left", Static), "_Static__content", None
        )
        if isinstance(content, Text):
            return content.plain
        return str(content or "")

    def set_session(self, session_id: str) -> None:
        self.query_one("#status-left", Static).update(f"session: {escape(session_id)}")

    def set_status(self, text: str) -> None:
        self.query_one("#status-left", Static).update(escape(text))
