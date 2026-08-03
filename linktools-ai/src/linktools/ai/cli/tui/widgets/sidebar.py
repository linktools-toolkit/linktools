#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The left sidebar: session selector + runs/approvals/resources overview.

A persistent panel (unlike the old full-screen ``RunsScreen`` overlay) that
reflects live backend state. Refreshes on demand via its ``render_*`` methods
without taking over the conversation. Selecting a session switches the chat's
active session; runs/approvals are read-only."""

from typing import TYPE_CHECKING, Any

from rich.markup import escape
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, ListItem, ListView, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


def _status_value(obj: Any) -> str:
    status = getattr(obj, "status", None)
    value = getattr(status, "value", status)
    return str(value) if value is not None else "?"


class SessionSelected(Message):
    """Emitted when the user picks a session in the sidebar."""

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id


class Sidebar(Vertical):
    """Sessions list, runs summary, pending approvals, resource counts.

    ``busy`` toggles the header indicator shown while a run is in flight."""

    DEFAULT_CSS = """
    Sidebar {
        width: 28;
        min-width: 20;
        max-width: 40;
        layout: vertical;
        background: $panel;
        border-right: solid $primary;
        overflow-y: auto;
    }
    Sidebar .sb-section { height: auto; margin: 0 0 1 0; padding: 0 1; }
    Sidebar .sb-title { color: $text-muted; text-style: bold; margin-bottom: 0; }
    Sidebar ListView {
        background: $panel; height: auto; max-height: 10; border: none;
    }
    Sidebar ListView > ListItem { padding: 0 1; }
    Sidebar .sb-empty { color: $text-disabled; padding: 0 1; }
    """

    BINDINGS = [
        Binding("up", "cursor_up", show=False),
        Binding("down", "cursor_down", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._active_session: "str | None" = None

    def compose(self) -> "ComposeResult":
        with Vertical(classes="sb-section"):
            yield Label("Sessions", classes="sb-title")
            yield ListView(id="sb-sessions")
        with Vertical(classes="sb-section"):
            yield Label("Runs", classes="sb-title")
            yield Static("(loading…)", id="sb-runs", classes="sb-empty")
        with Vertical(classes="sb-section"):
            yield Label("Pending approvals", classes="sb-title")
            yield Static("(loading…)", id="sb-approvals", classes="sb-empty")
        with Vertical(classes="sb-section"):
            yield Label("Resources", classes="sb-title")
            yield Static("(loading…)", id="sb-resources", classes="sb-empty")

    # -- updates ---------------------------------------------------------- #

    def set_active_session(self, session_id: str) -> None:
        self._active_session = session_id
        self._refresh_session_highlight()

    def render_sessions(self, sessions: "list[Any]", *, active: "str | None") -> None:
        view = self.query_one("#sb-sessions", ListView)
        view.clear()
        self._active_session = active
        if not sessions:
            view.append(ListItem(Label("(no sessions)")))
            return
        for record in sessions:
            sid = str(getattr(record, "id", "?"))
            label = f"{sid}  ({_status_value(record)})"
            # No fixed id: ListView.clear() does not free ids synchronously,
            # so re-rendering the same session id would raise DuplicateIds.
            view.append(ListItem(Label(label)))
        self._refresh_session_highlight()

    def render_runs(self, runs: "list[Any]") -> None:
        widget = self.query_one("#sb-runs", Static)
        if not runs:
            widget.update("(none)")
            return
        lines = [f"{getattr(r, 'id', '?')} ({_status_value(r)})" for r in runs[-20:]]
        widget.update("\n".join(escape(line) for line in lines))

    def render_approvals(self, approvals: "list[Any]") -> None:
        widget = self.query_one("#sb-approvals", Static)
        if not approvals:
            widget.update("(none)")
            return
        lines = [
            f"{getattr(a, 'id', '?')} tool={getattr(a, 'tool_name', '?')}"
            for a in approvals
        ]
        widget.update("\n".join(escape(line) for line in lines))

    def render_resources(
        self,
        *,
        agents: "tuple[str, ...]",
        skills: "tuple[str, ...]",
        mcp: "tuple[str, ...]",
    ) -> None:
        widget = self.query_one("#sb-resources", Static)
        widget.update(
            escape(f"agents: {len(agents)} · skills: {len(skills)} · mcp: {len(mcp)}")
        )

    # -- selection -------------------------------------------------------- #

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # Sessions render without fixed ids; the session id is the leading
        # token of the rendered label ("sid  (status)").
        session_id = self._label_session_id(event.item)
        if session_id:
            self.post_message(SessionSelected(session_id))

    def _refresh_session_highlight(self) -> None:
        view = self.query_one("#sb-sessions", ListView)
        target = self._active_session or ""
        for item in view.children:
            if isinstance(item, ListItem):
                sid = self._label_session_id(item)
                item.set_class(bool(target and sid == target), "-active")

    @staticmethod
    def _label_session_id(item: ListItem) -> str:
        """Extract the session id from a session ListItem's label text."""
        if not item.children:
            return ""
        label_widget = item.children[0]
        text = (
            getattr(getattr(label_widget, "_Static__content", None), "plain", "") or ""
        )
        first = text.split()[0] if text else ""
        return "" if first in {"(no", ""} else first
