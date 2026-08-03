#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command palette + slash commands for the TUI.

The command palette (Ctrl+P) is a GUI-style menu of actions that map to the
app's workspace-level actions; slash commands are typed in the composer
(/help, /new, /clear, /doctor, /catalog, /cancel, /exit). Together they give
two lightweight interaction paths over the persistent workspace."""

from typing import TYPE_CHECKING, Callable
from rich.markup import escape
from rich.text import Text
from textual.command import Hit, Provider

if TYPE_CHECKING:
    from textual.command import Hits

    from .screens.workspace import WorkspaceScreen


class AiCommandProvider(Provider):
    """Command-palette entries that map to app actions."""

    async def search(self, query: str) -> "Hits":
        app = self.app
        workspace = getattr(app, "workspace", None)
        commands: "list[tuple[str, str, Callable[[], None]]]" = [
            (
                "New session",
                "Start a fresh conversation",
                workspace.action_new_session if workspace else lambda: None,
            ),
            (
                "Clear conversation",
                "Clear the conversation area",
                workspace.action_clear_conversation if workspace else lambda: None,
            ),
            ("Catalog", "Open agents, skills, MCP", app.action_catalog),
            ("Doctor", "Validate project and Runtime", app.action_doctor),
            ("Quit", "Exit lt ai", app.exit),
        ]
        q = query.lower().strip()
        for name, help_text, action in commands:
            if not q or q in name.lower():
                yield Hit(1.0, Text(name), action, text=name, help=help_text)


def handle_slash_command(screen: "WorkspaceScreen", line: str) -> bool:
    """Dispatch a ``/``-prefixed composer line.

    Returns True if consumed (the line was a slash command, not a prompt).
    Unknown commands print a hint rather than sending the text to the agent."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    conv = screen.conversation

    if cmd in ("/exit", "/quit"):
        screen.app.exit()
        return True
    if cmd == "/help":
        conv.add_status(
            "slash: /new <id> /session /clear /cancel /doctor /catalog /help /exit"
        )
        screen.query_one("ConversationView").refresh_from()
        return True
    if cmd == "/clear":
        screen.action_clear_conversation()
        return True
    if cmd == "/session":
        conv.add_status(f"session: {screen.session_id}")
        screen.query_one("ConversationView").refresh_from()
        return True
    if cmd == "/new":
        from ..client import validate_session_id

        if not arg:
            conv.add_status("usage: /new <session-id>")
            screen.query_one("ConversationView").refresh_from()
            return True
        try:
            screen.session_id = validate_session_id(arg)
        except Exception as exc:
            conv.add_error(f"invalid session: {exc}")
            screen.query_one("ConversationView").refresh_from()
            return True
        conv.clear()
        conv.add_status(f"session: {escape(screen.session_id)}")
        screen.query_one("ConversationView").refresh_from()
        screen.query_one("Sidebar").set_active_session(screen.session_id)
        return True
    if cmd == "/cancel":
        screen.action_cancel_run()
        return True
    if cmd == "/doctor":
        screen.app.action_doctor()
        return True
    if cmd == "/catalog":
        screen.app.action_catalog()
        return True
    conv.add_error(f"unknown command: {cmd} (try /help)")
    screen.query_one("ConversationView").refresh_from()
    return True
