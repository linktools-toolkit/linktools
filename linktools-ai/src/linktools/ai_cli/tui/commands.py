#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command palette + slash commands for the TUI.

The command palette (Ctrl+P) is a GUI-style menu of actions; slash commands are
typed in the composer (/help, /new, /clear, ...). Together they replace the old
flat command surface with two lightweight interaction paths."""

from typing import TYPE_CHECKING, Callable

from rich.markup import escape
from rich.text import Text
from textual.command import Hit, Hits, Provider

if TYPE_CHECKING:
    from .screens.chat import ChatScreen


class AiCommandProvider(Provider):
    """Command-palette entries that map to app actions."""

    async def search(self, query: str) -> Hits:
        app = self.app
        commands: "list[tuple[str, str, Callable[[], None]]]" = [
            ("Resources", "Open agents, skills, MCP", app.action_resources),
            ("Runs", "Open sessions, runs, approvals", app.action_runs),
            ("Doctor", "Validate project and Runtime", app.action_doctor),
            ("Quit", "Exit lt ai", app.quit),
        ]
        q = query.lower().strip()
        for name, help_text, action in commands:
            if not q or q in name.lower():
                yield Hit(1.0, Text(name), action, text=name, help=help_text)


def handle_slash_command(screen: "ChatScreen", line: str) -> bool:
    """Dispatch a ``/``-prefixed composer line.

    Returns True if consumed (the line was a slash command, not a prompt).
    Unknown commands print a hint rather than sending the text to the agent."""
    from textual.widgets import RichLog

    from ..client import validate_session_id  # ai_cli.client

    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    log = screen.query_one("#conversation", RichLog)

    if cmd in ("/exit", "/quit"):
        screen.app.exit()
        return True
    if cmd == "/help":
        log.write(
            "[dim]slash: /new <session> /session /clear /cancel /help /exit[/dim]"
        )
        return True
    if cmd == "/clear":
        log.clear()
        return True
    if cmd == "/session":
        log.write(f"[dim]session: {escape(screen.session_id)}[/dim]")
        return True
    if cmd == "/new":
        if not arg:
            log.write("[dim]usage: /new <session-id>[/dim]")
            return True
        screen.session_id = validate_session_id(arg)
        log.write(f"[dim]session: {escape(screen.session_id)}[/dim]")
        return True
    if cmd == "/cancel":
        screen.action_cancel_run()
        return True
    log.write(f"[red]unknown command: {cmd} (try /help)[/red]")
    return True
