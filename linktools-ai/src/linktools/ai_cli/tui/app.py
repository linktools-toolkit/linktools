#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The Textual app entry point.

Holds the ``RuntimeClient`` and mounts the :class:`ChatScreen`. Global
keybindings open the Resources/Runs/Doctor screens over the chat; each of those
screens binds Esc to pop back. ``run_tui`` is the function the thin ``lt ai
tui`` shell reaches through :mod:`linktools.ai_cli.tui` (which translates a
missing Textual install)."""

from typing import TYPE_CHECKING

from textual.app import App
from textual.binding import Binding

from .screens.chat import ChatScreen
from .screens.doctor import DoctorScreen
from .screens.resources import ResourcesScreen
from .screens.runs import RunsScreen

if TYPE_CHECKING:
    from linktools.ai_cli.client import RuntimeClient


class LinktoolsAIApp(App):
    """The ``lt ai`` Textual app. The client is the only backend handle screens
    may use."""

    CSS = """
    Screen { layout: vertical; }
    #conversation { border: round $primary; height: 1fr; }
    #composer { dock: bottom; height: 3; }
    """

    # Priority bindings so navigation fires even while the composer Input is
    # focused (Textual's Input otherwise eats ctrl+d as delete-right, making
    # Ctrl+D→Doctor unreachable on the chat screen).
    BINDINGS = [
        Binding("ctrl+r", "resources", "Resources", priority=True),
        Binding("ctrl+o", "runs", "Runs", priority=True),
        Binding("ctrl+d", "doctor", "Doctor", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, *, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client

    def on_mount(self) -> None:
        self.push_screen(ChatScreen(self.client))

    def action_resources(self) -> None:
        self.push_screen(ResourcesScreen(self.client))

    def action_runs(self) -> None:
        self.push_screen(RunsScreen(self.client))

    def action_doctor(self) -> None:
        self.push_screen(DoctorScreen(self.client))


def run_tui(*, project, remote, client: "RuntimeClient | None" = None) -> int:
    """Start the interactive Textual interface. ``client`` is injectable for
    tests; when omitted a local client is built from the current project
    (interactive, so model config may be prompted -- future work)."""
    if client is None:
        from ..client import build_runtime_client

        client = build_runtime_client(
            remote=remote, with_model=True, project=project, interactive=True
        )
    LinktoolsAIApp(client=client).run()
    return 0
