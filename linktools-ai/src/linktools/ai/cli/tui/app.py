#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The Textual app entry point.

Holds the :class:`RuntimeClient` and mounts the :class:`WorkspaceScreen` -- a
single persistent three-pane layout (sidebar / conversation / context). The
``lt ai tui`` shell reaches :func:`run_tui` through
:mod:`linktools.ai.cli.tui` (which translates a missing Textual install).

There are no full-screen overlays anymore: Doctor/Catalog detail open as
modals on top of the workspace, and the approval flow is a modal pushed from
the workspace."""

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App
from textual.binding import Binding

from .commands import AiCommandProvider
from .modals.catalog import CatalogModal
from .modals.doctor import DoctorModal
from .screens.workspace import WorkspaceScreen

if TYPE_CHECKING:
    from ..client import RuntimeClient


class LinktoolsAIApp(App):
    """The ``lt ai`` Textual app. The client is the only backend handle screens
    may use."""

    CSS = """
    Screen { layout: vertical; layers: base overlay; }
    #workspace-body { height: 1fr; }
    #conversation { border: round $primary; }
    """

    COMMANDS = {AiCommandProvider}

    BINDINGS = [
        Binding("ctrl+d", "doctor", "Doctor", priority=True),
        Binding("ctrl+k", "catalog", "Catalog", priority=True),
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, *, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client
        self.workspace: "WorkspaceScreen | None" = None

    def on_mount(self) -> None:
        self.workspace = WorkspaceScreen(self.client)
        self.push_screen(self.workspace)

    def action_doctor(self) -> None:
        self.push_screen(DoctorModal(self.client))

    def action_catalog(self) -> None:
        self.push_screen(CatalogModal(self.client))


def run_tui(
    *,
    project: "str | Path | None",
    remote: "str | None",
    base_url: "str | None" = None,
    model: "str | None" = None,
    api_key: "str | None" = None,
    client: "RuntimeClient | None" = None,
) -> int:
    """Start the interactive Textual interface. ``client`` is injectable for
    tests; when omitted a local client is built from the current project."""
    if client is None:
        from ..client import build_runtime_client

        client = build_runtime_client(
            remote=remote,
            with_model=True,
            project=project,
            base_url=base_url,
            model=model,
            api_key=api_key,
        )
    LinktoolsAIApp(client=client).run()
    return 0
