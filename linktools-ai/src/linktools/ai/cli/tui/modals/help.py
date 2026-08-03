#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The help modal.

A scrollable overlay listing the keybindings and slash commands. Esc closes
it. Opened from the workspace via ``?`` or F1, or the command palette."""

from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import ComposeResult


_HELP = """[b]Keybindings[/b]

  Enter            send prompt
  Shift+Enter      insert newline
  Ctrl+L           clear conversation
  Ctrl+N           new session
  Esc / Ctrl+C     cancel active run
  Ctrl+P           command palette
  Ctrl+D           doctor report
  Ctrl+K           catalog (agents / skills / mcp)
  Ctrl+Q           quit
  F1               this help

[b]Slash commands[/b]

  /new <id>        switch to (or start) a session
  /session         show the current session id
  /clear           clear the conversation
  /cancel          cancel the active run
  /doctor          open the doctor report
  /catalog         open the catalog
  /help            this help
  /exit            quit
"""


class HelpModal(ModalScreen):
    """Keybindings + slash-command reference overlay."""

    BINDINGS = [Binding("escape,f1", "app.pop_screen", "Close")]

    DEFAULT_CSS = """
    HelpModal { align: center middle; }
    HelpModal > VerticalScroll {
        width: 64; height: 70%; border: round $primary;
        background: $surface; padding: 1 2;
    }
    HelpModal Static { height: auto; }
    """

    def compose(self) -> "ComposeResult":
        yield VerticalScroll(Static(_HELP, id="help-body"))
