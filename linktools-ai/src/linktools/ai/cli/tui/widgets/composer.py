#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The multi-line composer.

A :class:`TextArea` that submits on Enter (Shift+Enter for a literal newline)
and posts a :class:`Composer.Submitted` message. Slash commands are dispatched
by the screen, not here -- the composer only reports what was typed.

A subtle helper line under the area shows the keybinding hint and the active
session id; the screen updates it via :meth:`set_helper`."""

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static, TextArea

if TYPE_CHECKING:
    from textual.app import ComposeResult


class Composer(Vertical):
    """Multi-line text area + helper line.

    Enter submits the current text and clears the area; Shift+Enter inserts a
    literal newline. ``shift+enter`` is bound at higher priority than the
    submit binding so it wins and inserts instead of submitting. Empty submits
    are dropped."""

    DEFAULT_CSS = """
    Composer {
        dock: bottom;
        height: auto;
        max-height: 40%;
        min-height: 7;
        layout: vertical;
        background: $boost;
        border-top: solid $primary;
    }
    Composer TextArea {
        height: 5fr;
        min-height: 3;
        border: none;
        background: $boost;
    }
    Composer #composer-helper {
        height: 1;
        color: $text-disabled;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("shift+enter", "newline", "Newline", show=False, priority=True),
        Binding("enter", "submit", "Send", show=False, priority=True),
    ]

    class Submitted(Message):
        """The composer was submitted (Enter). ``value`` is the full text."""

        def __init__(self, composer: "Composer", value: str) -> None:
            super().__init__()
            self.composer = composer
            self.value = value

    def __init__(self, helper_text: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._helper_text = helper_text

    @property
    def text(self) -> str:
        try:
            return self.query_one("#composer-input", TextArea).text
        except Exception:
            return ""

    @text.setter
    def text(self, value: str) -> None:
        self.query_one("#composer-input", TextArea).text = value

    def compose(self) -> "ComposeResult":
        yield TextArea(id="composer-input")
        yield Static(self._helper_text, id="composer-helper")

    def action_newline(self) -> None:
        # Insert a literal newline at the cursor (Shift+Enter). TextArea.insert
        # takes a string; "\n" becomes a line break in the multi-line buffer.
        self.query_one("#composer-input", TextArea).insert("\n")

    def set_helper(self, text: str) -> None:
        self._helper_text = text
        try:
            self.query_one("#composer-helper", Static).update(text)
        except Exception:
            pass

    def focus_input(self) -> None:
        self.query_one("#composer-input", TextArea).focus()

    def clear_input(self) -> None:
        self.query_one("#composer-input", TextArea).text = ""

    def action_submit(self) -> None:
        value = self.text
        if not value.strip():
            self.clear_input()
            return
        self.clear_input()
        self.post_message(Composer.Submitted(self, value))
