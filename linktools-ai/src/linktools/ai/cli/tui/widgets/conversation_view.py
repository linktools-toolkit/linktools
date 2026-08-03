#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The conversation area widget.

A :class:`VerticalScroll` holding a single :class:`Static` whose renderable is
the whole conversation. Re-rendering replaces the renderable in place (no
mount/unmount), so there are no widget-id collisions or registry races even
under tight streaming updates. Streaming assistant text mutates the source
turn and the caller calls ``refresh_from`` again."""

from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from ..conversation import Conversation
from ..rendering import render_turn

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ConversationView(VerticalScroll):
    """Renders :class:`Conversation` turns into a single updating renderable.

    ``bind`` wires a conversation the screen owns; ``refresh_from`` rebuilds
    the bound Static's renderable from ``conversation.turns``. Safe to call
    synchronously from message handlers."""

    DEFAULT_CSS = """
    ConversationView {
        scrollbar-size: 1 1;
        padding: 0 1;
        background: $surface;
    }
    ConversationView > Static { height: auto; }
    ConversationView .placeholder {
        color: $text-disabled;
        text-align: center;
        padding: 2 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._conversation: "Conversation | None" = None

    def bind(self, conversation: Conversation) -> None:
        self._conversation = conversation

    def compose(self) -> "ComposeResult":
        yield Static(self._placeholder(), id="conv-body")

    def _placeholder(self) -> RenderableType:
        return Text("Start a conversation by typing below.", style="dim")

    def refresh_from(self) -> None:
        """Rebuild the renderable from the bound conversation."""
        body = self.query_one("#conv-body", Static)
        conv = self._conversation
        if conv is None or not conv.turns:
            body.update(self._placeholder())
            return
        renderables = tuple(render_turn(turn) for turn in conv.turns)
        body.update(Group(*renderables))
        # Jump to the latest row unless the user scrolled up.
        if self.scroll_y >= self.max_scroll_y - 1:
            self.scroll_end(animate=False)
