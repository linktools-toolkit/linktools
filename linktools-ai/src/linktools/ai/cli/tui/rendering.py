#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rich renderables for conversation turns.

Each :class:`ConversationTurn` maps to a :class:`rich.console.RenderableType`
(a panel/line/centered block) rather than raw markup. Building renderables
(not strings) lets the conversation area lay out cleanly under resize and
keeps escaping at the text-leaf level where it belongs.

Untrusted text (model output, tool args, error messages) is escaped before
being placed in a Text/Panel; the renderer never interpolates it into a markup
string."""

from rich.align import Align
from rich.panel import Panel
from rich.markup import escape
from rich.text import Text
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import RenderableType

    from .conversation import ConversationTurn

_TOOL_STATUS = {
    "running": ("…", "yellow"),
    "ok": ("✓", "green"),
    "error": ("✗", "red"),
    "paused": ("⏸", "yellow"),
}


def render_turn(turn: "ConversationTurn") -> "RenderableType":
    """Render one conversation turn to a Rich renderable."""
    kind = turn.kind
    if kind == "user":
        return Align.center(
            Panel(
                Text(escape(turn.text), style="bold"),
                border_style="blue",
                title="you",
                title_align="left",
                padding=(0, 1),
            )
        )
    if kind == "assistant":
        body = turn.text or ("…" if turn.streaming else "")
        return Text(body)
    if kind == "tool":
        return _render_tool(turn)
    if kind == "error":
        return Text(f"✗ {escape(turn.message)}", style="red")
    if kind == "status":
        return Text(escape(turn.message), style=turn.tone)
    return Text(escape(repr(turn)))


def _render_tool(turn: "ConversationTurn") -> "RenderableType":
    mark, color = _TOOL_STATUS.get(turn.status, ("?", "white"))
    label = Text.assemble((f"{mark} ", color), (escape(turn.name), f"bold {color}"))
    detail = getattr(turn, "detail", None)
    if detail:
        label.append(Text(f"  {escape(detail)}", style="dim"))
    return label
