#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversation view models.

Pure data the console renderer (and, later, the TUI conversation widget)
renders. The Runtime emits dict events; the renderer folds those events into
these items so the display layer never handles raw event dicts or internal
Runtime state. Tool/Subagent items default to a collapsed one-line summary."""

from dataclasses import dataclass

__all__ = [
    "UserItem",
    "AssistantItem",
    "ToolItem",
    "SubagentItem",
    "ApprovalItem",
    "ErrorItem",
]


@dataclass(slots=True)
class UserItem:
    """A user prompt in the conversation."""

    text: str


@dataclass(slots=True)
class AssistantItem:
    """A model response. ``streaming`` is True while tokens are still arriving
    so the renderer can show a partial line; flipped False on the final chunk."""

    markdown: str
    streaming: bool


@dataclass(slots=True)
class ToolItem:
    """One tool invocation, collapsed by default::

    ✓ read_file      22ms
    ⋯ bash           waiting approval
    """

    tool_call_id: str
    name: str
    source: str
    status: str
    summary: str
    duration_ms: "float | None"


@dataclass(slots=True)
class SubagentItem:
    """A delegated child-agent run surfaced in the parent conversation."""

    run_id: str
    agent_name: str
    status: str
    summary: "str | None"


@dataclass(slots=True)
class ApprovalItem:
    """An approval request the run paused on."""

    approval_id: str
    status: str


@dataclass(slots=True)
class ErrorItem:
    """A failure surfaced to the user as a conversation item."""

    message: str
