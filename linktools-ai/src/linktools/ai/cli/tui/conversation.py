#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversation model for the TUI.

Folds the dict events streamed from ``RuntimeClient.run_stream`` into a list of
:class:`ConversationTurn` items. Each turn is an addressable, renderable block
(user prompt, assistant text, tool call, error, status note) instead of a flat
log line -- so the conversation area can render structured rows, re-render on
resize, and let a sidebar reflect per-turn state.

The folder is pure data + event handling. It owns no widgets and posts no
messages; the screen pulls ``turns`` and renders them. A tool call arrives as
separate ``tool`` events with the same ``tool_call_id`` (phase start/end, ok/
err); the folder accumulates them into one :class:`ToolTurn`.
"""

from collections.abc import Mapping

__all__ = [
    "ConversationTurn",
    "UserTurn",
    "AssistantTurn",
    "ToolTurn",
    "ErrorTurn",
    "StatusTurn",
    "Conversation",
]


class ConversationTurn:
    """Base turn. ``kind`` is the discriminator the renderer switches on."""

    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind


class UserTurn(ConversationTurn):
    """A submitted prompt."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        super().__init__("user")
        self.text = text


class AssistantTurn(ConversationTurn):
    """Model output. ``streaming`` is True until the run's final text chunk;
    the renderer re-renders the row as tokens arrive."""

    __slots__ = ("text", "streaming")

    def __init__(self, text: str = "", *, streaming: bool = True) -> None:
        super().__init__("assistant")
        self.text = text
        self.streaming = streaming


class ToolTurn(ConversationTurn):
    """One tool invocation accumulated across phase events.

    ``status`` is one of ``running``/``ok``/``error``/``paused``. ``detail``
    holds the optional result/error summary surfaced by later events."""

    __slots__ = ("tool_call_id", "name", "status", "detail")

    def __init__(self, tool_call_id: str, name: str, status: str = "running") -> None:
        super().__init__("tool")
        self.tool_call_id = tool_call_id
        self.name = name
        self.status = status
        self.detail: "str | None" = None


class ErrorTurn(ConversationTurn):
    """A run failure surfaced inline."""

    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        super().__init__("error")
        self.message = message


class StatusTurn(ConversationTurn):
    """A non-content status line (cancelled, resumed, paused boundary)."""

    __slots__ = ("message", "tone")

    def __init__(self, message: str, *, tone: str = "dim") -> None:
        super().__init__("status")
        self.message = message
        self.tone = tone


class Conversation:
    """Accumulates streamed Runtime events into renderable turns.

    The folder appends a fresh :class:`AssistantTurn` the first time a ``text``
    event arrives after a non-text turn, and keeps appending to it while text
    events keep the same run flowing. A subsequent user/tool/error event starts
    a new block. Tool events with a known ``tool_call_id`` update the existing
    :class:`ToolTurn` in place rather than stacking duplicates."""

    def __init__(self) -> None:
        self._turns: "list[ConversationTurn]" = []

    @property
    def turns(self) -> "tuple[ConversationTurn, ...]":
        return tuple(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def load_from_turns(
        self, per_turn_messages: "tuple[tuple[Mapping[str, object], ...], ...]"
    ) -> None:
        """Rebuild the conversation from a session's per-turn message history.

        ``per_turn_messages`` is the nested tuple returned by
        ``RuntimeClient.get_session_messages``: one inner tuple per turn, each
        holding that turn's recorded model messages (request/response dicts
        with ``parts``). Each part's text content is folded into a user or
        assistant turn by message kind; non-text parts are skipped (tool calls
        are not reconstructed here -- the audit view carries them separately).
        Unknown shapes degrade gracefully to nothing rather than failing."""
        self._turns.clear()
        for messages in per_turn_messages:
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                kind = message.get("kind")
                for part in message.get("parts", ()) or ():
                    if not isinstance(part, Mapping):
                        continue
                    if part.get("type") != "text":
                        continue
                    text = str(part.get("content") or part.get("text") or "")
                    if not text:
                        continue
                    if kind == "request":
                        self.add_user(text)
                    elif kind == "response":
                        self._append_text(text)

    def add_user(self, text: str) -> None:
        self._turns.append(UserTurn(text))

    def add_error(self, message: str) -> None:
        self._turns.append(ErrorTurn(message))

    def add_status(self, message: str, *, tone: str = "dim") -> None:
        self._turns.append(StatusTurn(message, tone=tone))

    def apply(self, event: "Mapping[str, object]") -> None:
        """Fold one Runtime stream event into the conversation.

        Unknown event types are ignored: the runtime may emit forward-compat
        events a given TUI build does not render."""
        kind = event.get("type")
        if kind == "text":
            self._append_text(str(event.get("text", "")))
        elif kind == "tool":
            self._apply_tool(event)
        elif kind == "failed":
            self._turns.append(
                ErrorTurn(
                    f"{event.get('error_type', 'error')}: {event.get('message', '')}".strip()
                )
            )
        elif kind == "cancelled":
            self._turns.append(StatusTurn("run cancelled", tone="yellow"))
        elif kind == "paused":
            self._turns.append(StatusTurn("paused for approval", tone="yellow"))
        elif kind == "resumed":
            self._turns.append(StatusTurn("resumed", tone="dim"))

    def _append_text(self, text: str) -> None:
        last = self._turns[-1] if self._turns else None
        if isinstance(last, AssistantTurn):
            last.text += text
            return
        self._turns.append(AssistantTurn(text))

    def _apply_tool(self, event: "Mapping[str, object]") -> None:
        call_id = str(event.get("tool_call_id") or event.get("id") or "")
        name = str(event.get("name", "?"))
        phase = str(event.get("phase", ""))
        ok = bool(event.get("ok"))
        detail = event.get("detail") or event.get("message")
        if call_id:
            for turn in reversed(self._turns):
                if isinstance(turn, ToolTurn) and turn.tool_call_id == call_id:
                    if detail:
                        turn.detail = str(detail)
                    if phase == "end":
                        turn.status = "ok" if ok else "error"
                    elif phase == "paused":
                        turn.status = "paused"
                    return
        status = (
            "paused"
            if phase == "paused"
            else (
                "ok"
                if phase == "end" and ok
                else ("error" if phase == "end" else "running")
            )
        )
        turn = ToolTurn(call_id or name, name, status=status)
        if detail:
            turn.detail = str(detail)
        self._turns.append(turn)
