#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen pure projections for persisted transcript view coordinates."""

from collections.abc import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ...core import JsonValue
from ..service_api import SessionHistoryItem

SESSION_HISTORY_VIEW_V1 = 1
EXECUTION_TRANSCRIPT_VIEW_V1 = 1


def project_session_history_message(
    message: ModelMessage,
) -> tuple[SessionHistoryItem, ...]:
    """Project one message into the frozen Session history view."""
    values: list[SessionHistoryItem] = []
    for item_kind, content, tool_name, tool_call_id in _projected_parts(message):
        values.append(
            SessionHistoryItem(
                len(values) + 1,
                item_kind,
                content,
                tool_name,
                tool_call_id,
            )
        )
    return tuple(values)


def count_session_history_items(messages: Sequence[ModelMessage]) -> int:
    """Count items emitted by the frozen Session history view."""
    return sum(len(_projected_parts(message)) for message in messages)


def project_execution_transcript_message(message: ModelMessage) -> tuple[str, ...]:
    """Project one canonical message into the execution transcript view."""
    if isinstance(message, ModelRequest):
        values: list[str] = []
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    if part.content:
                        values.append(part.content)
                else:
                    values.extend(
                        item
                        for item in part.content
                        if (
                            isinstance(item, str)
                            and item
                        )
                        or (
                            isinstance(item, TextContent)
                            and item.content
                        )
                    )
        return tuple(values)
    if isinstance(message, ModelResponse):
        return tuple(
            part.content
            for part in message.parts
            if isinstance(part, TextPart) and part.content
        )
    return ()


def count_execution_transcript_items(messages: Sequence[ModelMessage]) -> int:
    """Count items emitted by the frozen execution transcript view."""
    return sum(len(project_execution_transcript_message(message)) for message in messages)


def _projected_parts(
    message: ModelMessage,
) -> tuple[tuple[str, JsonValue, str | None, str | None], ...]:
    if isinstance(message, ModelRequest):
        parts: list[tuple[str, JsonValue, str | None, str | None]] = []
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                parts.append(("system", _json_content(part.content), None, None))
            elif isinstance(part, UserPromptPart):
                content = _user_content(part)
                if content is not None:
                    parts.append(("user", content, None, None))
            elif isinstance(part, ToolReturnPart):
                parts.append(
                    (
                        "tool_result",
                        _json_content(part.content),
                        part.tool_name,
                        part.tool_call_id,
                    )
                )
            elif isinstance(part, RetryPromptPart):
                parts.append(("retry", str(part.content), None, None))
        return tuple(parts)
    if isinstance(message, ModelResponse):
        parts = []
        for part in message.parts:
            if isinstance(part, TextPart):
                parts.append(("assistant", part.content, None, None))
            elif isinstance(part, ThinkingPart):
                parts.append(("thinking", part.content, None, None))
            elif isinstance(part, ToolCallPart):
                parts.append(
                    (
                        "tool_call",
                        part.args_as_dict(),
                        part.tool_name,
                        part.tool_call_id,
                    )
                )
        return tuple(parts)
    return ()


def _user_content(part: UserPromptPart) -> JsonValue | None:
    if isinstance(part.content, str):
        return part.content
    values: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, TextContent):
            values.append(item.content)
    return values if values else None


def _json_content(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_content(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_content(item) for key, item in value.items()}
    return str(value)


__all__ = [
    "EXECUTION_TRANSCRIPT_VIEW_V1",
    "SESSION_HISTORY_VIEW_V1",
    "count_execution_transcript_items",
    "count_session_history_items",
    "project_execution_transcript_message",
    "project_session_history_message",
]
