#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure session-history view projection over model messages."""

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


def project_history_message(
    message: ModelMessage,
) -> "tuple[SessionHistoryItem, ...]":
    """Project one model message into ordered session-history view items.

    Sequences are 1-based within the returned tuple; page assembly assigns
    the final running sequence across messages.
    """
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


def count_history_items(messages: "Sequence[ModelMessage]") -> int:
    """Count the session-history view items produced by ``messages``."""
    return sum(len(_projected_parts(message)) for message in messages)


def _projected_parts(
    message: ModelMessage,
) -> "tuple[tuple[str, JsonValue, str | None, str | None], ...]":
    if isinstance(message, ModelRequest):
        parts: list[tuple[str, JsonValue, "str | None", "str | None"]] = []
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


def _user_content(part: UserPromptPart) -> "JsonValue | None":
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
