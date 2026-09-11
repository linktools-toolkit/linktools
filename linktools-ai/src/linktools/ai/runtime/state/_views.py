#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen pure projections for persisted transcript view coordinates."""

from collections.abc import Mapping, Sequence
from typing import cast

from pydantic_ai import ModelMessagesTypeAdapter
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

from ...core import JsonValue, normalize_json_value
from ...errors import AIError, ErrorCode
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
                    for item in part.content:
                        if isinstance(item, str):
                            if item:
                                values.append(item)
                        elif isinstance(item, TextContent) and item.content:
                            values.append(item.content)
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
    serialized = _serialized_parts(message)
    raw_parts = tuple(message.parts)
    if len(serialized) != len(raw_parts):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values: list[tuple[str, JsonValue, str | None, str | None]] = []
    for part, payload in zip(raw_parts, serialized, strict=True):
        if isinstance(part, SystemPromptPart):
            values.append(("system", _payload_content(payload), None, None))
            continue
        if isinstance(part, UserPromptPart):
            values.append(("user", _user_content(part, payload), None, None))
            continue
        if isinstance(part, ToolReturnPart):
            values.append(
                (
                    "tool_result",
                    _payload_content(payload),
                    part.tool_name,
                    part.tool_call_id,
                )
            )
            continue
        if isinstance(part, RetryPromptPart):
            values.append(
                (
                    "retry",
                    _payload_content(payload),
                    part.tool_name,
                    part.tool_call_id,
                )
            )
            continue
        if isinstance(part, TextPart):
            values.append(("assistant", part.content, None, None))
            continue
        if isinstance(part, ThinkingPart):
            values.append(("thinking", part.content, None, None))
            continue
        if isinstance(part, ToolCallPart):
            values.append(
                (
                    "tool_call",
                    normalize_json_value(part.args_as_dict()),
                    part.tool_name,
                    part.tool_call_id,
                )
            )
            continue
        values.append(
            (
                _part_kind(part, payload),
                payload,
                _optional_string(getattr(part, "tool_name", None)),
                _optional_string(getattr(part, "tool_call_id", None)),
            )
        )
    return tuple(values)


def _serialized_parts(message: ModelMessage) -> tuple[dict[str, JsonValue], ...]:
    try:
        value = ModelMessagesTypeAdapter.dump_python([message], mode="json")
        normalized = normalize_json_value(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if (
        not isinstance(normalized, list)
        or len(normalized) != 1
        or not isinstance(normalized[0], Mapping)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    parts = normalized[0].get("parts")
    if not isinstance(parts, list) or any(not isinstance(part, Mapping) for part in parts):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return tuple(cast(dict[str, JsonValue], dict(part)) for part in parts)


def _payload_content(payload: Mapping[str, JsonValue]) -> JsonValue:
    return payload.get("content")


def _user_content(part: UserPromptPart, payload: Mapping[str, JsonValue]) -> JsonValue:
    if isinstance(part.content, str):
        return part.content
    text_values: list[str] = []
    text_only = True
    for item in part.content:
        if isinstance(item, str):
            text_values.append(item)
        elif isinstance(item, TextContent):
            text_values.append(item.content)
        else:
            text_only = False
            break
    if text_only:
        return text_values
    return _payload_content(payload)


def _part_kind(part: object, payload: Mapping[str, JsonValue]) -> str:
    for candidate in (
        getattr(part, "part_kind", None),
        getattr(part, "kind", None),
        payload.get("part_kind"),
        payload.get("kind"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown_part"


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "EXECUTION_TRANSCRIPT_VIEW_V1",
    "SESSION_HISTORY_VIEW_V1",
    "count_execution_transcript_items",
    "count_session_history_items",
    "project_execution_transcript_message",
    "project_session_history_message",
]
