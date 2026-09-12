#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical persistence conversion and active model-context projection."""

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode


def _json_value(value: object, *, reading: bool) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if reading:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "model message persistence requires finite JSON values",
            )
        raise ValueError("model message persistence requires finite JSON values")
    if isinstance(value, list):
        return [_json_value(item, reading=reading) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                if reading:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                raise TypeError("model message persistence requires string object keys")
            result[key] = _json_value(item, reading=reading)
        return result
    if reading:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raise TypeError("model message persistence contains a non-JSON value")


def _consumed_binary_marker(value: BinaryContent) -> str:
    details = [value.media_type]
    if value.identifier:
        details.append(f"identifier={value.identifier}")
    return f"[binary content already consumed: {', '.join(details)}]"


def project_transient_binary_content(
    messages: Sequence[ModelMessage],
) -> tuple[ModelMessage, ...]:
    """Remove binary bodies already consumed by a complete model response."""
    values = tuple(messages)
    seen_complete_response = False
    changed = False
    projected_reversed: list[ModelMessage] = []

    for message in reversed(values):
        if isinstance(message, ModelResponse):
            if message.state == "complete":
                seen_complete_response = True
            projected_reversed.append(message)
            continue
        if not seen_complete_response or not isinstance(message, ModelRequest):
            projected_reversed.append(message)
            continue

        message_changed = False
        parts = []
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                parts.append(part)
                continue
            content = []
            part_changed = False
            for item in part.content:
                if isinstance(item, BinaryContent):
                    content.append(_consumed_binary_marker(item))
                    part_changed = True
                else:
                    content.append(item)
            if part_changed:
                parts.append(replace(part, content=content))
                message_changed = True
            else:
                parts.append(part)
        if message_changed:
            projected_reversed.append(replace(message, parts=parts))
            changed = True
        else:
            projected_reversed.append(message)

    if not changed:
        return values
    projected_reversed.reverse()
    return tuple(projected_reversed)


def binary_content_usage(messages: Sequence[ModelMessage]) -> tuple[int, int]:
    """Return BinaryContent part count and bytes in user-prompt content."""
    count = 0
    total_bytes = 0
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                continue
            for item in part.content:
                if isinstance(item, BinaryContent):
                    count += 1
                    total_bytes += len(item.data)
    return count, total_bytes


def encode_model_messages(messages: Sequence[ModelMessage]) -> bytes:
    value = ModelMessagesTypeAdapter.dump_python(
        list(messages),
        mode="json",
    )
    return canonical_json_bytes(_json_value(value, reading=False))


def decode_model_messages(raw: bytes) -> tuple[ModelMessage, ...]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    json_value = _json_value(value, reading=True)
    if canonical_json_bytes(json_value) != raw:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        messages = ModelMessagesTypeAdapter.validate_json(raw)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return tuple(messages)


__all__ = [
    "binary_content_usage",
    "decode_model_messages",
    "encode_model_messages",
    "project_transient_binary_content",
]
