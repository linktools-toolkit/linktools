#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical persistence conversion and active model-context projection."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    BaseToolReturnPart,
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UploadedFile,
    UserPromptPart,
)

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..model import (
    LinkToolsUploadedFile,
    declared_uploaded_file_media_type,
)

_CONSUMED_BINARY_MARKER = "[binary content already consumed]"


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
                    content.append(_CONSUMED_BINARY_MARKER)
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


def _restore_uploaded_file_media_types(
    encoded: object,
    messages: Sequence[ModelMessage],
) -> object:
    if not isinstance(encoded, list) or len(encoded) != len(messages):
        return encoded
    for encoded_message, message in zip(encoded, messages):
        if not isinstance(encoded_message, dict):
            continue
        encoded_parts = encoded_message.get("parts")
        if not isinstance(encoded_parts, list):
            continue
        for encoded_part, part in zip(encoded_parts, message.parts):
            if not isinstance(encoded_part, dict):
                continue
            if isinstance(part, UserPromptPart):
                _restore_uploaded_file_media_types_in_value(
                    encoded_part.get("content"),
                    part.content,
                )
            elif isinstance(part, BaseToolReturnPart):
                _restore_uploaded_file_media_types_in_value(
                    encoded_part.get("content"),
                    part.content,
                )
    return encoded


def _restore_uploaded_file_media_types_in_value(
    encoded: object,
    original: object,
) -> None:
    if isinstance(original, UploadedFile):
        if isinstance(encoded, dict):
            encoded["media_type"] = declared_uploaded_file_media_type(original)
        return
    if isinstance(original, Mapping) and isinstance(encoded, dict):
        for key, original_item in original.items():
            if isinstance(key, str) and key in encoded:
                _restore_uploaded_file_media_types_in_value(
                    encoded[key],
                    original_item,
                )
        return
    if isinstance(original, Sequence) and not isinstance(
        original,
        (str, bytes, bytearray),
    ) and isinstance(encoded, list):
        for encoded_item, original_item in zip(encoded, original):
            _restore_uploaded_file_media_types_in_value(
                encoded_item,
                original_item,
            )


def encode_model_messages(messages: Sequence[ModelMessage]) -> bytes:
    value = ModelMessagesTypeAdapter.dump_python(
        list(messages),
        mode="json",
    )
    _restore_uploaded_file_media_types(value, messages)
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
    return _rehydrate_uploaded_files(messages, json_value)


def _rehydrate_uploaded_files(
    messages: Sequence[ModelMessage],
    encoded: object,
) -> tuple[ModelMessage, ...]:
    if not isinstance(encoded, list) or len(encoded) != len(messages):
        return tuple(messages)
    result: list[ModelMessage] = []
    for message, encoded_message in zip(messages, encoded):
        if not isinstance(encoded_message, dict):
            result.append(message)
            continue
        encoded_parts = encoded_message.get("parts")
        if not isinstance(encoded_parts, list):
            result.append(message)
            continue
        parts = []
        changed = False
        for index, part in enumerate(message.parts):
            if index >= len(encoded_parts) or not isinstance(
                encoded_parts[index],
                dict,
            ):
                parts.append(part)
                continue
            encoded_part = encoded_parts[index]
            if isinstance(part, (UserPromptPart, BaseToolReturnPart)):
                content = _rehydrate_uploaded_files_in_value(
                    part.content,
                    encoded_part.get("content"),
                )
                if content is not part.content:
                    parts.append(replace(part, content=content))
                    changed = True
                    continue
            parts.append(part)
        result.append(replace(message, parts=parts) if changed else message)
    return tuple(result)


def _rehydrate_uploaded_files_in_value(
    value: object,
    encoded: object,
) -> object:
    if isinstance(value, UploadedFile):
        if not isinstance(encoded, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        media_type = encoded.get("media_type")
        if media_type is not None and not isinstance(media_type, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return LinkToolsUploadedFile(
            value.file_id,
            value.provider_name,
            media_type=media_type,
            identifier=value.identifier,
            vendor_metadata=value.vendor_metadata,
        )
    if isinstance(value, Mapping) and isinstance(encoded, Mapping):
        changed = False
        result: dict[object, object] = {}
        for key, item in value.items():
            encoded_item = encoded.get(key)
            restored = _rehydrate_uploaded_files_in_value(item, encoded_item)
            result[key] = restored
            changed = changed or restored is not item
        return result if changed else value
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and isinstance(encoded, list)
    ):
        restored_items = [
            _rehydrate_uploaded_files_in_value(item, encoded_item)
            for item, encoded_item in zip(value, encoded)
        ]
        if any(restored is not item for restored, item in zip(restored_items, value)):
            return restored_items
    return value


__all__ = [
    "binary_content_usage",
    "decode_model_messages",
    "encode_model_messages",
    "project_transient_binary_content",
]
