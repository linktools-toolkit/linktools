#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical persistence conversion and active model-context projection."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from pydantic import TypeAdapter
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    BaseToolReturnPart,
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    MultiModalContent,
    UserPromptPart,
    is_multi_modal_content,
)

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode

_CONSUMED_BINARY_MARKER = "[binary content already consumed]"
_TOOL_RETURN_CONTENT_HINT = "_linktools_tool_return_content"
_TOOL_RETURN_CONTENT_HINT_VERSION = 1
_TOOL_RETURN_MAPPING_MASK = "__linktools_plain_mapping__"
_MULTIMODAL_ADAPTER = TypeAdapter(MultiModalContent)


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


def encode_model_messages(messages: Sequence[ModelMessage]) -> bytes:
    value = _json_value(
        ModelMessagesTypeAdapter.dump_python(list(messages), mode="json"),
        reading=False,
    )
    if not isinstance(value, list) or len(value) != len(messages):
        raise RuntimeError("model message persistence dump is invalid")
    _annotate_tool_return_content(messages, value)
    return canonical_json_bytes(value)


def decode_model_messages(raw: bytes) -> tuple[ModelMessage, ...]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    json_value = _json_value(value, reading=True)
    if canonical_json_bytes(json_value) != raw:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not isinstance(json_value, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    hints = _mask_tool_return_content(json_value)
    try:
        messages = tuple(
            ModelMessagesTypeAdapter.validate_json(canonical_json_bytes(json_value))
        )
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    _restore_tool_return_content(messages, hints)
    return messages


def _annotate_tool_return_content(
    messages: Sequence[ModelMessage],
    value: list[JsonValue],
) -> None:
    for message, encoded_message in zip(messages, value, strict=True):
        if not isinstance(encoded_message, dict):
            raise RuntimeError("model message persistence dump is invalid")
        encoded_parts = encoded_message.get("parts")
        if not isinstance(encoded_parts, list) or len(encoded_parts) != len(message.parts):
            raise RuntimeError("model message persistence parts are invalid")
        for part, encoded_part in zip(message.parts, encoded_parts, strict=True):
            if not isinstance(part, BaseToolReturnPart):
                continue
            if not isinstance(encoded_part, dict) or "content" not in encoded_part:
                raise RuntimeError("tool return persistence dump is invalid")
            mapping_paths = _ambiguous_mapping_paths(
                part.content,
                encoded_part["content"],
            )
            if mapping_paths:
                encoded_part[_TOOL_RETURN_CONTENT_HINT] = {
                    "version": _TOOL_RETURN_CONTENT_HINT_VERSION,
                    "mapping_paths": [list(path) for path in mapping_paths],
                }


def _ambiguous_mapping_paths(
    source: object,
    encoded: JsonValue,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str | int, ...], ...]:
    if is_multi_modal_content(source):
        return ()
    paths: list[tuple[str | int, ...]] = []
    if isinstance(source, Mapping) and isinstance(encoded, dict):
        if _decodes_as_multimodal(encoded):
            paths.append(path)
        for key in sorted(encoded):
            if key in source:
                paths.extend(
                    _ambiguous_mapping_paths(
                        source[key],
                        encoded[key],
                        path + (key,),
                    )
                )
    elif (
        isinstance(source, Sequence)
        and not isinstance(source, (str, bytes, bytearray))
        and isinstance(encoded, list)
    ):
        for index, (item, encoded_item) in enumerate(
            zip(source, encoded, strict=False)
        ):
            paths.extend(
                _ambiguous_mapping_paths(
                    item,
                    encoded_item,
                    path + (index,),
                )
            )
    return tuple(paths)


def _decodes_as_multimodal(value: Mapping[str, JsonValue]) -> bool:
    if not isinstance(value.get("kind"), str) or not any(
        field in value for field in ("url", "media_type", "file_id")
    ):
        return False
    try:
        decoded = _MULTIMODAL_ADAPTER.validate_python(dict(value))
    except ValueError:
        return False
    return is_multi_modal_content(decoded)


def _mask_tool_return_content(
    value: list[JsonValue],
) -> dict[tuple[int, int], dict[tuple[str | int, ...], str]]:
    hints: dict[tuple[int, int], dict[tuple[str | int, ...], str]] = {}
    for message_index, message in enumerate(value):
        if not isinstance(message, dict):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict) or _TOOL_RETURN_CONTENT_HINT not in part:
                continue
            marker = part.pop(_TOOL_RETURN_CONTENT_HINT)
            paths = _decode_tool_return_content_hint(marker)
            hints[(message_index, part_index)] = _mask_mapping_paths(
                part.get("content"),
                paths,
            )
    return hints


def _decode_tool_return_content_hint(
    marker: JsonValue,
) -> tuple[tuple[str | int, ...], ...]:
    if not isinstance(marker, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = marker.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != _TOOL_RETURN_CONTENT_HINT_VERSION
    ):
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    raw_paths = marker.get("mapping_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    paths: list[tuple[str | int, ...]] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        path: list[str | int] = []
        for segment in raw_path:
            if isinstance(segment, str):
                path.append(segment)
            elif (
                isinstance(segment, int)
                and not isinstance(segment, bool)
                and segment >= 0
            ):
                path.append(segment)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        paths.append(tuple(path))
    if len(set(paths)) != len(paths):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return tuple(paths)


def _mask_mapping_paths(
    value: object,
    paths: Sequence[tuple[str | int, ...]],
) -> dict[tuple[str | int, ...], str]:
    targets = set(paths)
    originals: dict[tuple[str | int, ...], str] = {}

    def visit(candidate: object, path: tuple[str | int, ...]) -> None:
        if path in targets:
            if not isinstance(candidate, dict):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            kind = candidate.get("kind")
            if not isinstance(kind, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            originals[path] = kind
            candidate["kind"] = _TOOL_RETURN_MAPPING_MASK
        if isinstance(candidate, Mapping):
            for key, item in candidate.items():
                if isinstance(key, str):
                    visit(item, path + (key,))
        elif isinstance(candidate, Sequence) and not isinstance(
            candidate,
            (str, bytes, bytearray),
        ):
            for index, item in enumerate(candidate):
                visit(item, path + (index,))

    visit(value, ())
    if originals.keys() != targets:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return originals


def _restore_tool_return_content(
    messages: Sequence[ModelMessage],
    hints: Mapping[
        tuple[int, int],
        Mapping[tuple[str | int, ...], str],
    ],
) -> None:
    for (message_index, part_index), paths in hints.items():
        if message_index >= len(messages):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        message = messages[message_index]
        if part_index >= len(message.parts):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        part = message.parts[part_index]
        if not isinstance(part, BaseToolReturnPart):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _restore_mapping_paths(part.content, paths)


def _restore_mapping_paths(
    value: object,
    paths: Mapping[tuple[str | int, ...], str],
) -> None:
    restored: set[tuple[str | int, ...]] = set()

    def visit(candidate: object, path: tuple[str | int, ...]) -> None:
        kind = paths.get(path)
        if kind is not None:
            if not isinstance(candidate, dict):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if candidate.get("kind") != _TOOL_RETURN_MAPPING_MASK:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            candidate["kind"] = kind
            restored.add(path)
        if isinstance(candidate, Mapping):
            for key, item in candidate.items():
                if isinstance(key, str):
                    visit(item, path + (key,))
        elif isinstance(candidate, Sequence) and not isinstance(
            candidate,
            (str, bytes, bytearray),
        ):
            for index, item in enumerate(candidate):
                visit(item, path + (index,))

    visit(value, ())
    if restored != set(paths):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = [
    "binary_content_usage",
    "decode_model_messages",
    "encode_model_messages",
    "project_transient_binary_content",
]
