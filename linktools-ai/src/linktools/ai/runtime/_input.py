#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned canonical transport for Pydantic AI user content."""

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TypeAlias, cast

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelRequest, UploadedFile, UserContent, UserPromptPart

from ..core import JsonValue, canonical_json_bytes, normalize_json_value, validate_user_prompt
from ..errors import AIError, ErrorCode

_TEXT_CODEC = "text"
_USER_CONTENT_CODEC = "pydantic-user-content-v1"
_WIRE_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)

_UserPromptInput: TypeAlias = str | Sequence[UserContent]
_RuntimeUserPrompt: TypeAlias = str


class UserPromptTransport(str):
    """Internal text value carrying its explicit durable codec."""

    __slots__ = ("codec",)

    def __new__(cls, value: str, codec: str) -> "UserPromptTransport":
        if not isinstance(value, str) or not isinstance(codec, str) or not codec:
            raise TypeError("user prompt transport is invalid")
        instance = str.__new__(cls, value)
        instance.codec = codec
        return instance

    def __add__(self, other: object) -> "UserPromptTransport":
        if not isinstance(other, str):
            return NotImplemented
        return UserPromptTransport(str.__add__(self, other), self.codec)


def prepare_user_prompt(value: _UserPromptInput) -> UserPromptTransport:
    """Convert Pydantic-native user content into durable text transport."""
    if isinstance(value, str):
        validate_user_prompt(value)
        return UserPromptTransport(value, _TEXT_CODEC)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    content = tuple(value)
    if not content:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if any(isinstance(item, UploadedFile) for item in content):
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={
                "field": "user_prompt",
                "reason": "uploaded_file_not_durable",
            },
        )
    payload = _encode_user_content(content)
    wire = canonical_json_bytes(payload).decode("utf-8")
    validate_user_prompt(wire)
    return UserPromptTransport(wire, _USER_CONTENT_CODEC)


def user_prompt_transport(value: str, codec: str = _TEXT_CODEC) -> UserPromptTransport:
    """Restore an internal prompt transport after a durable boundary."""
    validate_user_prompt(value)
    if codec not in {_TEXT_CODEC, _USER_CONTENT_CODEC}:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return UserPromptTransport(value, codec)


def _restore_user_prompt(value: str) -> str | tuple[UserContent, ...]:
    """Restore durable text transport into the prompt shape Pydantic AI accepts."""
    validate_user_prompt(value)
    codec = value.codec if isinstance(value, UserPromptTransport) else _TEXT_CODEC
    if codec == _TEXT_CODEC:
        return str(value)
    if codec != _USER_CONTENT_CODEC:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return _decode_user_content_wire(str(value))


def _decode_user_content_wire(value: str) -> tuple[UserContent, ...]:
    try:
        payload, end = json.JSONDecoder().raw_decode(value)
    except json.JSONDecodeError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    payload_text = value[:end]
    try:
        normalized = normalize_json_value(payload)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if canonical_json_bytes(normalized) != payload_text.encode("utf-8"):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = _decode_user_content(normalized)
    suffix = value[end:]
    return (*content, suffix) if suffix else content


def _encode_user_content(content: Sequence[UserContent]) -> dict[str, JsonValue]:
    request = ModelRequest(
        parts=[
            UserPromptPart(
                content=tuple(content),
                timestamp=_WIRE_TIMESTAMP,
            )
        ]
    )
    try:
        encoded = ModelMessagesTypeAdapter.dump_python([request], mode="json")
        normalized = normalize_json_value(encoded)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if (
        not isinstance(normalized, list)
        or len(normalized) != 1
        or not isinstance(normalized[0], dict)
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return {"message": normalized[0]}


def _decode_user_content(payload: dict[str, JsonValue]) -> tuple[UserContent, ...]:
    if set(payload) != {"message"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    message = payload["message"]
    if not isinstance(message, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        raw = canonical_json_bytes([message])
        messages = ModelMessagesTypeAdapter.validate_json(raw)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(messages) != 1 or not isinstance(messages[0], ModelRequest):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    request = messages[0]
    if len(request.parts) != 1 or not isinstance(request.parts[0], UserPromptPart):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    part = request.parts[0]
    if part.timestamp != _WIRE_TIMESTAMP or isinstance(part.content, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = tuple(cast(Sequence[UserContent], part.content))
    if not content or any(isinstance(item, UploadedFile) for item in content):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return content


__all__: tuple[str, ...] = ()
