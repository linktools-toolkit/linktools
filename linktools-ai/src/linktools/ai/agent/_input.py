#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical transport for Pydantic AI user content."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TypeAlias, cast

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelRequest, UserContent, UserPromptPart

from ..core import JsonValue, canonical_json_bytes, normalize_json_value, validate_user_prompt
from ..errors import AIError, ErrorCode

_USER_CONTENT_KIND = "pydantic-user-content"
_USER_CONTENT_VERSION = 1
_USER_CONTENT_PREFIX = "linktools.ai:user-content:v1:"
_USER_TEXT_ESCAPE_PREFIX = "linktools.ai:user-content:text:"
_WIRE_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)
_DIGEST_SIZE = 64

_UserPromptInput: TypeAlias = str | Sequence[UserContent]
_RuntimeUserPrompt: TypeAlias = str | Mapping[str, JsonValue]


def _prepare_user_prompt(value: _UserPromptInput) -> _RuntimeUserPrompt:
    """Convert the public Pydantic-native prompt surface into durable transport."""
    if isinstance(value, str):
        validate_user_prompt(value)
        if value.startswith((_USER_CONTENT_PREFIX, _USER_TEXT_ESCAPE_PREFIX)):
            escaped = f"{_USER_TEXT_ESCAPE_PREFIX}{value}"
            validate_user_prompt(escaped)
            return escaped
        return value
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    content = tuple(value)
    if not content:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    payload = _encode_user_content(content)
    raw = canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    wire = f"{_USER_CONTENT_PREFIX}{digest}\n{raw.decode('utf-8')}"
    validate_user_prompt(wire)
    return wire


def _normalize_runtime_user_prompt(value: object) -> _RuntimeUserPrompt:
    """Validate a Runtime prompt without changing its durable identity."""
    if isinstance(value, str):
        validate_user_prompt(value)
        return value
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        normalized = normalize_json_value(dict(value))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    content = _decode_user_content(normalized, integrity_error=False)
    canonical = _encode_user_content(content)
    _validate_payload_size(canonical)
    return canonical


def _restore_user_prompt(value: _RuntimeUserPrompt) -> str | tuple[UserContent, ...]:
    """Restore durable transport into exactly the prompt shape Pydantic AI accepts."""
    if isinstance(value, str):
        validate_user_prompt(value)
        if value.startswith(_USER_TEXT_ESCAPE_PREFIX):
            restored = value[len(_USER_TEXT_ESCAPE_PREFIX) :]
            validate_user_prompt(restored)
            return restored
        if not value.startswith(_USER_CONTENT_PREFIX):
            return value
        return _decode_user_content_wire(value)
    try:
        normalized = normalize_json_value(dict(value))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _decode_user_content(normalized, integrity_error=True)


def _decode_user_content_wire(value: str) -> tuple[UserContent, ...]:
    encoded = value[len(_USER_CONTENT_PREFIX) :]
    digest, separator, payload_text = encoded.partition("\n")
    if (
        not separator
        or len(digest) != _DIGEST_SIZE
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        raw = payload_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if hashlib.sha256(raw).hexdigest() != digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    try:
        normalized = normalize_json_value(payload)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict) or canonical_json_bytes(normalized) != raw:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _decode_user_content(normalized, integrity_error=True)


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
    return {
        "kind": _USER_CONTENT_KIND,
        "version": _USER_CONTENT_VERSION,
        "message": normalized[0],
    }


def _decode_user_content(
    payload: dict[str, JsonValue],
    *,
    integrity_error: bool,
) -> tuple[UserContent, ...]:
    code = (
        ErrorCode.STORAGE_INTEGRITY_ERROR
        if integrity_error
        else ErrorCode.REQUEST_FIELD_INVALID
    )
    if set(payload) != {"kind", "version", "message"}:
        raise AIError(code)
    if (
        payload["kind"] != _USER_CONTENT_KIND
        or payload["version"] != _USER_CONTENT_VERSION
    ):
        raise AIError(code)
    message = payload["message"]
    if not isinstance(message, dict):
        raise AIError(code)
    try:
        raw = canonical_json_bytes([message])
        messages = ModelMessagesTypeAdapter.validate_json(raw)
    except (TypeError, ValueError) as error:
        raise AIError(code) from error
    if len(messages) != 1 or not isinstance(messages[0], ModelRequest):
        raise AIError(code)
    request = messages[0]
    if len(request.parts) != 1 or not isinstance(request.parts[0], UserPromptPart):
        raise AIError(code)
    part = request.parts[0]
    if part.timestamp != _WIRE_TIMESTAMP or isinstance(part.content, str):
        raise AIError(code)
    content = tuple(cast(Sequence[UserContent], part.content))
    if not content:
        raise AIError(code)
    try:
        canonical = _encode_user_content(content)
    except AIError as error:
        raise AIError(code) from error
    if canonical != payload:
        raise AIError(code)
    return content


def _validate_payload_size(payload: dict[str, JsonValue]) -> None:
    encoded = canonical_json_bytes(payload).decode("utf-8")
    validate_user_prompt(encoded)


__all__: tuple[str, ...] = ()
