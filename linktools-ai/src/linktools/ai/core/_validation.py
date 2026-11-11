#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boundary size and identifier validation shared by Entries and Services."""

import unicodedata
from enum import Enum

from ..errors import AIError, ErrorCode
from ._json import JsonValue, canonical_json_bytes, normalize_json_value


RESOURCE_ID_MAX_BYTES = 128


def validate_tenant_id(value: str) -> str:
    return _text(value, 128, ErrorCode.REQUEST_FIELD_INVALID)


def validate_persistence_namespace(value: str) -> str:
    return _text(value, 256, ErrorCode.REQUEST_FIELD_INVALID)


def validate_asset_namespace(value: str) -> str:
    return _text(value, 128, ErrorCode.REQUEST_FIELD_INVALID)


def validate_memory_scope(value: str) -> str:
    return _text(value, 256, ErrorCode.REQUEST_FIELD_INVALID)


def validate_asset_kind(value: str) -> str:
    return _text(value, 128, ErrorCode.REQUEST_FIELD_INVALID)


def validate_capability_id(value: str) -> str:
    """Validate a nonblank capability identity of at most 256 UTF-8 bytes."""
    if not isinstance(value, str) or not value.strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        if len(value.encode("utf-8")) > 256:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    return value


def validate_capability_revision(value: int) -> int:
    """Validate a capability's positive signed-64-bit behavior revision."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 9223372036854775807:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value


def validate_capability_write_key(value: str) -> str:
    """Keep an exact nonempty write key of at most 256 UTF-8 bytes."""
    if not isinstance(value, str) or not value:
        raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
    try:
        if len(value.encode("utf-8")) > 256:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID) from error
    return value


def validate_capability_provider(value: str) -> str:
    return _text(value, 128, ErrorCode.REQUEST_FIELD_INVALID)


def validate_principal_kind(value: str) -> str:
    return _text(value, 64, ErrorCode.REQUEST_FIELD_INVALID)


def validate_lease_owner(value: str) -> str:
    return _text(value, 256, ErrorCode.REQUEST_FIELD_INVALID)


def validate_lease_seconds(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 3600:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value


def validate_principal_id(value: str) -> str:
    return _text(value, 256, ErrorCode.REQUEST_FIELD_INVALID)


def validate_resource_id(value: str) -> str:
    """Validate an exact resource identity of at most 128 UTF-8 bytes."""
    return _text(value, RESOURCE_ID_MAX_BYTES, ErrorCode.REQUEST_FIELD_INVALID)


def validate_idempotency_key(value: str) -> str:
    return _text(value, 256, ErrorCode.IDEMPOTENCY_KEY_INVALID)


def validate_logical_id(value: str) -> str:
    """Validate an exact Agent or Skill identity without normalizing it."""
    if not isinstance(value, str):
        raise TypeError("logical id must be a string")
    if not value or len(value) > 64:
        raise ValueError("logical id must contain 1..64 characters")
    if value != value.strip() or any(
        unicodedata.category(character) == "Cc"
        or character in {"\\", "*"}
        for character in value
    ):
        raise ValueError("logical id contains a reserved character")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("logical id must be valid UTF-8") from error
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("logical id contains an invalid path segment")
    return value


def validate_agent_id(value: str) -> str:
    try:
        return validate_logical_id(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.AGENT_ID_INVALID) from error


def validate_user_prompt(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 1024 * 1024:
        raise AIError(ErrorCode.PROMPT_TOO_LARGE)
    return value


def validate_external_payload(value: JsonValue) -> JsonValue:
    normalized = normalize_json_value(value)
    if len(canonical_json_bytes(normalized)) > 4 * 1024 * 1024:
        raise AIError(ErrorCode.EXTERNAL_RESULT_TOO_LARGE)
    return normalized


def validate_tool_arguments(value: JsonValue) -> JsonValue:
    normalized = normalize_json_value(value)
    if len(canonical_json_bytes(normalized)) > 1024 * 1024:
        raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE)
    return normalized


def validate_observation_payload(value: JsonValue) -> JsonValue:
    normalized = normalize_json_value(value)
    if len(canonical_json_bytes(normalized)) > 1024 * 1024:
        raise AIError(ErrorCode.OBSERVATION_PAYLOAD_TOO_LARGE)
    return normalized


def validate_page_limit(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 200:
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
    return value


def validate_shell_timeout(value: int) -> int:
    if value < 1 or value > 900:
        raise AIError(ErrorCode.TOOL_TIMEOUT_INVALID)
    return value


def validate_enum(value: str, enum_type: type[Enum]) -> str:
    try:
        enum_type(value)
    except ValueError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    return value


def _text(value: str, maximum: int, code: ErrorCode) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise AIError(code)
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise AIError(code) from error
    if size > maximum:
        raise AIError(code)
    return value


__all__ = [
    "RESOURCE_ID_MAX_BYTES",
    "validate_agent_id",
    "validate_logical_id",
    "validate_asset_kind",
    "validate_asset_namespace",
    "validate_capability_id",
    "validate_capability_revision",
    "validate_capability_write_key",
    "validate_capability_provider",
    "validate_enum",
    "validate_external_payload",
    "validate_idempotency_key",
    "validate_lease_owner",
    "validate_lease_seconds",
    "validate_memory_scope",
    "validate_observation_payload",
    "validate_page_limit",
    "validate_persistence_namespace",
    "validate_principal_id",
    "validate_principal_kind",
    "validate_resource_id",
    "validate_shell_timeout",
    "validate_tenant_id",
    "validate_tool_arguments",
    "validate_user_prompt",
]
