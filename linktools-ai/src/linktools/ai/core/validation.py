#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boundary size and identifier validation shared by Entries and Services."""

import re
from enum import Enum

from .errors import ErrorCode, LinktoolsAIError
from .json import JsonValue, canonical_json_bytes


def validate_tenant_id(value: str) -> str:
    return _text(value, 128, ErrorCode.REQUEST_FIELD_INVALID)


def validate_principal_id(value: str) -> str:
    return _text(value, 256, ErrorCode.REQUEST_FIELD_INVALID)


def validate_resource_id(value: str) -> str:
    return _text(value, 256, ErrorCode.REQUEST_FIELD_INVALID)


def validate_idempotency_key(value: str) -> str:
    return _text(value, 256, ErrorCode.IDEMPOTENCY_KEY_INVALID)


def validate_agent_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", value):
        raise LinktoolsAIError(ErrorCode.AGENT_ID_INVALID)
    return value


def validate_prompt(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 1024 * 1024:
        raise LinktoolsAIError(ErrorCode.PROMPT_TOO_LARGE)
    return value


def validate_external_payload(value: JsonValue) -> JsonValue:
    if len(canonical_json_bytes(value)) > 4 * 1024 * 1024:
        raise LinktoolsAIError(ErrorCode.EXTERNAL_RESULT_TOO_LARGE)
    return value


def validate_tool_arguments(value: JsonValue) -> JsonValue:
    if len(canonical_json_bytes(value)) > 1024 * 1024:
        raise LinktoolsAIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE)
    return value


def validate_observation_payload(value: JsonValue) -> JsonValue:
    if len(canonical_json_bytes(value)) > 1024 * 1024:
        raise LinktoolsAIError(ErrorCode.OBSERVATION_PAYLOAD_TOO_LARGE)
    return value


def validate_page_limit(value: int) -> int:
    if value < 1 or value > 200:
        raise LinktoolsAIError(ErrorCode.PAGE_LIMIT_INVALID)
    return value


def validate_shell_timeout(value: int) -> int:
    if value < 1 or value > 900:
        raise LinktoolsAIError(ErrorCode.TOOL_TIMEOUT_INVALID)
    return value


def validate_enum(value: str, enum_type: type[Enum]) -> str:
    try:
        enum_type(value)
    except ValueError as error:
        raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    return value


def _text(value: str, maximum: int, code: ErrorCode) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise LinktoolsAIError(code)
    return value


__all__ = [
    "validate_agent_id", "validate_enum", "validate_external_payload", "validate_idempotency_key",
    "validate_observation_payload", "validate_page_limit", "validate_principal_id", "validate_prompt",
    "validate_resource_id", "validate_shell_timeout", "validate_tenant_id", "validate_tool_arguments",
]
