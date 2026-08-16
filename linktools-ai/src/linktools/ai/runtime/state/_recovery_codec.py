#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical JSON codecs shared by filesystem and SQL recovery storage."""

from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import cast

from ...core import (
    ExecutionEventType,
    ExecutionStatus,
    JsonValue,
    StopReason,
    UsageMetrics,
)
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef
from ._contracts import (
    ConversationCursor,
    RecoveryConversationIntent,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
)

_RECOVERY_INPUT_VERSION = 3


def recovery_input_to_json(value: RecoveryExecutionInput) -> dict[str, JsonValue]:
    payload = _json_mapping(asdict(value))
    payload["version"] = _RECOVERY_INPUT_VERSION
    return payload


def recovery_input_from_json(value: JsonValue) -> RecoveryExecutionInput:
    raw = _mapping(value)
    payload = dict(raw)
    if payload.pop("version", None) != _RECOVERY_INPUT_VERSION:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    identity = _mapping(payload.get("idempotency"))
    try:
        payload["idempotency"] = RecoveryIdempotencyInput(**identity)
        return RecoveryExecutionInput(**payload)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def recovery_handoff_to_json(value: RecoveryTerminalHandoff | None) -> JsonValue:
    if value is None:
        return None
    return _json_value(asdict(value))


def recovery_handoff_from_json(value: JsonValue) -> RecoveryTerminalHandoff | None:
    if value is None:
        return None
    raw = _mapping(value)
    try:
        outcome = _recovery_outcome(_mapping(raw["outcome"]))
        raw_conversation = raw.get("conversation")
        conversation = None
        if raw_conversation is not None:
            conversation_raw = _mapping(raw_conversation)
            expected = conversation_raw.get("expected_cursor")
            conversation = RecoveryConversationIntent(
                str(conversation_raw["session_id"]),
                None if expected is None else _conversation_cursor(expected),
                _conversation_cursor(conversation_raw["next_cursor"]),
            )
        source = raw.get("source_step_run_id")
        return RecoveryTerminalHandoff(
            outcome,
            None if source is None else str(source),
            conversation,
        )
    except AIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _recovery_outcome(value: dict[str, object]) -> RecoveryTerminalOutcome:
    raw_ref = value.get("recovery_object_ref")
    object_ref = None
    if raw_ref is not None:
        reference = _mapping(raw_ref)
        object_ref = ObjectRef(
            str(reference["store_id"]),
            str(reference["key"]),
            str(reference["digest"]),
            int(reference["size"]),
        )
    usage = _mapping(value["usage"])
    return RecoveryTerminalOutcome(
        terminal_status=ExecutionStatus(str(value["terminal_status"])),
        error_code=None if value.get("error_code") is None else str(value["error_code"]),
        safe_error_details=_json_mapping(value.get("safe_error_details", {})),
        stop_reason=StopReason(str(value["stop_reason"])),
        output_schema_id=None if value.get("output_schema_id") is None else str(value["output_schema_id"]),
        output_schema_revision=None if value.get("output_schema_revision") is None else int(value["output_schema_revision"]),
        output_schema_fingerprint=None if value.get("output_schema_fingerprint") is None else str(value["output_schema_fingerprint"]),
        recovery_object_ref=object_ref,
        usage=UsageMetrics(
            model_requests=int(usage["model_requests"]),
            tool_calls=int(usage["tool_calls"]),
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            cache_read_tokens=int(usage["cache_read_tokens"]),
            cache_write_tokens=int(usage["cache_write_tokens"]),
        ),
        terminal_event_type=ExecutionEventType(str(value["terminal_event_type"])),
        terminal_event_payload=_json_mapping(value["terminal_event_payload"]),
        result_created_at=_timestamp(value["result_created_at"]),
    )


def _conversation_cursor(value: object) -> ConversationCursor:
    raw = _mapping(value)
    return ConversationCursor(str(raw["step_run_id"]))


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _json_mapping(value: object) -> dict[str, JsonValue]:
    raw = _mapping(value)
    return {str(key): _json_value(item) for key, item in raw.items()}


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return cast(JsonValue, value)
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return _json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _timestamp(value: object) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = [
    "recovery_handoff_from_json",
    "recovery_handoff_to_json",
    "recovery_input_from_json",
    "recovery_input_to_json",
]
