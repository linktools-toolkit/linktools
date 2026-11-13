#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact tool failures reconstructed from immutable keyed native events."""

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai_harness.step_persistence import StepEvent

from ..core import JsonValue, canonical_json_bytes, deterministic_id, validate_resource_id
from ..errors import AIError, ErrorCode
from ..steps import (decode_native_store_key, encode_native_store_key,
                     validate_native_replay_key, validate_native_store_key)
from ..storage import StoredPayload


@dataclass(frozen=True, slots=True)
class NativeToolErrorReference:
    """A fixed event locator; error text and tool identity stay with their owners."""

    group_id: str
    store_key: bytes
    run_id: str
    replay_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("native event group is required")
        validate_native_store_key(self.store_key)
        validate_resource_id(self.run_id)
        if validate_native_replay_key(self.replay_key) is None:
            raise ValueError("native tool error requires a keyed event")

    def to_json(self) -> dict[str, JsonValue]:
        return {"$native_tool_error": {"group_id": self.group_id,
                "store_key": encode_native_store_key(self.store_key),
                "run_id": self.run_id, "replay_key": self.replay_key}}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "NativeToolErrorReference":
        if not isinstance(value, Mapping) or set(value) != {"$native_tool_error"}:
            raise ValueError("native tool error wire is invalid")
        fields = value["$native_tool_error"]
        if not isinstance(fields, Mapping) or set(fields) != {"group_id", "store_key", "run_id", "replay_key"}:
            raise ValueError("native tool error locator is invalid")
        return cls(fields["group_id"], decode_native_store_key(fields["store_key"]),
                   fields["run_id"], fields["replay_key"])


def restore_native_tool_error_payload(
    reference: NativeToolErrorReference, event: StepEvent, *, operation_id: str,
    tool_name: str, tool_call_id: str, fence: int, error_code: str,
) -> StoredPayload:
    """Reconstruct an original failure after full event-source integrity validation."""
    try:
        kinds = {ErrorCode.TOOL_RETRY_REQUIRED.value: "tool_call_rejected",
                 ErrorCode.TOOL_EXECUTION_FAILED.value: "tool_call_failed"}
        if (error_code not in kinds or type(fence) is not int or fence < 1
                or event.kind != "tool_call_failed" or not isinstance(event.error, str)
                or event.run_id != reference.run_id or event.idempotency_key != reference.replay_key
                or event.tool_name != tool_name or event.tool_call_id != tool_call_id
                or reference.replay_key != deterministic_id(
                    "tool", operation_id, event.step_index, fence, "tool_call_failed")):
            raise ValueError("native tool error source does not match its owner")
        return StoredPayload.inline_bytes(canonical_json_bytes(
            {"version": 1, "kind": kinds[error_code], "message": event.error}))
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def prepare_native_tool_error(
    reference: NativeToolErrorReference, event: StepEvent, payload: StoredPayload, *,
    operation_id: str, tool_name: str, tool_call_id: str, fence: int, error_code: str,
) -> NativeToolErrorReference | None:
    """Select only a byte-exact receipt without rewriting valid alternative encodings."""
    if payload.kind != "inline" or payload.encoding != "base64":
        return None
    try:
        restored = restore_native_tool_error_payload(reference, event, operation_id=operation_id,
            tool_name=tool_name, tool_call_id=tool_call_id, fence=fence, error_code=error_code)
    except AIError:
        return None
    return reference if restored.decode() == payload.decode() else None


__all__ = ["NativeToolErrorReference", "prepare_native_tool_error", "restore_native_tool_error_payload"]
