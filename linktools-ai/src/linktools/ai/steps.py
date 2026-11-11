#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native step records with explicitly owned Runtime lifecycle semantics."""

from collections.abc import Mapping
import base64
import binascii
import hashlib
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai_harness.step_persistence import StepEvent

from .core import validate_resource_id, validate_persistence_namespace, validate_tenant_id
from .errors import AIError


NATIVE_STORE_KEY_MAX_BYTES = 274
NATIVE_REPLAY_KEY_MAX_BYTES = 128


def native_store_key(namespace: str, tenant_id: str) -> bytes:
    """Exact native partition tuple, independent of State store partitions."""
    values = ("native_store", validate_persistence_namespace(namespace), validate_tenant_id(tenant_id))
    return b"".join(len(raw).to_bytes(2, "big") + raw for raw in (value.encode("utf-8") for value in values))


def validate_native_store_key(value: bytes) -> bytes:
    """Check the complete current native scope at authoring and durable boundaries."""
    if type(value) is not bytes or not 1 <= len(value) <= NATIVE_STORE_KEY_MAX_BYTES:
        raise ValueError("native store key is invalid")
    parts = []
    offset = 0
    try:
        while offset < len(value):
            if offset + 2 > len(value):
                raise ValueError("native store key is invalid")
            size = int.from_bytes(value[offset:offset + 2], "big")
            offset += 2
            if offset + size > len(value):
                raise ValueError("native store key is invalid")
            parts.append(value[offset:offset + size].decode("utf-8"))
            offset += size
        if len(parts) != 3 or parts[0] != "native_store" or native_store_key(parts[1], parts[2]) != value:
            raise ValueError("native store key is invalid")
    except (AIError, UnicodeError) as error:
        raise ValueError("native store key is invalid") from error
    return value


def native_store_locator(value: bytes) -> str:
    """Derive the indexed projection without changing the exact partition identity."""
    return hashlib.sha256(validate_native_store_key(value)).hexdigest()


def encode_native_store_key(value: bytes) -> str:
    """Encode an exact native scope only at a JSON transport boundary."""
    return base64.b64encode(validate_native_store_key(value)).decode("ascii")


def decode_native_store_key(value: object) -> bytes:
    """Reject malformed and noncanonical native scope transport tokens."""
    if not isinstance(value, str) or len(value) > 4 * ((NATIVE_STORE_KEY_MAX_BYTES + 2) // 3):
        raise ValueError("native store key transport is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
        if encode_native_store_key(raw) != value:
            raise ValueError("native store key transport is invalid")
    except (binascii.Error, UnicodeError) as error:
        raise ValueError("native store key transport is invalid") from error
    return raw


def validate_native_replay_key(value: str | None) -> str | None:
    """Preserve unkeyed occurrences and exact keyed strings, including empty keys."""
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("native replay key must be a string without NUL or None")
    try:
        if len(value.encode("utf-8")) > NATIVE_REPLAY_KEY_MAX_BYTES:
            raise ValueError("native replay key exceeds 128 UTF-8 bytes")
    except UnicodeEncodeError as error:
        raise ValueError("native replay key is invalid") from error
    return value


def encode_native_replay_key(value: str | None) -> bytes:
    """Prepare exact UTF-8 replay bytes or a distinct private occurrence identity."""
    return b"\x00" + uuid4().bytes if validate_native_replay_key(value) is None else value.encode("utf-8")


def decode_native_replay_key(value: bytes) -> str | None:
    """Decode stored replay bytes, keeping private occurrence identities private."""
    if type(value) is not bytes:
        raise ValueError("native replay row must contain bytes")
    if value.startswith(b"\x00"):
        if len(value) != 17:
            raise ValueError("native occurrence key is invalid")
        return None
    try:
        result = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("native replay row is invalid") from error
    return validate_native_replay_key(result)


def validate_native_step_ids(
    *, run_id: str, conversation_id: str | None = None, parent_run_id: str | None = None,
) -> None:
    """Apply the resource ID contract to native run identity and lineage."""
    validate_resource_id(run_id)
    for value in (conversation_id, parent_run_id):
        if value is not None:
            validate_resource_id(value)


@dataclass(frozen=True, slots=True)
class NativeStepSource:
    """Physical native checkpoint owner admitted with an execution."""

    group_id: str
    store_key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.group_id, str) or len(self.group_id) != 32
                or any(character not in "0123456789abcdef" for character in self.group_id)):
            raise ValueError("native checkpoint source group is invalid")
        validate_native_store_key(self.store_key)


def rekey_native_step_sources(value: object, groups: Mapping[str, str]) -> object:
    """Relocate only the explicit durable native source wire contract."""
    if isinstance(value, dict):
        result = {key: rekey_native_step_sources(item, groups) for key, item in value.items()}
        if value.get("$dataclass") == "native_step_source":
            fields = result.get("fields")
            if not isinstance(fields, dict):
                raise ValueError("native checkpoint source fields are invalid")
            raw_key = fields.get("store_key")
            if not isinstance(raw_key, dict) or set(raw_key) != {"$bytes"}:
                raise ValueError("native checkpoint source key is invalid")
            source = NativeStepSource(fields.get("group_id"), decode_native_store_key(raw_key["$bytes"]))
            if source.group_id not in groups:
                raise ValueError("native checkpoint source group is missing")
            fields["group_id"] = groups[source.group_id]
        return result
    if isinstance(value, list):
        return [rekey_native_step_sources(item, groups) for item in value]
    if isinstance(value, tuple):
        return tuple(rekey_native_step_sources(item, groups) for item in value)
    return value


LifecycleOutcome = Literal["succeeded", "interrupted", "failed", "cancelled"]


@dataclass(kw_only=True)
class NativeLifecycleEvent(StepEvent):
    """One native terminal body, without modifying shared run metadata."""

    outcome: LifecycleOutcome

    def __post_init__(self) -> None:
        validate_native_step_ids(run_id=self.run_id, conversation_id=self.conversation_id,
                                 parent_run_id=self.parent_run_id)
        validate_native_replay_key(self.idempotency_key)
        kind = {"succeeded": "run_completed", "interrupted": "run_completed",
                "failed": "run_failed", "cancelled": "run_failed"}.get(self.outcome)
        if (kind is None or self.kind != kind or self.idempotency_key is None
                or self.tool_call_id is not None or self.tool_name is not None
                or (kind == "run_completed" and self.error is not None)):
            raise ValueError("native lifecycle outcome does not match its event")

    @classmethod
    def source_key(cls) -> str:
        """The first shared terminal observation of this single-shot native run."""
        return "terminal"

    @property
    def event_type(self) -> Literal["AGENT_RUN_SUCCEEDED", "AGENT_RUN_INTERRUPTED", "AGENT_RUN_FAILED"]:
        if self.outcome == "succeeded":
            return "AGENT_RUN_SUCCEEDED"
        if self.outcome == "interrupted":
            return "AGENT_RUN_INTERRUPTED"
        return "AGENT_RUN_FAILED"


__all__ = [
    "NATIVE_STORE_KEY_MAX_BYTES", "NATIVE_REPLAY_KEY_MAX_BYTES", "native_store_key",
    "validate_native_store_key", "native_store_locator", "encode_native_store_key", "decode_native_store_key",
    "validate_native_replay_key", "encode_native_replay_key", "decode_native_replay_key",
    "validate_native_step_ids", "LifecycleOutcome", "NativeLifecycleEvent", "NativeStepSource",
    "rekey_native_step_sources",
]
