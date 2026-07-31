#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical identity for tool calls: binding fingerprints and idempotency keys."""


from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from ...json import canonical_json_bytes, normalize_json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...json import JsonValue

@dataclass(frozen=True, slots=True)
class ToolRevisionSet:
    descriptor: str
    handler: str
    provider: str
    policy: str
    feature: str
    result_processor: str


@dataclass(frozen=True, slots=True)
class ToolExecutionBinding:
    schema_version: int
    tool_name: str
    arguments_hash: str
    revisions: ToolRevisionSet

    def fingerprint(self) -> str:
        return sha256(
            canonical_json_bytes(normalize_json(asdict(self)))
        ).hexdigest()


def hash_tool_arguments(
    tool_name: str,
    arguments: "Mapping[str, JsonValue]",
) -> str:
    payload: "JsonValue" = {
        "tool_name": tool_name,
        "arguments": normalize_json(dict(arguments)),
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def operation_id(execution_id: str, tool_call_id: str) -> str:
    return sha256(
        canonical_json_bytes(
            {"execution_id": execution_id, "tool_call_id": tool_call_id}
        )
    ).hexdigest()
