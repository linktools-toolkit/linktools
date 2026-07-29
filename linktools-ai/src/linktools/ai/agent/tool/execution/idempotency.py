"""Canonical identity helpers for tool calls."""

from collections.abc import Mapping
from hashlib import sha256

from ....json import JsonValue, canonical_json_bytes, normalize_json


def hash_tool_arguments(
    tool_name: str,
    arguments: Mapping[str, JsonValue],
) -> str:
    payload: JsonValue = {
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
