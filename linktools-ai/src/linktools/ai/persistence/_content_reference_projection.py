#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-row SQL lookup projections of canonical content reference slots."""

from collections.abc import Mapping
from typing import NamedTuple
import re


class ContentReferenceSlot(NamedTuple):
    column: str
    codec: str | None


_PREFIX = ("value", "payload", "fields")
_OUTPUT = ("result", "fields", "output", "fields", "content")
_HANDOFF = ("recovery", "fields", "terminal_handoff", "fields")
_TOOL = ("fields", "content")
_PAYLOAD = "$execution_payload"
_EXECUTION_SLOTS = {
    ("runtime_policy", "$runtime_policy"): ContentReferenceSlot("execution_policy_sha256", "runtime-policy"),
    ("binding", "$execution_binding"): ContentReferenceSlot("execution_binding_sha256", "execution-binding"),
    ("stored_user_input", "fields", "content", _PAYLOAD): ContentReferenceSlot("execution_input_sha256", "execution-payload"),
    ("result", "fields", "content", _PAYLOAD): ContentReferenceSlot("execution_result_sha256", "execution-payload"),
    (*_OUTPUT, _PAYLOAD): ContentReferenceSlot("execution_output_sha256", "execution-payload"),
    ("terminal_details", "fields", "content", _PAYLOAD): ContentReferenceSlot("execution_terminal_details_sha256", "execution-payload"),
    ("recovery", "fields", "pending_tools", "fields", "content", _PAYLOAD):
        ContentReferenceSlot("execution_pending_tools_sha256", "execution-payload"),
    (*_HANDOFF, "content", _PAYLOAD): ContentReferenceSlot("execution_handoff_sha256", "execution-payload"),
    (*_HANDOFF, "terminal_details", "fields", "content", _PAYLOAD):
        ContentReferenceSlot("execution_handoff_details_sha256", "execution-payload"),
    ("stored_user_input", "fields", "content", "$native_user_input"): ContentReferenceSlot("execution_input_first_sha256", None),
    (*_OUTPUT, "$native_text"): ContentReferenceSlot("execution_output_first_sha256", None),
    (*_HANDOFF, "output", "fields", "content", "$native_text"): ContentReferenceSlot("execution_handoff_output_first_sha256", None),
}
_TOOL_SLOTS = {
    ("arguments_payload", *_TOOL, _PAYLOAD): ContentReferenceSlot("tool_arguments_sha256", "execution-payload"),
    ("result_payload", *_TOOL, _PAYLOAD): ContentReferenceSlot("tool_result_sha256", "execution-payload"),
    ("error_payload", *_TOOL, _PAYLOAD): ContentReferenceSlot("tool_error_sha256", "execution-payload"),
    ("arguments_payload", *_TOOL, "$native_tool_arguments"): ContentReferenceSlot("tool_arguments_first_sha256", None),
    ("result_payload", *_TOOL, "$native_tool_result"): ContentReferenceSlot("tool_result_first_sha256", None),
}
_CONTENT_SLOTS = {(*_PREFIX, *path): slot for path, slot in (_EXECUTION_SLOTS | _TOOL_SLOTS).items()}
CONTENT_REFERENCE_COLUMNS = {
    "ai_state_records": tuple(slot.column for slot in _CONTENT_SLOTS.values()),
    "ai_step_tool_operations": tuple(slot.column for slot in _TOOL_SLOTS.values()),
}
CONTENT_REFERENCE_COLUMN_NAMES = frozenset(CONTENT_REFERENCE_COLUMNS["ai_state_records"])
_SHA256 = re.compile(r"[a-f0-9]{64}")


def content_reference_slot(path: tuple[str, ...]) -> ContentReferenceSlot | None:
    return _CONTENT_SLOTS.get(path)


def _project_digest(value: object, codec: str | None) -> str | None:
    if codec is not None:
        if not isinstance(value, str) or not value.startswith(codec + ":"):
            return None
        digest = value[len(codec) + 1:]
    else:
        slices = value.get("slices") if isinstance(value, Mapping) else None
        if not isinstance(slices, list) or len(slices) != 1 or not isinstance(slices[0], Mapping):
            return None
        item = slices[0]
        count, start, end = item.get("count"), item.get("start"), item.get("end")
        # This is only a candidate; reading also validates the complete root and scope.
        if (type(count) is not int or type(start) is not int or type(end) is not int
                or not 1 <= count <= (1 << 63) - 1 or start != count - 1 or end != count):
            return None
        digest = item.get("first")
    return digest if isinstance(digest, str) and _SHA256.fullmatch(digest) else None


def content_reference_values(kind: str, data: Mapping[str, object]) -> dict[str, str | None]:
    slots = {"execution": _EXECUTION_SLOTS, "tool_operation": _TOOL_SLOTS}.get(kind.rstrip(" "))
    if slots is None:
        return {}
    fields: object = data
    for component in _PREFIX:
        fields = fields.get(component) if isinstance(fields, Mapping) else None
    result = {}
    for path, slot in slots.items():
        value = fields
        for component in path:
            value = value.get(component) if isinstance(value, Mapping) else None
        result[slot.column] = _project_digest(value, slot.codec)
    return result


__all__ = ["CONTENT_REFERENCE_COLUMNS", "CONTENT_REFERENCE_COLUMN_NAMES", "ContentReferenceSlot",
           "content_reference_slot", "content_reference_values"]
