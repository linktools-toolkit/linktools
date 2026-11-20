#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-row SQL lookup projections of canonical content reference slots."""

import re
from collections.abc import Mapping
from typing import NamedTuple

from ..errors import AIError, ErrorCode


class ContentReferenceSlot(NamedTuple):
    column: str
    codec: str | None


_PREFIX = ("value", "payload", "fields")
_OUTPUT = ("result", "fields", "output", "fields", "content")
_TOOL = ("fields", "content")
_PAYLOAD = "$execution_payload"
_EXECUTION_SLOTS = {
    ("runtime_policy", "$runtime_policy"): ContentReferenceSlot("policy_sha256", "runtime-policy"),
    ("binding", "$execution_binding"): ContentReferenceSlot("binding_sha256", "execution-binding"),
    ("stored_user_input", "fields", "content", _PAYLOAD): ContentReferenceSlot("input_sha256", "execution-payload"),
    ("result", "fields", "content", _PAYLOAD): ContentReferenceSlot("result_sha256", "execution-payload"),
    (*_OUTPUT, _PAYLOAD): ContentReferenceSlot("output_sha256", "execution-payload"),
    ("stored_user_input", "fields", "content", "$native_user_input"): ContentReferenceSlot("input_first_sha256", None),
    (*_OUTPUT, "$native_text"): ContentReferenceSlot("output_first_sha256", None),
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
    "ai_execution_records": tuple(slot.column for slot in _EXECUTION_SLOTS.values()),
    "ai_step_tool_operations": tuple(slot.column for slot in _TOOL_SLOTS.values()),
}
EXECUTION_REFERENCE_COLUMNS = CONTENT_REFERENCE_COLUMNS["ai_execution_records"]
CONTENT_REFERENCE_COLUMN_NAMES = frozenset(slot.column for slot in _CONTENT_SLOTS.values())
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


def tool_content_reference_values(kind: str, data: Mapping[str, object]) -> dict[str, str | None]:
    if kind != "tool_operation":
        return {}
    fields: object = data
    for component in _PREFIX:
        fields = fields.get(component) if isinstance(fields, Mapping) else None
    result = {}
    for path, slot in _TOOL_SLOTS.items():
        value = fields
        for component in path:
            value = value.get(component) if isinstance(value, Mapping) else None
        result[slot.column] = _project_digest(value, slot.codec)
    return result


_ROW_REFERENCE = True


def _execution_field_map(value: Mapping) -> Mapping | None:
    envelope = value.get("value")
    if not isinstance(envelope, Mapping) or envelope.get("type") != "execution_record":
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping) or payload.get("$dataclass") != "execution_record":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return fields


def _slot_value(value: object, path: tuple[str | int, ...]) -> object:
    for part in path:
        if isinstance(part, int):
            value = value[part] if isinstance(value, list) and len(value) > part else None
        else:
            value = value.get(part) if isinstance(value, Mapping) else None
    return value


def _replace_slot(value: object, path: tuple[str | int, ...], replacement: object) -> object:
    if not path:
        return replacement
    part, *tail = path
    if isinstance(part, int):
        updated = list(value)
    else:
        updated = dict(value)
    updated[part] = _replace_slot(updated[part], tuple(tail), replacement)
    return updated


def pack_execution_references(value: Mapping) -> tuple[Mapping, dict[str, str | None]]:
    """Move a digest, never the complete archive/native descriptor, to its SQL cell."""
    fields = _execution_field_map(value)
    if fields is None:
        return value, {}
    updated = value
    columns = {}
    for path, slot in _EXECUTION_SLOTS.items():
        source = _slot_value(fields, path)
        digest = _project_digest(source, slot.codec)
        columns[slot.column] = digest
        cell_path = path if slot.codec is not None else (*path, "slices", 0, "first")
        if _slot_value(fields, cell_path) is _ROW_REFERENCE:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if digest is not None:
            updated = _replace_slot(updated, (*_PREFIX, *cell_path), _ROW_REFERENCE)
    return updated, columns


def unpack_execution_references(value: Mapping, row: Mapping) -> Mapping:
    """Restore the exact logical v1 envelope before any semantic or archive read."""
    fields = _execution_field_map(value)
    updated = value
    for path, slot in _EXECUTION_SLOTS.items():
        digest = row.get(slot.column)
        if fields is None:
            if digest is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            continue
        cell_path = path if slot.codec is not None else (*path, "slices", 0, "first")
        source = _slot_value(fields, cell_path)
        if digest is None:
            if source is _ROW_REFERENCE or _project_digest(_slot_value(fields, path), slot.codec) is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            continue
        if source is not _ROW_REFERENCE or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        restored = digest if slot.codec is None else slot.codec + ":" + digest
        updated = _replace_slot(updated, (*_PREFIX, *cell_path), restored)
        if _project_digest(_slot_value(updated, (*_PREFIX, *path)), slot.codec) != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return updated


__all__ = ["CONTENT_REFERENCE_COLUMNS", "CONTENT_REFERENCE_COLUMN_NAMES", "ContentReferenceSlot", "EXECUTION_REFERENCE_COLUMNS",
           "content_reference_slot", "tool_content_reference_values", "pack_execution_references", "unpack_execution_references"]
