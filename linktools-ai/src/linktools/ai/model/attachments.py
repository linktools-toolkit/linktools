#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate binary bodies only at the durable model codec's typed positions."""

import base64
import binascii
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterator
from typing import cast

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, StoredPayload

_Path = tuple[str | int, ...]


def _user_slot(value: object, path: _Path, native: bool) -> Iterator[tuple[_Path, bool]]:
    if isinstance(value, dict) and value.get("kind") == "binary":
        yield (*path, "data"), native


def _tool_slots(value: object, path: _Path) -> Iterator[tuple[_Path, bool]]:
    if not isinstance(value, dict):
        return
    kind = value.get("type")
    if kind == "multimodal":
        yield from _user_slot(value.get("value"), (*path, "value"), False)
    elif kind == "tool-return":
        yield from _tool_slots(value.get("return_value"), (*path, "return_value"))
        content = value.get("content")
        if isinstance(content, list):
            for i, item in enumerate(content):
                yield from _user_slot(item, (*path, "content", i), False)
    elif kind in {"sequence", "mapping"}:
        items = value.get("items")
        if isinstance(items, list):
            for i, item in enumerate(items):
                yield from _tool_slots(item, (*path, "items", i))
        elif isinstance(items, dict):
            for key, item in items.items():
                yield from _tool_slots(item, (*path, "items", key))
    # scalar/json-snapshot/text/structured metadata are opaque business values.


def _slots(value: object, format: str) -> Iterator[tuple[_Path, bool]]:
    if format == "user-content":
        if not isinstance(value, dict) or not isinstance(value.get("items"), list):
            raise ValueError("user content envelope is invalid")
        for i, item in enumerate(value["items"]):
            yield from _user_slot(item, ("items", i), False)
    elif format == "tool-return":
        if not isinstance(value, dict) or value.get("contract") != "linktools.tool-return":
            raise ValueError("tool return envelope is invalid")
        yield from _tool_slots(value.get("value"), ("value",))
    elif format == "model-messages":
        if not isinstance(value, list):
            raise ValueError("model message envelope is invalid")
        for mi, message in enumerate(value):
            if not isinstance(message, dict) or not isinstance(message.get("parts"), list):
                raise ValueError("model message parts are invalid")
            for pi, part in enumerate(message["parts"]):
                if not isinstance(part, dict):
                    raise ValueError("model message part is invalid")
                path = (mi, "parts", pi)
                kind = part.get("part_kind")
                if kind == "user-prompt" and isinstance(part.get("content"), list):
                    for ci, item in enumerate(part["content"]):
                        yield from _user_slot(item, (*path, "content", ci), True)
                elif kind in {"tool-return", "native-tool-return", "builtin-tool-return"}:
                    content = part.get("content")
                    if isinstance(content, dict) and content.get("contract") == "linktools.tool-return":
                        yield from _tool_slots(content.get("value"), (*path, "content", "value"))
                elif kind == "file":
                    yield from _user_slot(part.get("content"), (*path, "content"), True)
                elif kind == "speech":
                    yield from _user_slot(part.get("audio"), (*path, "audio"), True)
    else:
        raise ValueError("attachment payload format is unsupported")


def _parent(value: object, path: _Path) -> object:
    for key in path[:-1]:
        value = value[key]
    return value


def _split(raw: bytes, format: str) -> tuple[JsonValue, list[tuple[_Path, bytes]]]:
    value = json.loads(raw)
    if canonical_json_bytes(value) != raw:
        raise ValueError("attachment payload must use canonical JSON bytes")
    bodies = []
    for path, native in _slots(value, format):
        parent = _parent(value, path)
        encoded = parent[path[-1]]
        if not isinstance(encoded, str):
            raise ValueError("binary content data must be base64")
        data = base64.b64decode(encoded, altchars=b"-_" if native else None, validate=True)
        if (base64.urlsafe_b64encode(data) if native else base64.b64encode(data)).decode("ascii") != encoded:
            raise ValueError("binary content encoding is not canonical")
        bodies.append((path, data))
        parent[path[-1]] = None
    return value, bodies


def attachment_payload_objects(raw: bytes, payload: StoredPayload) -> tuple[tuple[ObjectRef, bytes], ...]:
    """Verify source bytes and expose their file bodies without reading objects again."""
    if payload.encoding != "json-attachments":
        raise ValueError("attachment payload encoding is required")
    manifest = payload.value
    content, bodies = _split(raw, manifest["format"])
    if (len(raw) != payload.size or hashlib.sha256(raw).hexdigest() != payload.digest
            or content != manifest["content"] or len(bodies) != len(manifest["attachments"])):
        raise ValueError("attachment manifest does not match source bytes")
    objects = []
    for (path, data), attachment, ref in zip(bodies, manifest["attachments"], payload.object_refs(), strict=True):
        if (list(path) != attachment["path"] or len(data) != ref.size
                or hashlib.sha256(data).hexdigest() != ref.digest):
            raise ValueError("attachment reference does not match source bytes")
        objects.append((ref, data))
    return tuple(objects)


def validate_attachment_payload(raw: bytes, payload: StoredPayload) -> None:
    """Prove prepared references match original bytes without reading them back."""
    attachment_payload_objects(raw, payload)


async def store_attachment_payload(
    raw: bytes, *, format: str, write_object: Callable[[bytes], Awaitable[ObjectRef]],
) -> StoredPayload | None:
    """Write typed binary bodies and keep the surrounding message in the database."""
    content, bodies = _split(raw, format)
    if not bodies:
        return None
    attachments: list[JsonValue] = []
    for path, data in bodies:
        ref = await write_object(data)
        if ref.size != len(data) or ref.digest != hashlib.sha256(data).hexdigest():
            raise ValueError("attachment writer returned different bytes")
        attachments.append({"path": list(path), "ref": {
            "store_id": ref.store_id, "key": ref.key, "digest": ref.digest, "size": ref.size,
        }})
    return StoredPayload("inline", "json-attachments", hashlib.sha256(raw).hexdigest(), len(raw),
                         {"format": format, "content": content, "attachments": attachments})


async def read_attachment_payload(
    payload: StoredPayload, *, read_object: Callable[[ObjectRef], Awaitable[bytes]],
) -> bytes:
    """Restore the byte-exact source and verify each file and the complete message."""
    try:
        if payload.encoding != "json-attachments":
            raise ValueError("attachment payload encoding is required")
        manifest = payload.value
        content = manifest["content"]
        slots = tuple(_slots(content, manifest["format"]))
        attachments = manifest["attachments"]
        if len(slots) != len(attachments):
            raise ValueError("attachment slots are incomplete")
        for (path, native), attachment, ref in zip(slots, attachments, payload.object_refs(), strict=True):
            parent = _parent(content, path)
            if list(path) != attachment["path"] or parent[path[-1]] is not None:
                raise ValueError("attachment slot is invalid")
            data = await read_object(ref)
            if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.digest:
                raise ValueError("attachment bytes are corrupt")
            parent[path[-1]] = (base64.urlsafe_b64encode(data) if native else base64.b64encode(data)).decode("ascii")
        raw = canonical_json_bytes(cast(JsonValue, content))
        if len(raw) != payload.size or hashlib.sha256(raw).hexdigest() != payload.digest:
            raise ValueError("attachment source bytes are corrupt")
        return raw
    except (KeyError, IndexError, TypeError, ValueError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["store_attachment_payload", "read_attachment_payload", "validate_attachment_payload", "attachment_payload_objects"]
