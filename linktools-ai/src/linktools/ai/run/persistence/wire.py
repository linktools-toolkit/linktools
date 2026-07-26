#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run commit wire protocol: a typed, reversible, deterministic encoding of
the seven Run commit operations.

The wire format is canonical JSON carrying type-tagged envelopes so a payload
round-trips back into the EXACT domain dataclass it came from -- not into a
bare dict the caller must reconstruct by hand. Encoding rules:

- dataclass -> ``{"__dc__": "<module>.<qualname>", "f": {field: enc(v)}}``
- Enum      -> ``{"__enum__": "<module>.<qualname>", "v": <value>}``
- bytes     -> ``{"__b64__": "<base64>"}``
- datetime  -> ``{"__dt__": "<UTC ISO8601>"}`` (decoded value keeps its tz)
- date      -> ``{"__date__": "<ISO8601>"}``
- tuple     -> ``{"__tuple__": [enc(x)...]}`` (tuple-ness is preserved)
- set /     -> ``{"__set__": [enc(x)... sorted by canonical json]}``
- frozenset    (deterministic across PYTHONHASHSEED)
- mapping   -> ``{"__map__": [[str(k), enc(v)]... sorted by key]}``
- list      -> ``[enc(x)...]``
- primitive -> as-is (str/int/float/bool/None)

There is deliberately NO ``vars()`` / ``__dict__`` / bare-``Any`` fallback:
every value is either a known primitive or an explicitly tagged envelope, so
adding a field to a Run command/result dataclass is detected (the field is
encoded/decoded with the dataclass, never silently absorbed). The whole
envelope is emitted with ``sort_keys=True`` and ``separators=(",", ":")`` so
identical logical values produce byte-identical wire bytes regardless of
dict insertion order or hash seed."""

from __future__ import annotations

import base64
import importlib
import json
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping


class RunCommitOperation(str, Enum):
    """The seven Run commit operations. Wire payloads are tagged with this
    enum's value so a stored request/result is self-describing and an unknown
    operation fails closed at decode time rather than being misinterpreted."""

    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    COMPLETE = "complete"
    FAIL = "fail"
    REQUEST_CANCEL = "request_cancel"
    ACKNOWLEDGE_CANCEL = "acknowledge_cancel"


class RunCommitCodecError(Exception):
    """The wire payload cannot be encoded or decoded: unknown schema version,
    unknown operation, a corrupted envelope, or a value with no typed
    representation. Raised instead of returning a half-decoded dict."""


class RunCommitIntegrityError(Exception):
    """A persisted completion payload is missing or unreadable where the
    replay contract requires one. Raised rather than falling back to the
    current command, so a replay always returns the FIRST persisted result."""


SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    """Deterministic JSON: sorted keys, minimal separators, UTF-8. The single
    source of byte-stability for both the stored payload and the request
    hash, so SQL and Filesystem produce identical bytes for one logical
    value."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# --- the typed reversible engine --------------------------------------------

def _qualname(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _resolve(qualified: str) -> Any:
    module_name, _, attr = qualified.rpartition(".")
    if not module_name:
        raise RunCommitCodecError(f"cannot resolve type {qualified!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise RunCommitCodecError(
            f"cannot resolve type {qualified!r}: {exc}"
        ) from exc


def _encode(value: Any) -> Any:
    """Recursively encode ``value`` into a JSON-native structure with type
    tags. Raises RunCommitCodecError for anything that has no typed wire
    representation -- there is no reflective fallback."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bool):
        # bool is an int subclass; the branch above already captured it. This
        # guard is defensive in case the ordering above ever changes.
        return value
    if isinstance(value, bytes):
        return {"__b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.astimezone(timezone.utc)
        return {"__dt__": dt.astimezone(timezone.utc).isoformat()}
    if isinstance(value, date):
        return {"__date__": value.isoformat()}
    if isinstance(value, Enum):
        return {"__enum__": _qualname(type(value)), "v": _encode(value.value)}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, (frozenset, set)):
        encoded = [_encode(item) for item in value]
        # Sort by canonical-json bytes so set encoding is independent of
        # iteration order / PYTHONHASHSEED.
        encoded.sort(key=lambda item: _canonical_json(item))
        return {"__set__": encoded}
    if isinstance(value, Mapping):
        items = [[str(key), _encode(val)] for key, val in value.items()]
        items.sort(key=lambda pair: pair[0])
        return {"__map__": items}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        payload = {
            field.name: _encode(getattr(value, field.name))
            for field in fields(value)
        }
        return {"__dc__": _qualname(type(value)), "f": payload}
    raise RunCommitCodecError(
        f"unsupported wire value of type {type(value).__name__!r}"
    )


def _decode(value: Any) -> Any:
    """Inverse of :func:`_encode`. Reconstructs the original typed value,
    including dataclass instances and Enum members."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, Mapping):
        raise RunCommitCodecError(
            f"unsupported wire node of type {type(value).__name__!r}"
        )
    # Mapping markers -- check by single distinguishing key.
    if "__b64__" in value:
        try:
            return base64.b64decode(value["__b64__"], validate=True)
        except (ValueError, TypeError) as exc:
            raise RunCommitCodecError(f"invalid base64 payload: {exc}") from exc
    if "__dt__" in value:
        try:
            dt = datetime.fromisoformat(value["__dt__"])
        except (ValueError, TypeError) as exc:
            raise RunCommitCodecError(f"invalid datetime payload: {exc}") from exc
        if dt.tzinfo is None:
            raise RunCommitCodecError(
                "decoded datetime has no timezone; wire datetimes must be UTC"
            )
        return dt
    if "__date__" in value:
        try:
            return date.fromisoformat(value["__date__"])
        except (ValueError, TypeError) as exc:
            raise RunCommitCodecError(f"invalid date payload: {exc}") from exc
    if "__enum__" in value:
        enum_cls = _resolve(value["__enum__"])
        if not (isinstance(enum_cls, type) and issubclass(enum_cls, Enum)):
            raise RunCommitCodecError(
                f"wire enum tag resolves to non-enum: {value['__enum__']!r}"
            )
        return enum_cls(_decode(value["v"]))
    if "__tuple__" in value:
        return tuple(_decode(item) for item in value["__tuple__"])
    if "__set__" in value:
        return frozenset(_decode(item) for item in value["__set__"])
    if "__map__" in value:
        return {pair[0]: _decode(pair[1]) for pair in value["__map__"]}
    if "__dc__" in value:
        cls = _resolve(value["__dc__"])
        if not (isinstance(cls, type) and is_dataclass(cls)):
            raise RunCommitCodecError(
                f"wire dataclass tag resolves to non-dataclass: "
                f"{value['__dc__']!r}"
            )
        field_payload = value.get("f", {})
        if not isinstance(field_payload, Mapping):
            raise RunCommitCodecError("dataclass wire node has non-object fields")
        decoded_fields = {
            name: _decode(val) for name, val in field_payload.items()
        }
        try:
            return cls(**decoded_fields)
        except TypeError as exc:
            raise RunCommitCodecError(
                f"cannot reconstruct {value['__dc__']!r}: {exc}"
            ) from exc
    # An untagged mapping is treated as a plain dict (preserves mapping
    # values carried inside Any-typed fields like RunResult.metadata).
    return {str(key): _decode(val) for key, val in value.items()}


# --- public encode/decode of the wire envelope ------------------------------

def encode_envelope(operation: "RunCommitOperation | str", payload: Any) -> bytes:
    """Encode a typed ``payload`` into the canonical wire envelope bytes."""
    op_value = operation.value if isinstance(operation, RunCommitOperation) else operation
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "operation": op_value,
        "payload": _encode(payload),
    }
    return _canonical_json(envelope)


def decode_envelope(
    payload: bytes,
    *,
    expected_operation: "RunCommitOperation | str | None" = None,
) -> Any:
    """Decode wire ``payload`` bytes back into the typed value. When
    ``expected_operation`` is given, the envelope's operation must match or
    the decode fails closed."""
    try:
        envelope = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise RunCommitCodecError(f"wire payload is not valid JSON: {exc}") from exc
    if not isinstance(envelope, Mapping):
        raise RunCommitCodecError("wire envelope is not a JSON object")
    if envelope.get("schema_version") != SCHEMA_VERSION:
        raise RunCommitCodecError(
            f"unsupported wire schema_version: {envelope.get('schema_version')!r}"
        )
    op = envelope.get("operation")
    if expected_operation is not None:
        want = (
            expected_operation.value
            if isinstance(expected_operation, RunCommitOperation)
            else expected_operation
        )
        if op != want:
            raise RunCommitCodecError(
                f"wire operation {op!r} does not match expected {want!r}"
            )
    try:
        operation = RunCommitOperation(op)
    except ValueError as exc:
        raise RunCommitCodecError(f"unknown wire operation {op!r}") from exc
    # operation is reconstructed (and validated) so callers can trust it; the
    # decoded typed payload is the useful return value.
    _ = operation
    if "payload" not in envelope:
        raise RunCommitCodecError("wire envelope has no payload")
    return _decode(envelope["payload"])


__all__ = [
    "RunCommitOperation",
    "RunCommitCodecError",
    "RunCommitIntegrityError",
    "SCHEMA_VERSION",
    "encode_envelope",
    "decode_envelope",
]
