#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict canonical JSON for every hash/fingerprint/persistence path.

The ``default`` stringification argument to ``json.dumps`` is forbidden here:
it silently coerces arbitrary objects (datetime, Path, UUID, Decimal, custom
classes) whose repr is unstable across versions and can collide (two types
with the same string). ``normalize_json`` rejects anything that is not
genuinely JSON-compatible, and ``canonical_json_bytes`` emits stable, sorted,
compact bytes two processes can agree on. Use it for idempotency request
hashes, exact-call keys, spec fingerprints, MCP connection fingerprints, and
any persisted JSON."""

import dataclasses
import json
import math
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias
from uuid import UUID

from .errors import JsonEncodingError

JsonScalar: "TypeAlias" = str | int | float | bool | None
JsonValue: "TypeAlias" = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def freeze_value(value: Any) -> Any:
    """Recursively freeze container values used by immutable public models."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_value(item) for item in value)
    return value


def normalize_json(value: Any, *, path: str = "$") -> JsonValue:
    """Return a JSON-compatible view of ``value`` or raise ``JsonEncodingError``.

    Accepted: ``None``/``str``/``bool``/``int``; finite ``float``; ``UUID``;
    timezone-aware ``datetime`` (rendered as UTC ISO 8601 -- naive datetimes are
    rejected, their zone is ambiguous); ``Enum`` (by its value); ``list``/``tuple``;
    ``Mapping`` (string keys only); dataclasses; pydantic models (via
    ``model_dump``). Everything else (``bytes``, ``set``, ``Decimal``, custom
    classes) is rejected -- callers must convert those explicitly rather than
    rely on an unstable string fallback."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JsonEncodingError(f"{path}: non-finite float is not valid JSON")
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise JsonEncodingError(f"{path}: naive datetime is not valid JSON (use tz-aware)")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return normalize_json(value.value, path=path)
    if isinstance(value, (list, tuple)):
        return [normalize_json(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out: "dict[str, Any]" = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise JsonEncodingError(f"{path}: non-string mapping key {key!r}")
            out[key] = normalize_json(item, path=f"{path}.{key}")
        return out
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return normalize_json(model_dump(mode="python"), path=path)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: normalize_json(
                getattr(value, field.name),
                path=f"{path}.{field.name}",
            )
            for field in dataclasses.fields(value)
        }
    raise JsonEncodingError(
        f"{path}: {type(value).__name__} is not JSON-compatible "
        f"(convert it explicitly instead of relying on a string fallback)"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON encoded as UTF-8 bytes for hashing and fingerprints."""
    return json.dumps(
        normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
