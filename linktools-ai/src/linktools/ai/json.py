#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict canonical JSON for every hash/fingerprint/persistence path.

The ``default`` stringification argument to ``json.dumps`` is forbidden here:
it silently coerces arbitrary objects (datetime, Path, UUID, Decimal, custom
classes) whose repr is unstable across versions and can collide (two types
with the same string). ``normalize_json`` rejects anything that is not
genuinely JSON-compatible, and ``canonical_json`` emits a stable, sorted,
compact string two processes can agree on. Use it for idempotency request
hashes, exact-call keys, spec fingerprints, MCP connection fingerprints, and
any persisted JSON."""

import dataclasses
import json
import math
import types
import typing
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias
from uuid import UUID

from .errors import JsonEncodingError

JsonScalar: "TypeAlias" = str | int | float | bool | None
JsonValue: "TypeAlias" = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
# Backwards-compatible spelling retained for existing string annotations.
JSONValue: "TypeAlias" = JsonValue


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


def normalize_json(value: Any, *, path: str = "$") -> "JSONValue":
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
        return normalize_json(dataclasses.asdict(value), path=path)
    raise JsonEncodingError(
        f"{path}: {type(value).__name__} is not JSON-compatible "
        f"(convert it explicitly instead of relying on a string fallback)"
    )


def canonical_json(value: Any) -> str:
    """Stable, sorted, compact JSON encoding of ``value`` (``normalize_json``
    then ``json.dumps`` with ``sort_keys``, compact separators, ``ensure_ascii``
    off, ``allow_nan=False``). Deterministic across processes for equal inputs."""
    return json.dumps(
        normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def encode_json(value: Any) -> str:
    """Compact JSON encoding of ``value`` (not sorted -- preserves insertion
    order). Use ``canonical_json`` when the bytes must be stable for a hash."""
    return json.dumps(normalize_json(value), ensure_ascii=False, separators=(",", ":"))


def decode_json(raw: str) -> "JsonValue":
    return normalize_json(json.loads(raw))  # type: ignore[return-value]


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON encoded as UTF-8 bytes (for hashing/fingerprinting)."""
    return canonical_json(value).encode("utf-8")


# ----------------------------------------------------------- generic serde --
# to_jsonable / from_jsonable: a symmetric pair for persisting frozen
# dataclasses + str-enums + tz-aware datetimes + homogeneous tuples + Mappings.
# Lives in this neutral module so every store (Run / Task / Evaluation, file and
# SQL) encodes identically without any domain reaching across another for the
# helper.

_NONE = type(None)
_PRIMITIVES = (str, int, float, bool)


def to_jsonable(obj: object) -> object:
    if obj is None or isinstance(obj, _PRIMITIVES):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (tuple, list)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    raise TypeError(f"cannot serialize {type(obj)!r}")


def from_jsonable(cls: object, data: object) -> object:
    if data is None:
        return None
    origin = typing.get_origin(cls)
    if origin in (typing.Union, types.UnionType):
        # All optionals here are ``X | None``; the non-None arm carries type.
        candidates = [arg for arg in typing.get_args(cls) if arg is not _NONE]
        if len(candidates) == 1:
            return from_jsonable(candidates[0], data)
        raise TypeError(f"cannot reconstruct non-optional union {cls!r}")
    if cls is _NONE:
        return None
    if cls in _PRIMITIVES:
        return cls(data)  # type: ignore[call-arg]
    if cls is datetime:
        return datetime.fromisoformat(data)
    if isinstance(cls, type) and issubclass(cls, Enum):
        return cls(data)  # type: ignore[call-arg]
    if origin in (tuple, list) or cls in (tuple, list):
        item_type = typing.get_args(cls)[0] if typing.get_args(cls) else object
        seq = [from_jsonable(item_type, item) for item in data]
        return tuple(seq) if (cls is tuple or origin is tuple) else seq
    if origin is dict or cls is dict:
        value_type = (
            typing.get_args(cls)[1] if len(typing.get_args(cls)) >= 2 else object
        )
        return {key: from_jsonable(value_type, value) for key, value in data.items()}
    if isinstance(cls, type) and issubclass(cls, Mapping):
        return dict(data)
    if dataclasses.is_dataclass(cls) and isinstance(cls, type):
        return _reconstruct_dataclass(cls, data)
    return data


def _reconstruct_dataclass(cls: type, data: object) -> object:
    hints = typing.get_type_hints(cls)
    kwargs: dict = {}
    for field_obj in dataclasses.fields(cls):
        if field_obj.name not in data:
            if (
                field_obj.default is not dataclasses.MISSING
                or field_obj.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            ):
                continue
            raise ValueError(f"missing field {field_obj.name!r} reconstructing {cls!r}")
        kwargs[field_obj.name] = from_jsonable(
            hints[field_obj.name], data[field_obj.name]
        )
    return cls(**kwargs)
