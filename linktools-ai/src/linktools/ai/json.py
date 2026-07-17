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
from typing import TYPE_CHECKING, Any, Mapping
from uuid import UUID

if TYPE_CHECKING:
    from typing import TypeAlias

    JSONValue: "TypeAlias" = (
        "None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]"
    )


def normalize_json(value: Any, *, path: str = "$") -> "JSONValue":
    """Return a JSON-compatible view of ``value`` or raise ``TypeError``.

    Rules: ``None``/``str``/``bool`` pass; ``int`` passes (``bool`` is handled
    first so ``True`` is not collapsed to ``1``); ``float`` must be finite;
    ``UUID`` is rendered as its canonical string; timezone-aware ``datetime``
    is rendered as UTC ISO 8601 (naive datetimes are rejected -- their zone is
    ambiguous); ``list``/``tuple`` become JSON arrays; ``Mapping`` becomes a
    JSON object whose keys must be strings. Everything else (``bytes``,
    ``set``, ``Decimal``, custom classes) is rejected -- callers must convert
    those explicitly rather than rely on an unstable string fallback."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path}: non-finite float is not valid JSON")
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise TypeError(f"{path}: naive datetime is not valid JSON (use tz-aware)")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (list, tuple)):
        return [normalize_json(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out: "dict[str, Any]" = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: non-string mapping key {key!r}")
            out[key] = normalize_json(item, path=f"{path}.{key}")
        return out
    raise TypeError(
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
