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

import json
import math
from datetime import datetime, timezone
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
