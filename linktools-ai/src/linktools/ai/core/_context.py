#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable correlation context shared by Runtime-owned operations."""

import re
from collections.abc import Mapping
from typing import cast

from ._json import ImmutableJsonMapping, JsonValue, canonical_json_bytes

_CONTEXT_MAX_ITEMS = 8
_CONTEXT_KEY_MAX = 128
_CONTEXT_STRING_MAX = 256
_CONTEXT_BYTES_MAX = 4 * 1024
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_CONTEXT_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESERVED_PREFIX = "linktools."

RunContextValue = str | int
RunContextData = Mapping[str, RunContextValue]


def normalize_run_context(
    value: "Mapping[str, object] | None",
) -> RunContextData:
    """Validate one portable context and return an immutable canonical mapping."""
    if value is None:
        return cast(RunContextData, ImmutableJsonMapping({}))
    if not isinstance(value, Mapping):
        raise TypeError("context must be a mapping")
    if len(value) > _CONTEXT_MAX_ITEMS:
        raise ValueError("context contains too many entries")
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or len(key) > _CONTEXT_KEY_MAX
            or _CONTEXT_KEY_RE.fullmatch(key) is None
            or key.startswith(_RESERVED_PREFIX)
        ):
            raise ValueError("context key is invalid")
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise TypeError("context value must be string or integer")
        if isinstance(item, str):
            if not item or len(item) > _CONTEXT_STRING_MAX:
                raise ValueError("context string value is invalid")
        elif item < _INT64_MIN or item > _INT64_MAX:
            raise ValueError("context integer value is out of range")
        normalized[key] = item
    if len(canonical_json_bytes(normalized)) > _CONTEXT_BYTES_MAX:
        raise ValueError("context payload is too large")
    return cast(RunContextData, ImmutableJsonMapping(normalized))


def merge_run_context(
    base: "Mapping[str, object] | None",
    overlay: "Mapping[str, object] | None",
) -> RunContextData:
    """Merge portable context without allowing identity-changing overrides."""
    left = normalize_run_context(base)
    right = normalize_run_context(overlay)
    merged: dict[str, object] = dict(left)
    for key, value in right.items():
        previous = merged.get(key)
        if previous is not None and previous != value:
            raise ValueError("context key conflicts with inherited value")
        merged[key] = value
    return normalize_run_context(merged)


__all__ = [
    "RunContextData",
    "RunContextValue",
    "merge_run_context",
    "normalize_run_context",
]
