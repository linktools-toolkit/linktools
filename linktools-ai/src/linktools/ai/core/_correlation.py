#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable correlation metadata shared by Runtime-owned operations."""

import re
from collections.abc import Mapping
from typing import cast

from ._json import ImmutableJsonMapping, JsonValue, canonical_json_bytes

_CORRELATION_MAX_ITEMS = 8
_CORRELATION_KEY_MAX = 128
_CORRELATION_STRING_MAX = 256
_CORRELATION_BYTES_MAX = 4 * 1024
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_CORRELATION_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESERVED_CORRELATION_PREFIX = "linktools."

CorrelationValue = str | int
CorrelationData = Mapping[str, CorrelationValue]


def normalize_correlation(
    value: "Mapping[str, object] | None",
) -> CorrelationData:
    """Validate correlation metadata and return an immutable canonical mapping."""
    if value is None:
        return cast(CorrelationData, ImmutableJsonMapping({}))
    if not isinstance(value, Mapping):
        raise TypeError("correlation must be a mapping")
    if len(value) > _CORRELATION_MAX_ITEMS:
        raise ValueError("correlation contains too many entries")
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or len(key) > _CORRELATION_KEY_MAX
            or _CORRELATION_KEY_RE.fullmatch(key) is None
            or key.startswith(_RESERVED_CORRELATION_PREFIX)
        ):
            raise ValueError("correlation key is invalid")
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise TypeError("correlation value must be string or integer")
        if isinstance(item, str):
            if not item or len(item) > _CORRELATION_STRING_MAX:
                raise ValueError("correlation string value is invalid")
        elif item < _INT64_MIN or item > _INT64_MAX:
            raise ValueError("correlation integer value is out of range")
        normalized[key] = item
    if len(canonical_json_bytes(normalized)) > _CORRELATION_BYTES_MAX:
        raise ValueError("correlation payload is too large")
    return cast(CorrelationData, ImmutableJsonMapping(normalized))


def overlay_correlation(
    base: "Mapping[str, object] | None",
    overlay: "Mapping[str, object] | None",
) -> CorrelationData:
    """Overlay operation correlation metadata on inherited Runtime defaults."""
    left = normalize_correlation(base)
    right = normalize_correlation(overlay)
    return normalize_correlation({**left, **right})


__all__ = [
    "CorrelationData",
    "CorrelationValue",
    "normalize_correlation",
    "overlay_correlation",
]
