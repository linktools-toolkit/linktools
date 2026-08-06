#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic JSON primitives used by public fingerprints and storage."""

import json
from typing import Any, TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def normalize_json(value: Any) -> JsonValue:
    """Convert a Pydantic or mapping value into JSON-compatible data."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [normalize_json(item) for item in value]
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value with stable ordering and separators."""
    return json.dumps(
        normalize_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    """Return stable JSON text."""
    return canonical_json_bytes(value).decode("utf-8")


__all__ = ["JsonScalar", "JsonValue", "canonical_json_bytes", "canonical_json_text", "normalize_json"]
