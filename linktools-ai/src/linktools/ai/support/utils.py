#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Low-level utility functions shared across modules."""

import json
import os
from collections.abc import Mapping
from typing import Any


def stable_json(payload: dict[str, Any]) -> str:
    """Serialize a dict with sorted keys and no extra whitespace, for stable hashing."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_ready(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a DB row to JSON-native types: datetimes → ISO strings, rest pass through.

    All capability/job columns are INT/VARCHAR/TEXT/CHAR/DATETIME, so no Decimal/bytes
    coercion is needed — a single typed pass replaces the old dumps/loads round-trip.
    """
    return {
        key: value.isoformat() if hasattr(value, "isoformat") else value
        for key, value in row.items()
    }


def resolve_ref(value: Any) -> Any:
    """Resolve env:<NAME>[:<FALLBACK>...] references to environment variables.

    Supports chained fallbacks: env:A:B:C returns the first non-empty value among
    os.getenv("A"), os.getenv("B"), os.getenv("C"), or None if all are unset/empty.
    """
    if not isinstance(value, str):
        return value
    if value.startswith("env:"):
        names = value[4:].split(":")
        for name in names:
            v = os.getenv(name)
            if v:
                return v
        return None
    return value


def truthy(value: Any) -> bool:
    """Normalize common string/boolean forms to True/False."""
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def model_type(value: Any) -> str:
    """Extract the model type string from a config value (str or dict)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("type") or value.get("model") or "standard")
    return "standard"


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge nested dicts while keeping unspecified default keys intact."""
    merged = dict(base or {})
    if not isinstance(override, Mapping):
        return merged
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged



def call_id(*parts: str) -> str:
    return ":".join(_safe_id_part(part) for part in parts if part)


def _safe_id_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _to_safe_name(value: str, fallback: str) -> str:
    result = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return result.strip("._-") or fallback



def safe_filename(value: str, fallback: str = "default") -> str:
    """Convert an arbitrary string to a valid filename (keeping alphanumerics and . _ -)."""
    return _to_safe_name(str(value), fallback)
