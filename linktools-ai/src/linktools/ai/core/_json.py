#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical JSON encoding used by immutable manifests and bindings."""

import json
from datetime import date, datetime
from enum import Enum
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _default(value: "datetime | date | Enum") -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Encode a JSON-compatible value deterministically."""
    return json.dumps(
        value,
        default=_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = ["JsonValue", "canonical_json_bytes"]
