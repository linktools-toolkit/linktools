"""Strict canonical JSON primitives for local and SQL persistence."""

from __future__ import annotations

import json
import dataclasses
from datetime import date, datetime, time
from enum import Enum
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _normalize(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _normalize(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalize(model_dump(mode="python"))
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def encode_json(value: JsonValue) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, separators=(",", ":"))


def normalize_json(value: object) -> JsonValue:
    return _normalize(value)


def decode_json(raw: str) -> JsonValue:
    value = json.loads(raw)
    return _normalize(value)


def canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = ["JsonScalar", "JsonValue", "canonical_json_bytes", "decode_json", "encode_json", "normalize_json"]
