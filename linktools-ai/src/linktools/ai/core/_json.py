"""Canonical JSON encoding used by immutable manifests and bindings."""

import json
import math
from collections.abc import Iterator, Mapping
from datetime import date, datetime
from enum import Enum
from typing import TypeAlias, cast

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class ImmutableJsonMapping(Mapping[str, JsonValue]):
    """Store one JSON object canonically and return detached values on access."""

    __slots__ = ("_payload",)

    def __init__(self, value: Mapping[str, JsonValue]) -> None:
        self._payload = canonical_json_bytes(_normalize_mapping(value))

    def __getitem__(self, key: str) -> JsonValue:
        return self._decode()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decode())

    def __len__(self) -> int:
        return len(self._decode())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self._decode() == dict(other)

    def _decode(self) -> "dict[str, JsonValue]":
        value = json.loads(self._payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("immutable JSON mapping payload must be an object")  # noqa: TRY004
        return cast("dict[str, JsonValue]", value)


def _normalize_mapping(value: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("JSON object keys must be non-empty strings")
        normalized[key] = _normalize_value(item)
    return normalized


def _normalize_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Mapping):
        return _normalize_mapping(cast("Mapping[str, JsonValue]", value))
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


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


__all__ = ["ImmutableJsonMapping", "JsonValue", "canonical_json_bytes"]
