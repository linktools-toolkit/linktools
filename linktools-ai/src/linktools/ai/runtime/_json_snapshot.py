#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic JSON snapshots for runtime-owned durable codecs."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import cast

from pydantic import BaseModel
from pydantic_core import PydanticSerializationError, to_jsonable_python

from ..core import JsonValue, canonical_json_bytes


def stable_json_snapshot(value: object) -> JsonValue:
    """Convert one portable Python value to deterministic JSON data."""
    if value is None or isinstance(value, (bool, int, str)):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("durable JSON requires finite floats")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized_key = _mapping_key(key)
            if normalized_key in normalized:
                raise TypeError("durable JSON mapping keys collide after encoding")
            normalized[normalized_key] = stable_json_snapshot(item)
        return {
            key: normalized[key]
            for key in sorted(normalized)
        }
    if isinstance(value, (set, frozenset)):
        items = [stable_json_snapshot(item) for item in value]
        items.sort(key=canonical_json_bytes)
        return items
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [stable_json_snapshot(item) for item in value]
    if isinstance(value, BaseModel):
        return stable_json_snapshot(
            value.model_dump(mode="python", by_alias=True)
        )
    if is_dataclass(value) and not isinstance(value, type):
        return stable_json_snapshot(
            {
                field.name: getattr(value, field.name)
                for field in fields(value)
            }
        )
    try:
        snapshot = to_jsonable_python(
            value,
            by_alias=True,
            bytes_mode="base64",
        )
    except PydanticSerializationError as error:
        raise TypeError("value is not JSON portable") from error
    if snapshot is value:
        raise TypeError("value is not JSON portable")
    return stable_json_snapshot(snapshot)


def _mapping_key(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        snapshot = to_jsonable_python(
            {value: None},
            by_alias=True,
            bytes_mode="base64",
        )
    except (PydanticSerializationError, TypeError) as error:
        raise TypeError("durable JSON mapping key is not portable") from error
    if not isinstance(snapshot, dict) or len(snapshot) != 1:
        raise TypeError("durable JSON mapping key is not portable")
    key = next(iter(snapshot))
    if not isinstance(key, str):
        raise TypeError("durable JSON mapping key is not portable")
    return key


__all__: list[str] = []
