"""Canonical persistence conversion for Pydantic AI ModelMessage values."""

import json
import math
from collections.abc import Sequence
from typing import cast

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode


def _json_value(value: object, *, reading: bool) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if reading:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "model message persistence requires finite JSON values",
            )
        raise ValueError("model message persistence requires finite JSON values")
    if isinstance(value, list):
        return [_json_value(item, reading=reading) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                if reading:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                raise ValueError("model message persistence requires string object keys")
            result[key] = _json_value(item, reading=reading)
        return result
    if reading:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raise ValueError("model message persistence contains a non-JSON value")


def encode_model_messages(messages: Sequence[ModelMessage]) -> bytes:
    value = ModelMessagesTypeAdapter.dump_python(
        list(messages),
        mode="json",
    )
    return canonical_json_bytes(_json_value(value, reading=False))


def decode_model_messages(raw: bytes) -> tuple[ModelMessage, ...]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    json_value = _json_value(value, reading=True)
    if canonical_json_bytes(json_value) != raw:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        messages = ModelMessagesTypeAdapter.validate_json(raw)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return tuple(messages)


__all__ = ["decode_model_messages", "encode_model_messages"]
