#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen MESSAGE_WIRE_V1 codec for durable ModelMessage values."""

import hashlib
import json
import math
from collections.abc import Sequence
from typing import cast

from jsonschema import Draft202012Validator
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from ...core import JsonValue, canonical_json_bytes
from ...errors import AIError, ErrorCode
from ._v1_schema import (
    V1_SERIALIZATION_SCHEMA_JSON,
    V1_SERIALIZATION_SCHEMA_SHA256,
    V1_VALIDATION_SCHEMA_JSON,
    V1_VALIDATION_SCHEMA_SHA256,
)


def _load_schema(schema_json: str, expected_sha256: str) -> dict[str, object]:
    actual_sha256 = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError("GA v1 model-message schema manifest is corrupt")
    try:
        schema = json.loads(schema_json)
    except (TypeError, ValueError) as error:
        raise RuntimeError("GA v1 model-message schema manifest is invalid") from error
    if not isinstance(schema, dict):
        raise RuntimeError("GA v1 model-message schema manifest is invalid")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise RuntimeError("GA v1 model-message schema manifest is invalid") from error
    return schema


_SERIALIZATION_SCHEMA = _load_schema(
    V1_SERIALIZATION_SCHEMA_JSON,
    V1_SERIALIZATION_SCHEMA_SHA256,
)
_VALIDATION_SCHEMA = _load_schema(
    V1_VALIDATION_SCHEMA_JSON,
    V1_VALIDATION_SCHEMA_SHA256,
)
_SERIALIZATION_VALIDATOR = Draft202012Validator(_SERIALIZATION_SCHEMA)
_VALIDATION_VALIDATOR = Draft202012Validator(_VALIDATION_SCHEMA)


def _validate_json_value(value: object, *, reading: bool) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        if reading:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "GA v1 model-message wire requires finite JSON values",
            )
        raise ValueError("GA v1 model-message wire requires finite JSON values")
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, reading=reading)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                if reading:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                raise ValueError(
                    "GA v1 model-message wire requires JSON object string keys"
                )
            _validate_json_value(item, reading=reading)
        return
    if reading:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raise ValueError("GA v1 model-message wire contains a non-JSON value")


def _validate_serialization(value: object, *, reading: bool) -> JsonValue:
    _validate_json_value(value, reading=reading)
    return cast(JsonValue, value)


def _raise_serializer_unsupported() -> AIError:
    return AIError(
        ErrorCode.STORAGE_VERSION_UNSUPPORTED,
        "current model-message serializer cannot write GA v1",
    )


def _raise_reader_unsupported() -> AIError:
    return AIError(
        ErrorCode.STORAGE_VERSION_UNSUPPORTED,
        "current model-message reader cannot read GA v1",
    )


def encode_v1_model_messages(messages: Sequence[ModelMessage]) -> bytes:
    value = ModelMessagesTypeAdapter.dump_python(
        list(messages),
        mode="json",
    )
    json_value = _validate_serialization(value, reading=False)
    if not _SERIALIZATION_VALIDATOR.is_valid(json_value):
        raise _raise_serializer_unsupported()
    return canonical_json_bytes(json_value)


def _decode_json(raw: bytes) -> JsonValue:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (
        AttributeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
    ) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return _validate_serialization(value, reading=True)


def _decode_current_messages(value: JsonValue) -> tuple[ModelMessage, ...]:
    try:
        messages = ModelMessagesTypeAdapter.validate_python(value)
    except Exception as error:
        raise _raise_reader_unsupported() from error
    return tuple(messages)


def decode_v1_model_messages(raw: bytes) -> tuple[ModelMessage, ...]:
    value = _decode_json(raw)
    if canonical_json_bytes(value) != raw:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not _VALIDATION_VALIDATOR.is_valid(value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    messages = _decode_current_messages(value)
    try:
        encoded = ModelMessagesTypeAdapter.dump_python(
            list(messages),
            mode="json",
        )
        round_trip = _validate_serialization(encoded, reading=False)
    except Exception as error:
        raise _raise_reader_unsupported() from error
    if not _SERIALIZATION_VALIDATOR.is_valid(round_trip):
        raise _raise_reader_unsupported()
    if canonical_json_bytes(round_trip) != raw:
        raise _raise_reader_unsupported()
    return messages


__all__ = ["decode_v1_model_messages", "encode_v1_model_messages"]
