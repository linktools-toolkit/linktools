#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinkTools-owned durable codec for portable tool-return content."""

import binascii
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import ConfigDict, TypeAdapter
from pydantic_core import PydanticSerializationError
from pydantic_ai.messages import ToolReturnContent, is_multi_modal_content
from pydantic_ai.tools import DeferredToolResults

from ..core import JsonValue, canonical_sha256, normalize_json_value
from ..errors import AIError, ErrorCode
from ._input import _decode_user_content_item, _encode_user_content_item

_JSON_SCALARS = (str, int, float, bool, type(None))
_TOOL_RETURN_CONTRACT = "linktools.tool-return"
_TOOL_RETURN_VERSION = 1
_TOOL_RETURN_JSON_ADAPTER = TypeAdapter(
    Any,
    config=ConfigDict(ser_json_bytes="base64"),
)


def encode_tool_return_content(value: object) -> JsonValue:
    """Encode portable tool content without depending on Pydantic wire serialization."""
    try:
        return {
            "contract": _TOOL_RETURN_CONTRACT,
            "version": _TOOL_RETURN_VERSION,
            "value": _encode_node(value),
        }
    except AIError:
        raise
    except (TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={"field": "external_result"},
        ) from error


def decode_tool_return_content(value: JsonValue) -> ToolReturnContent:
    """Restore one LinkTools durable tool-return value."""
    try:
        if not isinstance(value, Mapping):
            raise ValueError("tool-return envelope must be an object")
        _require_keys(value, {"contract", "version", "value"})
        if value["contract"] != _TOOL_RETURN_CONTRACT:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        version = value["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != _TOOL_RETURN_VERSION
        ):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return cast(ToolReturnContent, _decode_node(value["value"]))
    except AIError:
        raise
    except (TypeError, ValueError, KeyError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _encode_node(value: object) -> JsonValue:
    if is_multi_modal_content(value):
        return {
            "type": "multimodal",
            "value": _encode_user_content_item(value),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("tool-return mapping keys must be strings")
        return {
            "type": "mapping",
            "items": {
                key: _encode_node(value[key])
                for key in sorted(value)
            },
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return {
            "type": "sequence",
            "items": [_encode_node(item) for item in value],
        }
    if isinstance(value, _JSON_SCALARS):
        return {
            "type": "scalar",
            "value": normalize_json_value(value),
        }
    try:
        snapshot = _TOOL_RETURN_JSON_ADAPTER.dump_python(
            value,
            mode="json",
            by_alias=True,
        )
    except PydanticSerializationError as error:
        raise TypeError("tool-return content is not JSON serializable") from error
    return {
        "type": "json-snapshot",
        "value": normalize_json_value(snapshot),
    }


def _decode_node(value: object) -> object:
    if not isinstance(value, Mapping):
        raise ValueError("tool-return node must be an object")
    node_type = value.get("type")
    if node_type == "scalar":
        _require_keys(value, {"type", "value"})
        scalar = normalize_json_value(value["value"])
        if not isinstance(scalar, _JSON_SCALARS):
            raise ValueError("tool-return scalar is invalid")
        return scalar
    if node_type == "mapping":
        _require_keys(value, {"type", "items"})
        items = value["items"]
        if not isinstance(items, Mapping) or any(
            not isinstance(key, str) for key in items
        ):
            raise ValueError("tool-return mapping is invalid")
        return {
            key: _decode_node(items[key])
            for key in sorted(items)
        }
    if node_type == "sequence":
        _require_keys(value, {"type", "items"})
        items = value["items"]
        if not isinstance(items, list):
            raise ValueError("tool-return sequence is invalid")
        return [_decode_node(item) for item in items]
    if node_type == "multimodal":
        _require_keys(value, {"type", "value"})
        item = _decode_user_content_item(value["value"])
        if not is_multi_modal_content(item):
            raise ValueError("tool-return multimodal item is invalid")
        return item
    if node_type == "json-snapshot":
        _require_keys(value, {"type", "value"})
        return normalize_json_value(value["value"])
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _require_keys(value: Mapping[object, object], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError("tool-return durable shape is invalid")


def rehydrate_deferred_tool_results(results: DeferredToolResults) -> DeferredToolResults:
    """Restore successful external durable values before the Pydantic boundary."""
    calls: dict[str, object] = {}
    for tool_call_id, result in results.calls.items():
        if (
            isinstance(result, Mapping)
            and result.get("contract") == _TOOL_RETURN_CONTRACT
        ):
            try:
                calls[tool_call_id] = decode_tool_return_content(
                    cast(JsonValue, normalize_json_value(result))
                )
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        else:
            calls[tool_call_id] = result
    return DeferredToolResults(
        calls=calls,
        approvals=dict(results.approvals),
        metadata={key: dict(value) for key, value in results.metadata.items()},
    )


def tool_return_content_digest(value: object) -> str | None:
    """Return a stable digest for content accepted by the durable codec."""
    try:
        return canonical_sha256(encode_tool_return_content(value))
    except AIError:
        return None


__all__: list[str] = []
