#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinkTools-owned durable codec for portable tool-return content."""

import binascii
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic_ai.messages import StructuredContent, TextContent, ToolReturn, ToolReturnContent, is_multi_modal_content
from pydantic_ai.tools import DeferredToolResults

from ..core import JsonValue, canonical_json_bytes, canonical_sha256, normalize_json_value
from ..errors import AIError, ErrorCode
from . import ReferencedContent
from ._user_content_codec import (
    RetiredReferencedContent, decode_user_content_item, encode_user_content_item, project_user_content_identity,
)
from ._json_snapshot import stable_json_snapshot

_JSON_SCALARS = (str, int, float, bool, type(None))
_TOOL_RETURN_CONTRACT = "linktools.tool-return"
_TOOL_RETURN_VERSION = 2


def encode_tool_return_content(value: object) -> JsonValue:
    """Encode portable tool content in the LinkTools durable result envelope."""
    try:
        return {
            "contract": _TOOL_RETURN_CONTRACT,
            "version": _TOOL_RETURN_VERSION,
            "value": _encode_node(value, version=_TOOL_RETURN_VERSION),
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
            or version not in (1, _TOOL_RETURN_VERSION)
        ):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return cast(ToolReturnContent, _decode_node(value["value"]))
    except AIError:
        raise
    except (TypeError, ValueError, KeyError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def tool_return_content_matches(value: object, stored: JsonValue) -> bool:
    """Compare a retry using the verified durable result envelope's one projection."""
    decode_tool_return_content(stored)
    envelope = cast(Mapping[str, JsonValue], stored)
    try:
        candidate = {
            "contract": _TOOL_RETURN_CONTRACT,
            "version": envelope["version"],
            "value": _encode_node(value, version=cast(int, envelope["version"])),
        }
        return canonical_json_bytes(candidate) == canonical_json_bytes(stored)
    except AIError:
        raise
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "external_result"}) from error


def _encode_node(
    value: object, *, version: int, reference_allowed: bool = True, tool_return_allowed: bool = True,
) -> JsonValue:
    if isinstance(value, ReferencedContent):
        if not reference_allowed:
            raise ValueError("referenced tool input must be a direct result or a top-level result item")
        return {"type": "referenced-content", "value": encode_user_content_item(value)}
    if isinstance(value, TextContent):
        return {
            "type": "text-content",
            "value": project_user_content_identity(encode_user_content_item(value), version=version),
        }
    if isinstance(value, ToolReturn):
        if not tool_return_allowed:
            _reject_nested_references(value.return_value)
            _reject_nested_references(value.content)
        native_items = value.return_value if isinstance(value.return_value, list) else [value.return_value]
        extra_items = value.content if value.content is not None and not isinstance(value.content, str) else ()
        if value.tools and any(isinstance(item, ReferencedContent) for item in (*native_items, *extra_items)):
            raise ValueError("referenced tool content cannot persist new executable tools")
        return {
            "type": "tool-return",
            "return_value": _encode_node(value.return_value, version=version, tool_return_allowed=False),
            "content": (value.content if value.content is None or isinstance(value.content, str)
                        else [project_user_content_identity(encode_user_content_item(item), version=version)
                              for item in value.content]),
            "metadata": stable_json_snapshot(value.metadata),
            "tools": None if value.tools is None else list(value.tools),
        }
    if is_multi_modal_content(value):
        return {
            "type": "multimodal",
            "value": encode_user_content_item(value),
        }
    if isinstance(value, Mapping) and all(
        isinstance(key, str) for key in value
    ):
        return {
            "type": "mapping",
            "items": {
                key: _encode_node(value[key], version=version, reference_allowed=False, tool_return_allowed=False)
                for key in sorted(value)
            },
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return {
            "type": "sequence",
            "items": [_encode_node(item, version=version, reference_allowed=reference_allowed and isinstance(value, list)
                                    and not isinstance(item, (Mapping, list, tuple)),
                                    tool_return_allowed=False) for item in value],
        }
    if isinstance(value, _JSON_SCALARS):
        return {
            "type": "scalar",
            "value": normalize_json_value(value),
        }
    try:
        snapshot = stable_json_snapshot(value)
    except (TypeError, ValueError) as error:
        raise TypeError("tool-return content is not JSON serializable") from error
    return {
        "type": "json-snapshot",
        "value": snapshot,
    }


def _reject_nested_references(value: object) -> None:
    if isinstance(value, ReferencedContent):
        raise ValueError("referenced tool input must be a direct result or a top-level result item")
    if isinstance(value, ToolReturn):
        _reject_nested_references(value.return_value)
        _reject_nested_references(value.content)
    elif isinstance(value, Mapping):
        for item in value.values():
            _reject_nested_references(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_nested_references(item)


def _decode_node(
    value: object, *, reference_allowed: bool = True, tool_return_allowed: bool = True,
) -> object:
    if not isinstance(value, Mapping):
        raise ValueError("tool-return node must be an object")
    node_type = value.get("type")
    if node_type == "tool-return":
        _require_keys(value, {"type", "return_value", "content", "metadata", "tools"})
        if ((value["content"] is not None and not isinstance(value["content"], (str, list)))
            or (value["tools"] is not None and (not isinstance(value["tools"], list)
                or any(not isinstance(tool, str) for tool in value["tools"])))):
            raise ValueError("referenced tool-return content is invalid")
        result = ToolReturn(
            return_value=_decode_node(value["return_value"], tool_return_allowed=False),
            content=([decode_user_content_item(item) for item in value["content"]]
                     if isinstance(value["content"], list) else value["content"]),
            metadata=normalize_json_value(value["metadata"]), tools=value["tools"],
        )
        if not tool_return_allowed:
            _reject_nested_references(result.return_value)
            _reject_nested_references(result.content)
        native_items = result.return_value if isinstance(result.return_value, list) else [result.return_value]
        extra_items = result.content if result.content is not None and not isinstance(result.content, str) else ()
        if result.tools and any(isinstance(item, ReferencedContent) for item in (*native_items, *extra_items)):
            raise ValueError("referenced tool-return tools are invalid")
        return result
    if node_type == "referenced-content":
        if not reference_allowed:
            raise ValueError("nested tool-return reference is invalid")
        _require_keys(value, {"type", "value"})
        item = decode_user_content_item(value["value"])
        if not isinstance(item, ReferencedContent):
            raise ValueError("tool-return reference is invalid")
        return item
    if node_type == "text-content":
        _require_keys(value, {"type", "value"})
        item = decode_user_content_item(value["value"])
        if type(item) not in (TextContent, StructuredContent, RetiredReferencedContent):
            raise ValueError("tool-return text content is invalid")
        return item
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
            key: _decode_node(items[key], reference_allowed=False, tool_return_allowed=False)
            for key in sorted(items)
        }
    if node_type == "sequence":
        _require_keys(value, {"type", "items"})
        items = value["items"]
        if not isinstance(items, list):
            raise ValueError("tool-return sequence is invalid")
        return [_decode_node(item, reference_allowed=reference_allowed and isinstance(item, Mapping)
                             and item.get("type") not in {"mapping", "sequence"},
                             tool_return_allowed=False) for item in items]
    if node_type == "multimodal":
        _require_keys(value, {"type", "value"})
        item = decode_user_content_item(value["value"])
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
