#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable JSON-mode codec for portable Pydantic tool-return content."""

import json
from typing import cast

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.messages import ToolReturnContent
from pydantic_ai.tools import DeferredToolResults

from ..core import JsonValue, canonical_json_bytes, canonical_sha256, normalize_json_value
from ..errors import AIError, ErrorCode

_TOOL_RETURN_CONTENT_ADAPTER = TypeAdapter(ToolReturnContent)
_JSON_TYPES = (str, int, float, bool, list, dict, type(None))


def encode_tool_return_content(value: object) -> JsonValue:
    """Encode one portable tool result using Pydantic's public JSON-mode contract."""
    try:
        encoded = _TOOL_RETURN_CONTENT_ADAPTER.dump_json(value)
        restored = _TOOL_RETURN_CONTENT_ADAPTER.validate_json(encoded)
        canonical = _TOOL_RETURN_CONTENT_ADAPTER.dump_json(restored)
        return normalize_json_value(json.loads(canonical.decode("utf-8")))
    except (TypeError, ValueError, UnicodeError, ValidationError) as error:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={"field": "external_result"},
        ) from error


def decode_tool_return_content(value: JsonValue) -> ToolReturnContent:
    """Restore one durable JSON-mode result to the public Pydantic content types."""
    try:
        return cast(
            ToolReturnContent,
            _TOOL_RETURN_CONTENT_ADAPTER.validate_json(canonical_json_bytes(value)),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def rehydrate_deferred_tool_results(results: DeferredToolResults) -> DeferredToolResults:
    """Restore successful external JSON values at the Pydantic execution boundary."""
    calls: dict[str, object] = {}
    for tool_call_id, result in results.calls.items():
        if isinstance(result, _JSON_TYPES):
            try:
                value = normalize_json_value(result)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            calls[tool_call_id] = decode_tool_return_content(value)
        else:
            calls[tool_call_id] = result
    return DeferredToolResults(
        calls=calls,
        approvals=dict(results.approvals),
        metadata={key: dict(value) for key, value in results.metadata.items()},
    )


def tool_return_content_digest(value: object) -> str | None:
    """Return one stable digest when the tool result has a portable content contract."""
    try:
        return canonical_sha256(encode_tool_return_content(value))
    except AIError:
        return None


__all__: list[str] = []