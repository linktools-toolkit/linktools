#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable JSON-mode codec for portable Pydantic tool-return content."""

from typing import cast

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.messages import ToolReturnContent

from ..core import JsonValue, canonical_json_bytes, normalize_json_value
from ..errors import AIError, ErrorCode

_TOOL_RETURN_CONTENT_ADAPTER = TypeAdapter(ToolReturnContent)


def encode_tool_return_content(value: object) -> JsonValue:
    """Encode one portable tool result using Pydantic's public JSON-mode contract."""
    try:
        encoded = _TOOL_RETURN_CONTENT_ADAPTER.dump_json(value)
        restored = _TOOL_RETURN_CONTENT_ADAPTER.validate_json(encoded)
        canonical = _TOOL_RETURN_CONTENT_ADAPTER.dump_json(restored)
        import json

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


__all__: list[str] = []
