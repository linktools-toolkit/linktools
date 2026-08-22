#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured redaction at observation and error boundaries."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..errors import AIError, ErrorCode, SafeError
from ._ids import canonical_sha256
from ._json import JsonValue


class RedactionClass(StrEnum):
    PUBLIC = "PUBLIC"
    IDENTIFIER = "IDENTIFIER"
    SENSITIVE = "SENSITIVE"
    SECRET = "SECRET"
    MODEL_REASONING = "MODEL_REASONING"


@dataclass(frozen=True, slots=True)
class RedactedValue:
    value: JsonValue
    redacted: bool
    digest: "str | None"


class RedactionPolicy(Protocol):
    def redact(self, value: JsonValue, *, classification: RedactionClass) -> RedactedValue: ...
    def safe_error(self, error: BaseException, *, operation_id: str) -> SafeError: ...


class StructuredRedactor:
    def redact(self, value: JsonValue, *, classification: RedactionClass) -> RedactedValue:
        try:
            if classification is RedactionClass.MODEL_REASONING:
                return RedactedValue(None, True, None)
            if classification is RedactionClass.SECRET:
                return RedactedValue("<redacted>", True, canonical_sha256(value))
            if classification is RedactionClass.SENSITIVE:
                return RedactedValue(_redact_nested(value, set()), True, canonical_sha256(value))
            return RedactedValue(value, False, None)
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.REDACTION_FAILED) from error

    def safe_error(self, error: BaseException, *, operation_id: str) -> SafeError:
        if isinstance(error, AIError):
            return error.to_safe_error(operation_id=operation_id)
        return SafeError(
            ErrorCode.INTERNAL_ERROR.value,
            "INTERNAL",
            False,
            operation_id,
            {},
            canonical_sha256({"type": type(error).__name__}),
        )


def _redact_nested(value: JsonValue, seen: set[int]) -> JsonValue:
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            raise AIError(ErrorCode.REDACTION_FAILED)
        seen.add(marker)
        result = {str(key): _redact_nested(item, seen) for key, item in value.items()}
        seen.remove(marker)
        return result
    if isinstance(value, list):
        marker = id(value)
        if marker in seen:
            raise AIError(ErrorCode.REDACTION_FAILED)
        seen.add(marker)
        result = [_redact_nested(item, seen) for item in value]
        seen.remove(marker)
        return result
    return "<redacted>"


__all__ = ["RedactedValue", "RedactionClass", "RedactionPolicy", "StructuredRedactor"]
