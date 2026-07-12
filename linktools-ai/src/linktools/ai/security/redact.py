#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument safety for audit-facing copies: mask values under secret-looking
keys and cap size, so tool arguments persisted to the approval store / event
log never leak credentials or unbounded payloads.

Applied ONLY to audit copies (e.g. RunPaused.arguments -> ApprovalRequest),
never to the arguments actually passed to the tool handler -- the handler must
receive the real values to do its job, and on resume the model re-emits the
real arguments from message history, not the masked audit copy."""

import re
from typing import Any, Mapping

# Key substrings that mark a value as sensitive. Conservative: matches common
# credential field names; a real value under e.g. "command" stays visible so an
# approver can still see what a terminal tool will run.
_SECRET_KEY_MARKERS = (
    "authorization",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "cookie",
    "set-cookie",
)

_MASK = "***REDACTED***"
# Per-value and total caps for an audit copy.
_MAX_VALUE_CHARS = 2048
_MAX_TOTAL_CHARS = 16384


def _looks_secret(key: str) -> bool:
    k = key.lower()
    return any(m in k for m in _SECRET_KEY_MARKERS)


def redact_for_audit(arguments: "Mapping[str, Any] | None") -> "dict[str, Any]":
    """Return an audit-safe copy of ``arguments``: values under secret-looking
    keys are replaced with a mask, oversized string values are truncated, and
    the total encoded size is capped. The input is never mutated."""
    if not arguments:
        return {}
    out: "dict[str, Any]" = {}
    total = 0
    for key, value in arguments.items():
        if _looks_secret(str(key)):
            out[str(key)] = _MASK
            continue
        redacted = _redact_value(value)
        total += len(str(redacted))
        if total > _MAX_TOTAL_CHARS:
            out[str(key)] = (
                f"***TRUNCATED: total audit size cap {_MAX_TOTAL_CHARS} reached***"
            )
            break
        out[str(key)] = redacted
    return out


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_VALUE_CHARS:
        return value[:_MAX_VALUE_CHARS] + f"...***TRUNCATED ({len(value)} chars)***"
    return value


_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[^\s,;]+"),
    re.compile(r"(?i)\bBasic\s+[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]+"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
)


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (_MASK if _looks_secret(str(key)) else _redact_value(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(_MASK, value)
        return _truncate(value)
    return value


def redact_exception(error: BaseException, *, max_chars: int = _MAX_VALUE_CHARS) -> str:
    text = str(error)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_MASK, text)
    if len(text) > max_chars:
        text = text[:max_chars] + "...<truncated>"
    return text
