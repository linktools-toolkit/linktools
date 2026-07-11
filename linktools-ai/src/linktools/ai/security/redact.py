#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument safety for audit-facing copies: mask values under secret-looking
keys and cap size, so tool arguments persisted to the approval store / event
log never leak credentials or unbounded payloads.

Applied ONLY to audit copies (e.g. RunPaused.arguments -> ApprovalRequest),
never to the arguments actually passed to the tool handler -- the handler must
receive the real values to do its job, and on resume the model re-emits the
real arguments from message history, not the masked audit copy."""

from typing import Any, Mapping

# Key substrings that mark a value as sensitive. Conservative: matches common
# credential field names; a real value under e.g. "command" stays visible so an
# approver can still see what a terminal tool will run.
_SECRET_KEY_MARKERS = (
    "secret", "password", "passwd", "token", "credential", "api_key",
    "apikey", "private_key", "access_key", "auth", "session",
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
        redacted = _truncate(value)
        total += len(str(redacted))
        if total > _MAX_TOTAL_CHARS:
            out[str(key)] = f"***TRUNCATED: total audit size cap {_MAX_TOTAL_CHARS} reached***"
            break
        out[str(key)] = redacted
    return out


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_VALUE_CHARS:
        return value[:_MAX_VALUE_CHARS] + f"...***TRUNCATED ({len(value)} chars)***"
    return value
