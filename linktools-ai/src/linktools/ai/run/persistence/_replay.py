#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared replay-hash utility for Run commit coordinators.

Both the SQL commit-log and the Filesystem journal dedupe a retried commit by
(commit_id, request_hash). The hash must be identical across backends so a
caller that re-issues the same payload against either backend produces the
same key -- the Filesystem coordinator therefore uses the SAME
canonical_request_hash the SQL coordinator's commit-log table keys on."""

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping


def _json_default(value: Any) -> Any:
    """JSON encoder for the few non-JSON-native types a commit request/result
    carries: datetime/date (ISO 8601 string), Enum (its value). Anything else
    raises TypeError so an exotic type is surfaced rather than silently
    stringified by a catch-all coercion (which would let a payload drift
    across versions without notice)."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"commit payload is not JSON-serializable: {type(value)!r}")


def canonical_request_hash(operation: str, payload: "Mapping[str, Any]") -> bytes:
    """SHA-256 of the operation tag + the canonical-JSON payload. The payload
    is JSON-serialized with sort_keys=True so two calls with the same logical
    content hash identically regardless of dict insertion order. Non-JSON-native
    values (datetime / Enum) are stringified via _json_default so they round-
    trip deterministically; callers needing a stricter canonical form pre-
    serialize their payload."""
    payload_json = json.dumps(payload, sort_keys=True, default=_json_default)
    hasher = hashlib.sha256()
    hasher.update(len(operation).to_bytes(8, "big"))
    hasher.update(operation.encode("utf-8"))
    hasher.update(len(payload_json).to_bytes(8, "big"))
    hasher.update(payload_json.encode("utf-8"))
    return hasher.digest()
