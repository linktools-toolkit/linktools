#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semantic argument validation for local workspace operations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..errors import AIError, ErrorCode

_REQUEST_FIELDS = {
    "run_command": (
        frozenset({"command"}),
        frozenset({"timeout_seconds"}),
    ),
    "start_command": (frozenset({"command"}), frozenset()),
    "check_command": (frozenset({"command_id"}), frozenset()),
    "stop_command": (frozenset({"command_id"}), frozenset()),
    "inspect_file": (frozenset({"path"}), frozenset({"max_bytes"})),
    "read_text": (
        frozenset({"path"}),
        frozenset({"offset", "limit", "cursor", "full_metadata", "max_bytes", "max_read_bytes"}),
    ),
    "list_files": (frozenset(), frozenset({"path", "include_glob"})),
    "search_text": (frozenset({"pattern"}), frozenset({"path", "include_glob"})),
    "write_file": (
        frozenset({"path", "content"}),
        frozenset({"old_text", "expected_hash"}),
    ),

}
_REQUEST_VALUE_TYPES = {
    "path": (str,),
    "pattern": (str,),
    "include_glob": (str, type(None)),
    "content": (str,),
    "old_text": (str, type(None)),
    "expected_hash": (str, type(None)),
    "offset": (int, type(None)),
    "cursor": (str, type(None)),
    "full_metadata": (bool,),
    "max_bytes": (int, type(None)),
    "max_read_bytes": (int, type(None)),
    "limit": (int, type(None)),
    "command": (str,),
    "command_id": (str,),
    "timeout_seconds": (int, float, type(None)),
}


def validate_request_params(
    method: str,
    params: Mapping[str, Any],
) -> None:
    """Validate the fixed fields and value types for one workspace operation."""
    if not isinstance(method, str):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    schema = _REQUEST_FIELDS.get(method)
    if schema is None or not isinstance(params, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    required, optional = schema
    fields = set(params)
    if (
        any(not isinstance(field, str) for field in fields)
        or not required.issubset(fields)
        or fields - required - optional
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    for field in fields:
        value = params[field]
        accepted_types = _REQUEST_VALUE_TYPES[field]
        if isinstance(value, bool) and (int in accepted_types or float in accepted_types):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(value, accepted_types):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if field == "timeout_seconds" and isinstance(value, float):
            if not math.isfinite(value):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)



__all__ = ["validate_request_params"]
