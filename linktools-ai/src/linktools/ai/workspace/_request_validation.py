#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semantic argument validation for local workspace operations."""

from __future__ import annotations

import math

from ..errors import AIError, ErrorCode

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


def validate_request_values(**params: object) -> None:
    """Validate argument values; Python signatures enforce operation fields."""
    for field, value in params.items():
        accepted_types = _REQUEST_VALUE_TYPES[field]
        if isinstance(value, bool) and (int in accepted_types or float in accepted_types):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(value, accepted_types):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if field == "timeout_seconds" and isinstance(value, float):
            if not math.isfinite(value):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)



__all__ = ["validate_request_values"]
