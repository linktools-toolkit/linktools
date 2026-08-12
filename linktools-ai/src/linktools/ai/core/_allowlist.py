#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical string allowlist values."""

from collections.abc import Sequence

from ..errors import AIError, ErrorCode


def canonical_string_tuple(value: Sequence[str], *, field: str) -> "tuple[str, ...]":
    """Validate, deduplicate, and sort one string selector sequence."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field} must be an array of strings")
    selectors: list[str] = []
    for selector in value:
        if not isinstance(selector, str) or not selector or selector != selector.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field} contains an invalid selector")
        selectors.append(selector)
    normalized = tuple(sorted(set(selectors)))
    if "*" in normalized and normalized != ("*",):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{field} cannot mix '*' with other selectors")
    return normalized


__all__ = ["canonical_string_tuple"]
