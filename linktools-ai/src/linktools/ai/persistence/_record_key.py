#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded binary record references shared by state and archive reads."""

RECORD_KEY_MAX_BYTES = 1152


def validate_record_key(value: bytes) -> bytes:
    """Validate an exact bounded record address, including custom StateStore keys."""
    if type(value) is not bytes or not 1 <= len(value) <= RECORD_KEY_MAX_BYTES:
        raise ValueError(f"record key must contain 1..{RECORD_KEY_MAX_BYTES} bytes")
    return value


__all__ = ["RECORD_KEY_MAX_BYTES", "validate_record_key"]
