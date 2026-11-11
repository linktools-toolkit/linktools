#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded binary record references shared by state and archive reads."""

from collections.abc import Sequence


RECORD_KEY_MAX_BYTES = 1152


def validate_record_key(value: bytes) -> bytes:
    """Validate an exact bounded record address, including custom StateStore keys."""
    if type(value) is not bytes or not 1 <= len(value) <= RECORD_KEY_MAX_BYTES:
        raise ValueError(f"record key must contain 1..{RECORD_KEY_MAX_BYTES} bytes")
    return value


def encode_key_components(values: Sequence[str], maximum: int) -> bytes:
    components: list[bytes] = []
    size = 0
    for value in values:
        if not isinstance(value, str):
            raise TypeError("address components must be strings")
        encoded = value.encode("utf-8")
        size += 2 + len(encoded)
        if size > maximum:
            raise ValueError("address exceeds its byte limit")
        components.append(len(encoded).to_bytes(2, "big") + encoded)
    return b"".join(components)


def decode_key_components(value: bytes, maximum: int) -> tuple[str, ...]:
    """Decode an exact fixed-u16 UTF-8 tuple, rejecting partial components."""
    if type(value) is not bytes or not 1 <= len(value) <= maximum:
        raise ValueError("address exceeds its byte limit")
    parts: list[str] = []
    offset = 0
    while offset < len(value):
        if offset + 2 > len(value):
            raise ValueError("incomplete address length")
        size = int.from_bytes(value[offset:offset + 2], "big")
        offset += 2
        end = offset + size
        if end > len(value):
            raise ValueError("incomplete address component")
        parts.append(value[offset:end].decode("utf-8"))
        offset = end
    return tuple(parts)


__all__ = ["RECORD_KEY_MAX_BYTES", "validate_record_key", "encode_key_components", "decode_key_components"]
