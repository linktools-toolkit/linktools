#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Step run sequence addresses owned by local admission rows."""

from ._record_key import decode_key_components, encode_key_components
from ._record_owner import RUN_COUNTER_COLUMNS
from .state import RECORD_KEY_MAX_BYTES, STREAM_KEY_MAX_BYTES, record_key, stream_key

_FAMILIES = ("event", "checkpoint", "interaction")
_DOMAINS = {"trace", "checkpoint", "conversation"}


def run_counter_address(key: bytes) -> tuple[bytes, str] | None:
    """Recognize reserved run streams; opaque State keys keep their own owner."""
    try:
        parts = decode_key_components(key, STREAM_KEY_MAX_BYTES)
    except (ValueError, UnicodeDecodeError):
        return None
    if len(parts) != 6 or parts[0] != "stream" or parts[3] not in _DOMAINS or parts[4] not in _FAMILIES:
        return None
    return record_key(parts[1], parts[2], parts[3], "agent_run", (parts[5],)), RUN_COUNTER_COLUMNS[_FAMILIES.index(parts[4])]


def run_counter_streams(key: bytes) -> tuple[bytes, ...]:
    """Project the three logical sequence addresses of an admission owner."""
    try:
        parts = decode_key_components(key, RECORD_KEY_MAX_BYTES)
    except (ValueError, UnicodeDecodeError):
        return ()
    if len(parts) != 6 or parts[0] != "record" or parts[3] not in _DOMAINS or parts[4] != "agent_run":
        return ()
    return tuple(stream_key(parts[1], parts[2], parts[3], family, (parts[5],)) for family in _FAMILIES)


def run_counter_page_start(after: bytes | None, column: str) -> tuple[bytes | None, bool] | None:
    """Map a logical exclusive cursor to one role's ordered admission key range.

    None means this role has no possible key after the cursor. Otherwise the
    pair is the physical lower bound (None for unbounded) and its inclusivity.
    The copied prefix is binary: opaque and truncated cursors need no decoding.
    """
    family = _FAMILIES[RUN_COUNTER_COLUMNS.index(column)]
    if after is None:
        return None, False
    source = encode_key_components(("stream",), STREAM_KEY_MAX_BYTES)
    target = encode_key_components(("record",), RECORD_KEY_MAX_BYTES)
    if not after.startswith(source):
        return (None, False) if after < source else None
    offset = len(source)
    for _ in range(3):
        if offset + 2 > len(after):
            return target + after[len(source):], False
        end = offset + 2 + int.from_bytes(after[offset:offset + 2], "big")
        if end > len(after):
            return target + after[len(source):], False
        offset = end
    prefix = target + after[len(source):offset]
    role = encode_key_components((family,), STREAM_KEY_MAX_BYTES)
    owner = encode_key_components(("agent_run",), RECORD_KEY_MAX_BYTES)
    suffix = after[offset:]
    if suffix.startswith(role):
        return prefix + owner + suffix[len(role):], False
    if suffix < role:
        return prefix, False
    # Skip the entire namespace/tenant/domain prefix for an earlier role.
    for index in range(len(prefix) - 1, -1, -1):
        if prefix[index] != 255:
            return prefix[:index] + bytes((prefix[index] + 1,)), True
    return None


__all__ = ["RUN_COUNTER_COLUMNS", "run_counter_address", "run_counter_streams", "run_counter_page_start"]
