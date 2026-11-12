#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Step run sequence addresses owned by local admission rows."""

from ._record_key import decode_key_components
from .state import RECORD_KEY_MAX_BYTES, STREAM_KEY_MAX_BYTES, record_key, stream_key

RUN_COUNTER_COLUMNS = ("event_high_water", "checkpoint_high_water", "interaction_high_water")
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


__all__ = ["RUN_COUNTER_COLUMNS", "run_counter_address", "run_counter_streams"]
