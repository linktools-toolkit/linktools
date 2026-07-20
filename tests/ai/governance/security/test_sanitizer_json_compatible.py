#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sanitizer produces JSON-compatible values.

The sanitized event must round-trip through ``json.dumps`` without a
``default=`` fallback: containers normalize to lists, bytes/Enum/UUID/datetime
convert to JSON natives, and unknown objects become truncated strings.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID

from linktools.ai.governance.security.emitter import DefaultSecurityEventSanitizer


class _Color(Enum):
    RED = "red"


def _sanitize(value):
    return DefaultSecurityEventSanitizer().sanitize({"v": value})["v"]


def test_tuple_becomes_list():
    out = _sanitize(("a", "b"))
    assert out == ["a", "b"]
    assert isinstance(out, list)


def test_set_and_frozenset_become_list():
    out_set = _sanitize({1, 2, 3})
    out_frozen = _sanitize(frozenset({4, 5}))
    assert isinstance(out_set, list) and sorted(out_set) == [1, 2, 3]
    assert isinstance(out_frozen, list) and sorted(out_frozen) == [4, 5]


def test_bytes_become_binary_redacted():
    assert _sanitize(b"secret-bytes") == "<binary-redacted>"
    assert _sanitize(bytearray(b"x")) == "<binary-redacted>"


def test_enum_becomes_value():
    assert _sanitize(_Color.RED) == "red"


def test_uuid_becomes_str():
    u = UUID("12345678-1234-5678-1234-567812345678")
    assert _sanitize(u) == str(u)


def test_datetime_becomes_iso8601():
    dt = datetime(2026, 7, 12, 1, 2, 3, tzinfo=timezone.utc)
    assert _sanitize(dt) == dt.isoformat()


def test_unknown_object_becomes_truncated_string():
    class _Unknown:
        def __str__(self):
            return "z" * 5000

    out = _sanitize(_Unknown())
    assert isinstance(out, str)
    assert out.endswith("<truncated>")
    assert len(out) <= 4096 + len("<truncated>")


def test_sanitized_event_is_json_serializable_without_default():
    """The whole point: json.dumps succeeds with no default fallback."""
    event = {
        "tuple": (1, 2),
        "set": {1, 2},
        "bytes": b"x",
        "enum": _Color.RED,
        "uuid": UUID("12345678-1234-5678-1234-567812345678"),
        "dt": datetime(2026, 7, 12, tzinfo=timezone.utc),
        "nested": [{"a": (1, 2)}],
    }
    sanitized = DefaultSecurityEventSanitizer().sanitize(event)
    # No default= -> would raise TypeError if any non-JSON native survived.
    json.dumps(sanitized)
