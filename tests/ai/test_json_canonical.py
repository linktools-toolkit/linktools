#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict canonical JSON: stable across processes, rejects non-JSON values
instead of silently stringifying them (the ``default=str`` hazard)."""

import math
from datetime import datetime, timezone
from uuid import UUID

import pytest

from linktools.ai.json import canonical_json, normalize_json


def test_dict_key_order_does_not_affect_output():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_list_order_affects_output():
    assert canonical_json([1, 2, 3]) != canonical_json([3, 2, 1])


def test_int_and_string_are_distinct():
    assert canonical_json(1) != canonical_json("1")


def test_bool_and_int_are_distinct():
    assert canonical_json(True) != canonical_json(1)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_float_rejected(bad):
    with pytest.raises(TypeError):
        canonical_json({"x": bad})


@pytest.mark.parametrize("bad", [b"bytes", {1, 2}, object()])
def test_non_json_types_rejected(bad):
    with pytest.raises(TypeError):
        canonical_json(bad)


def test_uuid_rendered_as_canonical_string():
    u = UUID("12345678-1234-5678-1234-567812345678")
    assert canonical_json(u) == f'"{u}"'


def test_timezone_aware_datetime_rendered_as_utc_iso():
    dt = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert canonical_json(dt) == '"2026-01-02T03:04:05+00:00"'


def test_naive_datetime_rejected():
    with pytest.raises(TypeError):
        canonical_json(datetime(2026, 1, 2, 3, 4, 5))


def test_non_string_mapping_key_rejected():
    with pytest.raises(TypeError):
        canonical_json({1: "a"})


def test_nested_structure_normalized():
    assert canonical_json({"a": (1, 2), "b": None}) == '{"a":[1,2],"b":null}'


def test_cross_call_stable():
    payload = {"tool": "t", "args": {"x": 1, "y": [True, "s", None]}}
    assert canonical_json(payload) == canonical_json(
        dict(reversed(list(payload.items())))
    )


def test_normalize_returns_json_compatible():
    # normalize_json is what makes the stability possible: it converts to a
    # plain JSON structure (list/tuple -> list, Mapping -> dict) first.
    out = normalize_json({"a": (1, 2)})
    assert out == {"a": [1, 2]}
