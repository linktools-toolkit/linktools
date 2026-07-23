#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StrictConfigReader distinguishes a MISSING field (default applies) from an
explicit ``null`` (rejected). Covers every accessor, plus the ModelPolicy
timeout NaN/Infinity rejection that the hand-written check used to miss."""

import math

import pytest

from linktools.ai.errors import InvalidSpecError
from linktools.ai.mcp.codec import parse_mcp_spec
from linktools.ai.catalog.parsing import StrictConfigReader, parse_model_policy


def _reader(payload, *, allowed, context="test"):
    return StrictConfigReader(payload, allowed=allowed, context=context)


# ---------------------------------------------------------------------------
# 1. enabled_tools (string_tuple): missing -> None, [] -> (), null -> reject.
# ---------------------------------------------------------------------------


def test_string_tuple_missing_returns_default():
    reader = _reader({}, allowed={"enabled_tools"})
    assert reader.string_tuple("enabled_tools", default=None) is None
    assert reader.string_tuple("enabled_tools", default=()) == ()


def test_string_tuple_empty_list_is_empty_tuple():
    reader = _reader({"enabled_tools": []}, allowed={"enabled_tools"})
    assert reader.string_tuple("enabled_tools", default=None) == ()


def test_string_tuple_null_rejected():
    reader = _reader({"enabled_tools": None}, allowed={"enabled_tools"})
    with pytest.raises(InvalidSpecError, match="must not be null"):
        reader.string_tuple("enabled_tools")


def test_mcp_enabled_tools_three_state():
    missing = parse_mcp_spec("s", {"transport": "stdio", "command": ["a"]})
    assert missing.enabled_tools is None
    empty = parse_mcp_spec(
        "s", {"transport": "stdio", "command": ["a"], "enabled_tools": []}
    )
    assert empty.enabled_tools == ()
    with pytest.raises(InvalidSpecError, match="must not be null"):
        parse_mcp_spec(
            "s", {"transport": "stdio", "command": ["a"], "enabled_tools": None}
        )


# ---------------------------------------------------------------------------
# 2. positive_int / non_negative_int: missing -> default, null -> reject.
# ---------------------------------------------------------------------------


def test_positive_int_missing_returns_default_null_rejected():
    reader = _reader({}, allowed={"max_rounds"})
    assert reader.positive_int("max_rounds", default=5) == 5
    reader = _reader({"max_rounds": None}, allowed={"max_rounds"})
    with pytest.raises(InvalidSpecError, match="must not be null"):
        reader.positive_int("max_rounds", default=5)


def test_non_negative_int_missing_returns_default_null_rejected():
    reader = _reader({}, allowed={"max_delegations"})
    assert reader.non_negative_int("max_delegations", default=0) == 0
    reader = _reader({"max_delegations": None}, allowed={"max_delegations"})
    with pytest.raises(InvalidSpecError, match="must not be null"):
        reader.non_negative_int("max_delegations", default=0)


# ---------------------------------------------------------------------------
# 3. positive_number (timeout): missing -> default, null -> reject,
#    NaN/+Inf/-Inf -> reject (the hand-written check used to let them through).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "inf", "-inf"],
)
def test_positive_number_rejects_non_finite(bad):
    reader = _reader({"timeout_seconds": bad}, allowed={"timeout_seconds"})
    with pytest.raises(InvalidSpecError, match="must be a positive number"):
        reader.positive_number("timeout_seconds", default=30.0)


def test_positive_number_missing_returns_default_null_rejected():
    reader = _reader({}, allowed={"timeout_seconds"})
    assert reader.positive_number("timeout_seconds", default=30.0) == 30.0
    reader = _reader({"timeout_seconds": None}, allowed={"timeout_seconds"})
    with pytest.raises(InvalidSpecError, match="must not be null"):
        reader.positive_number("timeout_seconds", default=30.0)


def test_model_policy_timeout_rejects_nan_inf_and_null():
    base = {"primary": "gpt"}
    with pytest.raises(InvalidSpecError, match="positive number"):
        parse_model_policy({**base, "timeout_seconds": math.nan})
    with pytest.raises(InvalidSpecError, match="positive number"):
        parse_model_policy({**base, "timeout_seconds": math.inf})
    with pytest.raises(InvalidSpecError, match="must not be null"):
        parse_model_policy({**base, "timeout_seconds": None})


def test_model_policy_request_retries_null_rejected_missing_defaults():
    base = {"primary": "gpt"}
    assert parse_model_policy(dict(base)).request_retries == 1
    with pytest.raises(InvalidSpecError, match="must not be null"):
        parse_model_policy({**base, "request_retries": None})


# ---------------------------------------------------------------------------
# 4. mapping / string_mapping: missing -> default/None, {} -> {}, null -> reject.
# ---------------------------------------------------------------------------


def test_mapping_missing_returns_none_empty_kept_null_rejected():
    reader = _reader({}, allowed={"metadata"})
    assert reader.mapping("metadata") is None
    reader = _reader({"metadata": {}}, allowed={"metadata"})
    assert reader.mapping("metadata") == {}
    reader = _reader({"metadata": None}, allowed={"metadata"})
    with pytest.raises(InvalidSpecError, match="must not be null"):
        reader.mapping("metadata")


def test_optional_str_missing_returns_none_null_rejected():
    reader = _reader({}, allowed={"name"})
    assert reader.optional_str("name") is None
    reader = _reader({"name": None}, allowed={"name"})
    with pytest.raises(InvalidSpecError, match="must not be null"):
        reader.optional_str("name")


def test_bool_missing_returns_default_null_rejected():
    reader = _reader({}, allowed={"flag"})
    assert reader.bool("flag", default=True) is True
    reader = _reader({"flag": None}, allowed={"flag"})
    with pytest.raises(InvalidSpecError, match="must not be null"):
        reader.bool("flag", default=True)
