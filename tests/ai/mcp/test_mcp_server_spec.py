#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPServerSpec enforces its domain invariants at construction, so a custom
provider that builds one directly (bypassing the registry parser) cannot create
an ungovernable server. The registry is the first line of defense; this is the
second."""

import math

import pytest

from linktools.ai.registry.mcp import MCPServerSpec


def _valid(**overrides) -> MCPServerSpec:
    base = {"id": "s", "name": "s", "transport": "stdio", "command": ("python",)}
    base.update(overrides)
    return MCPServerSpec(**base)


def test_valid_stdio_spec_accepted():
    spec = _valid()
    assert spec.id == "s"
    assert spec.command == ("python",)


@pytest.mark.parametrize(
    "field,value,exc",
    [
        ("env", {"K": 1}, ValueError),
        ("headers", {"X": True}, ValueError),
        ("enabled_tools", (1,), ValueError),
        ("disabled_tools", (None,), ValueError),
    ],
)
def test_typed_fields_reject_bad_element(field, value, exc):
    with pytest.raises(exc):
        _valid(**{field: value})


def test_tool_prefix_empty_string_rejected():
    with pytest.raises(ValueError, match="tool_prefix must not be empty"):
        _valid(tool_prefix="   ")


def test_tool_prefix_wrong_type_rejected():
    # A list is neither str/bool/None.
    with pytest.raises(TypeError, match="tool_prefix must be a string"):
        _valid(tool_prefix=[])


def test_tool_prefix_accepts_bool_and_string():
    assert _valid(tool_prefix=True).tool_prefix is True
    assert _valid(tool_prefix=False).tool_prefix is False
    assert _valid(tool_prefix="srv").tool_prefix == "srv"


def test_invalid_discovery_mode_rejected():
    with pytest.raises(ValueError, match="unknown discovery_mode"):
        _valid(discovery_mode="lax")


def test_invalid_transport_rejected():
    with pytest.raises(ValueError, match="unknown transport"):
        _valid(transport="ftp")


def test_empty_id_and_name_rejected():
    with pytest.raises(ValueError, match="id must be a non-empty"):
        _valid(id="")
    with pytest.raises(ValueError, match="name must be a non-empty"):
        _valid(name="")


def test_stdio_requires_command():
    with pytest.raises(
        ValueError, match="stdio transport requires a non-empty command"
    ):
        MCPServerSpec(id="s", name="s", transport="stdio")


def test_sse_requires_url():
    with pytest.raises(ValueError, match="sse transport requires a url"):
        MCPServerSpec(id="s", name="s", transport="sse")


@pytest.mark.parametrize(
    "bad", [math.nan, math.inf, -1, 0], ids=["nan", "inf", "neg", "zero"]
)
def test_timeout_must_be_positive_finite(bad):
    with pytest.raises(ValueError, match="timeout_seconds"):
        _valid(timeout_seconds=bad)


def test_command_must_be_tuple_of_strings():
    with pytest.raises(ValueError, match="command must be a tuple"):
        _valid(command=("python", 1))
    with pytest.raises(ValueError, match="command must be a tuple"):
        _valid(command=["python"])
