#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nested registry declarations are strictly validated: unknown fields are
rejected (so a typo like ``agentd_id`` cannot be silently ignored), names are
stripped, and types are enforced. Grouped by object."""

import pytest

from linktools.ai.agent.spec import MiddlewareRef, ToolRef
from linktools.ai.errors import InvalidSpecError
from linktools.ai.agent.codec import parse_middleware_refs
from linktools.ai.mcp.codec import parse_mcp_spec
from linktools.ai.tool.codec import parse_tool_refs
from linktools.ai.swarm.codec import _parse_agent_ref
from linktools.ai.swarm.spec import AgentRef


# ---------------------------------------------------------------------------
# ToolRef
# ---------------------------------------------------------------------------


def test_tool_ref_rejects_unknown_field():
    with pytest.raises(InvalidSpecError, match="unknown fields"):
        parse_tool_refs([{"kind": "builtin", "name": "t", "extra": 1}])


def test_tool_ref_rejects_blank_kind_and_name():
    with pytest.raises(InvalidSpecError, match="kind must not be blank"):
        parse_tool_refs([{"kind": "   ", "name": "t"}])
    with pytest.raises(InvalidSpecError, match="name must not be blank"):
        parse_tool_refs([{"kind": "builtin", "name": "   "}])


def test_tool_ref_strips_surrounding_whitespace():
    refs = parse_tool_refs([{"kind": "  builtin ", "name": " t "}])
    assert refs == (ToolRef(name="t", kind="builtin", config={}),)


def test_tool_ref_rejects_non_mapping_config():
    with pytest.raises(InvalidSpecError, match="config must be a mapping"):
        parse_tool_refs([{"kind": "builtin", "name": "t", "config": []}])


def test_tool_ref_rejects_null_config():
    with pytest.raises(InvalidSpecError, match="must not be null"):
        parse_tool_refs([{"kind": "builtin", "name": "t", "config": None}])


# ---------------------------------------------------------------------------
# MiddlewareRef
# ---------------------------------------------------------------------------


def test_middleware_ref_rejects_unknown_field():
    with pytest.raises(InvalidSpecError, match="unknown fields"):
        parse_middleware_refs([{"name": "m", "extra": 1}])


def test_middleware_ref_rejects_blank_name():
    with pytest.raises(InvalidSpecError, match="name must not be blank"):
        parse_middleware_refs(["   "])
    with pytest.raises(InvalidSpecError, match="name must not be blank"):
        parse_middleware_refs([{"name": "  "}])
    with pytest.raises(InvalidSpecError, match="name is required"):
        parse_middleware_refs([{}])


def test_middleware_ref_strips_name():
    refs = parse_middleware_refs(["  log  "])
    assert refs == (MiddlewareRef(name="log"),)
    refs = parse_middleware_refs([{"name": " log ", "config": {"x": 1}}])
    assert refs == (MiddlewareRef(name="log", config={"x": 1}),)


def test_middleware_ref_rejects_non_mapping_config():
    with pytest.raises(InvalidSpecError, match="config must be a mapping"):
        parse_middleware_refs([{"name": "m", "config": []}])


# ---------------------------------------------------------------------------
# AgentRef
# ---------------------------------------------------------------------------


def test_agent_ref_string_rejects_blank():
    with pytest.raises(InvalidSpecError, match="agent_id must not be blank"):
        _parse_agent_ref("   ", swarm_id="sw", kind="agent")


def test_agent_ref_mapping_rejects_unknown_field():
    with pytest.raises(InvalidSpecError, match="unknown fields"):
        _parse_agent_ref({"agentd_id": "worker"}, swarm_id="sw", kind="agent")


def test_agent_ref_strips_agent_id():
    ref = _parse_agent_ref("  worker ", swarm_id="sw", kind="agent")
    assert ref == AgentRef(agent_id="worker")
    ref = _parse_agent_ref(
        {"agent_id": " worker ", "role": " analyst "},
        swarm_id="sw",
        kind="agent",
    )
    assert ref == AgentRef(agent_id="worker", role="analyst")


def test_agent_ref_rejects_blank_role():
    with pytest.raises(InvalidSpecError, match="role must not be blank"):
        _parse_agent_ref({"agent_id": "w", "role": "   "}, swarm_id="sw", kind="agent")


def test_agent_ref_rejects_non_string_agent_id():
    with pytest.raises(InvalidSpecError, match="agent_id must be a string"):
        _parse_agent_ref({"agent_id": 5}, swarm_id="sw", kind="agent")


# ---------------------------------------------------------------------------
# MCP command
# ---------------------------------------------------------------------------


def _mcp(command) -> None:
    parse_mcp_spec("s", {"transport": "stdio", "command": command})


def test_mcp_command_rejects_whitespace_only_part():
    with pytest.raises(InvalidSpecError, match=r"command\[1\] must not be blank"):
        _mcp(["python", "   "])


def test_mcp_command_rejects_non_string_part():
    with pytest.raises(InvalidSpecError, match=r"command\[0\] must be a string"):
        _mcp([1])


def test_mcp_command_rejects_empty_list():
    with pytest.raises(InvalidSpecError, match="non-empty string or list"):
        _mcp([])


def test_mcp_command_accepts_valid_list():
    spec = parse_mcp_spec(
        "s", {"transport": "stdio", "command": ["python", "-m", "server"]}
    )
    assert spec.command == ("python", "-m", "server")
