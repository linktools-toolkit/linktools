#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default file-backed registries satisfy the Provider contracts (contract).
Downstream may replace any of these with a business provider that returns the
same standard Specs."""

import pytest

from linktools.ai.spec.parsing import SpecLoader
from linktools.ai.agent.index import AgentSpecIndex
from linktools.ai.tool.mcp.index import MCPServerSpecIndex
from linktools.ai.agent.skill.index import SkillSpecIndex
from linktools.ai.agent.skill.spec import SkillSpec
from linktools.ai.tool.index import ToolSpecIndex

from ._assertions import (
    assert_spec_provider_contract,
    assert_tool_policy_provider_contract,
)


@pytest.fixture
def base(tmp_path):
    # agent/skill/mcp/tool sample specs
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "writer.md").write_text(
        "---\nname: writer\nmodel:\n  primary: gpt-4o\n---\nYou write.\n",
        encoding="utf-8",
    )
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "sql.md").write_text(
        "---\nname: sql\n---\nSQL instructions.\n", encoding="utf-8"
    )
    mcp = tmp_path / "mcp"
    mcp.mkdir()
    (mcp / "search.yaml").write_text(
        "name: search\ntransport: stdio\ncommand: [python, -m, search]\n",
        encoding="utf-8",
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "read_file.yaml").write_text(
        "name: read_file\npermissions: [read]\nrisk: LOW\n", encoding="utf-8"
    )
    return tmp_path


@pytest.mark.asyncio
async def test_agent_registry_contract(base):
    from linktools.ai.agent.spec import AgentSpec

    reg = AgentSpecIndex.from_specloader(SpecLoader.from_filesystem(base / "agents"))
    await assert_spec_provider_contract(
        reg, sample_id="writer", expected_type=AgentSpec
    )


@pytest.mark.asyncio
async def test_skill_registry_contract(base):
    reg = SkillSpecIndex.from_specloader(SpecLoader.from_filesystem(base / "skills"))
    await assert_spec_provider_contract(reg, sample_id="sql", expected_type=SkillSpec)


@pytest.mark.asyncio
async def test_mcp_registry_contract(base):
    from linktools.ai.tool.mcp.spec import MCPServerSpec

    reg = MCPServerSpecIndex.from_specloader(SpecLoader.from_filesystem(base / "mcp"))
    await assert_spec_provider_contract(
        reg, sample_id="search", expected_type=MCPServerSpec
    )


@pytest.mark.asyncio
async def test_tool_registry_contract(base):
    reg = ToolSpecIndex.from_specloader(SpecLoader.from_filesystem(base / "tools"))
    await assert_tool_policy_provider_contract(reg, sample_name="read_file")
