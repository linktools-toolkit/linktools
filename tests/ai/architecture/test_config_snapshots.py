#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final config-format snapshots.

Each fixture under ``tests/ai/fixtures/config/`` is loaded through its registry
so the canonical config format is locked: a that silently renames a
field or reintroduces an alias fails here. Agent/Skill are markdown + YAML
frontmatter (``.md``); Tool/MCP/Swarm are YAML (``.yaml``) -- the native
formats each registry reads. No deprecated alias is used in any fixture.
"""

import asyncio
from pathlib import Path


CONFIG_DIR = Path(__file__).parent.parent / "fixtures" / "config"


def _loader():
    from linktools.ai.catalog.parsing import SpecLoader

    return SpecLoader.from_filesystem(CONFIG_DIR)


def test_agent_config_snapshot():
    from linktools.ai.agent.catalog import AgentCatalog

    registry = AgentCatalog.from_specloader(_loader())
    spec = asyncio.run(registry.get("agent"))
    assert spec.name == "writer"
    assert spec.model.primary == "gpt-4o"
    assert [t.name for t in spec.tools] == ["file", "terminal"]
    assert "careful writer" in spec.instructions.instructions


def test_skill_config_snapshot():
    from linktools.ai.skill.catalog import SkillCatalog

    registry = SkillCatalog.from_specloader(_loader())
    spec = asyncio.run(registry.get("skill"))
    assert spec.name == "greeter"
    assert spec.description == "says hello"
    assert "Greet" in spec.instructions


def test_tool_config_snapshot():
    from linktools.ai.governance.policy.rule import (
        ApprovalMode,
        Permission,
        RiskLevel,
        SideEffectKind,
    )
    from linktools.ai.tool.catalog import ToolCatalog

    spec = asyncio.run(ToolCatalog.from_specloader(_loader()).get("tool"))
    assert spec.name == "tool"
    assert spec.description == "shell"
    assert spec.permissions == frozenset({Permission.EXECUTE, Permission.WRITE})
    assert spec.risk is RiskLevel.HIGH
    assert spec.side_effect is SideEffectKind.DESTRUCTIVE
    assert spec.approval is ApprovalMode.ON_RISK


def test_mcp_config_snapshot():
    from linktools.ai.mcp.catalog import MCPCatalog

    spec = asyncio.run(MCPCatalog.from_specloader(_loader()).get("mcp"))
    assert spec.id == "mcp"
    assert spec.name == "search"
    assert spec.transport == "stdio"
    assert " ".join(spec.command) == "python -m search_server"
    assert dict(spec.env) == {"API_KEY": "xxx"}


def test_swarm_config_snapshot():
    from linktools.ai.swarm.catalog import SwarmCatalog
    from linktools.ai.swarm.aggregation import AggregationMode

    spec = asyncio.run(SwarmCatalog.from_specloader(_loader()).get("swarm"))
    assert spec.name == "research"
    assert [a.agent_id for a in spec.agents] == ["searcher", "writer"]
    assert spec.coordinator.agent_id == "planner"
    assert spec.strategy.kind == "parallel_fan_out"
    assert spec.limits.max_rounds == 5
    assert spec.aggregation.mode is AggregationMode.CONCAT
