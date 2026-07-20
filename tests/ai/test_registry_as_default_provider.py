#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The default file-backed registries satisfy their Provider Protocols -- they
are the recommended-format Provider implementations, usable wherever a Runtime
accepts a Provider. A business stub Provider satisfies the same Protocol too."""

import pytest

from linktools.ai.agent.catalog import AgentCatalog
from linktools.ai.mcp.catalog import MCPCatalog
from linktools.ai.agent.spec import AgentSpecProvider
from linktools.ai.mcp.spec import MCPServerSpecProvider
from linktools.ai.skill.models import SkillSpecProvider
from linktools.ai.swarm.spec import SwarmSpecProvider
from linktools.ai.governance.policy.rule import ToolPolicyMetadataSource
from linktools.ai.skill.catalog import SkillCatalog
from linktools.ai.swarm.catalog import SwarmCatalog
from linktools.ai.tool.catalog import ToolCatalog


class _DummyLoader:
    """registries only touch the loader inside list_ids/get, which we don't
    call here -- conformance is structural."""

    async def revision(self):
        return 0

    async def list_ids(self, suffix):
        return ()

    async def read(self, name):
        raise FileNotFoundError(name)


def _registries():
    return (
        AgentCatalog.from_specloader(_DummyLoader()),
        SkillCatalog.from_specloader(_DummyLoader()),
        MCPCatalog.from_specloader(_DummyLoader()),
        SwarmCatalog.from_specloader(_DummyLoader()),
        ToolCatalog.from_specloader(_DummyLoader()),
    )


@pytest.mark.parametrize("registry", _registries())
def test_default_registries_satisfy_protocols(registry):
    if isinstance(registry, AgentCatalog):
        assert isinstance(registry, AgentSpecProvider)
    if isinstance(registry, SkillCatalog):
        assert isinstance(registry, SkillSpecProvider)
    if isinstance(registry, MCPCatalog):
        assert isinstance(registry, MCPServerSpecProvider)
    if isinstance(registry, SwarmCatalog):
        assert isinstance(registry, SwarmSpecProvider)
    if isinstance(registry, ToolCatalog):
        assert isinstance(registry, ToolPolicyMetadataSource)


def test_tool_registry_exposes_protocol_method_name():
    # ToolPolicyMetadataSource.get_metadata_map -- the canonical Protocol name.
    tr = ToolCatalog.from_specloader(_DummyLoader())
    assert callable(getattr(tr, "get_metadata_map", None))


class _BusinessAgentProvider:
    """Business stub: no agent.md anywhere, just returns Specs."""

    async def list_ids(self):
        return ("audit-reviewer",)

    async def get(self, agent_id):
        raise KeyError(agent_id)


def test_business_provider_satisfies_protocol_without_any_registry():
    assert isinstance(_BusinessAgentProvider(), AgentSpecProvider)
