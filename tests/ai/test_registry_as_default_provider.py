#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The default file-backed registries satisfy their Provider Protocols -- they
are the recommended-format Provider implementations, usable wherever a Runtime
accepts a Provider. A business stub Provider satisfies the same Protocol too."""

import pytest

from linktools.ai.providers.agent import AgentSpecProvider
from linktools.ai.providers.mcp import MCPServerSpecProvider
from linktools.ai.providers.skill import SkillSpecProvider
from linktools.ai.providers.swarm import SwarmSpecProvider
from linktools.ai.providers.tool_policy import ToolPolicyMetadataSource
from linktools.ai.registry import (
    AgentRegistry,
    MCPRegistry,
    SkillRegistry,
    SwarmRegistry,
    ToolRegistry,
)


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
        AgentRegistry(_DummyLoader()),
        SkillRegistry(_DummyLoader()),
        MCPRegistry(_DummyLoader()),
        SwarmRegistry(_DummyLoader()),
        ToolRegistry(_DummyLoader()),
    )


@pytest.mark.parametrize("registry", _registries())
def test_default_registries_satisfy_protocols(registry):
    if isinstance(registry, AgentRegistry):
        assert isinstance(registry, AgentSpecProvider)
    if isinstance(registry, SkillRegistry):
        assert isinstance(registry, SkillSpecProvider)
    if isinstance(registry, MCPRegistry):
        assert isinstance(registry, MCPServerSpecProvider)
    if isinstance(registry, SwarmRegistry):
        assert isinstance(registry, SwarmSpecProvider)
    if isinstance(registry, ToolRegistry):
        assert isinstance(registry, ToolPolicyMetadataSource)


def test_tool_registry_exposes_protocol_method_name():
    # ToolPolicyMetadataSource.get_metadata_map -- the canonical Protocol name.
    tr = ToolRegistry(_DummyLoader())
    assert callable(getattr(tr, "get_metadata_map", None))


class _BusinessAgentProvider:
    """Business stub: no agent.md anywhere, just returns Specs."""

    async def list_ids(self):
        return ("audit-reviewer",)

    async def get(self, agent_id):
        raise KeyError(agent_id)


def test_business_provider_satisfies_protocol_without_any_registry():
    assert isinstance(_BusinessAgentProvider(), AgentSpecProvider)
