#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider Protocols exist and are structurally satisfiable; the default
registries conform to them; the format aliases point at the right classes."""

import pytest

from linktools.ai.providers import (
    AgentBackedSubagentSpecProvider,
    AgentSpecProvider,
    MCPServerSpecProvider,
    PackageResourceProvider,
    PackageSpecProvider,
    SkillSpecProvider,
    SubagentSpecProvider,
    SwarmSpecProvider,
    ToolPolicyProvider,
)
from linktools.ai.registry import (
    AgentRegistry,
    MarkdownAgentRegistry,
    MarkdownSkillRegistry,
    MCPRegistry,
    SkillRegistry,
    SwarmRegistry,
    ToolRegistry,
    YamlMCPRegistry,
    YamlSwarmRegistry,
    YamlToolPolicyRegistry,
)


def test_format_aliases_are_the_canonical_registries():
    assert MarkdownAgentRegistry is AgentRegistry
    assert MarkdownSkillRegistry is SkillRegistry
    assert YamlMCPRegistry is MCPRegistry
    assert YamlToolPolicyRegistry is ToolRegistry
    assert YamlSwarmRegistry is SwarmRegistry


def test_protocols_importable_and_distinct():
    protos = {
        AgentSpecProvider, SkillSpecProvider, MCPServerSpecProvider,
        ToolPolicyProvider, SwarmSpecProvider, SubagentSpecProvider,
        PackageSpecProvider, PackageResourceProvider,
    }
    assert len(protos) == 8


class _RecordingAgentProvider:
    def __init__(self, ids=("a1", "a2")):
        self._ids = ids
        self.calls = []

    async def list_ids(self):
        self.calls.append("list_ids")
        return self._ids

    async def get(self, agent_id):
        self.calls.append(("get", agent_id))
        return object()  # AgentSpec stand-in; delegation does not introspect it


@pytest.mark.asyncio
async def test_agent_backed_subagent_provider_delegates():
    backing = _RecordingAgentProvider()
    sub = AgentBackedSubagentSpecProvider(backing)
    assert isinstance(sub, SubagentSpecProvider)
    assert await sub.list_ids() == ("a1", "a2")
    await sub.get("a1")
    assert backing.calls == ["list_ids", ("get", "a1")]


def test_runtime_checkable_protocols_match_duck_types():
    # runtime_checkable verifies attribute/method presence only.
    class _Stub:
        async def list_ids(self): ...
        async def get(self, x): ...
    assert isinstance(_Stub(), AgentSpecProvider)
    assert isinstance(_Stub(), SkillSpecProvider)
    assert isinstance(_Stub(), MCPServerSpecProvider)
    assert isinstance(_Stub(), SwarmSpecProvider)
    assert isinstance(_Stub(), SubagentSpecProvider)
    assert isinstance(_Stub(), PackageSpecProvider)
