#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider Protocols exist and are structurally satisfiable; the default
Catalogs conform to them."""

import pytest

from linktools.ai.agent.spec import AgentSpecProvider
from linktools.ai.mcp.spec import MCPServerSpecProvider
from linktools.ai.extension.spec import ExtensionResourceProvider, ExtensionSpecProvider
from linktools.ai.skill.models import SkillSpecProvider
from linktools.ai.subagent.models import (
    AgentBackedSubagentSpecProvider,
    SubagentSpecProvider,
)
from linktools.ai.swarm.spec import SwarmSpecProvider
from linktools.ai.governance.policy.rule import ToolPolicyMetadataSource


def test_protocols_importable_and_distinct():
    protos = {
        AgentSpecProvider,
        SkillSpecProvider,
        MCPServerSpecProvider,
        ToolPolicyMetadataSource,
        SwarmSpecProvider,
        SubagentSpecProvider,
        ExtensionSpecProvider,
        ExtensionResourceProvider,
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
    assert isinstance(_Stub(), ExtensionSpecProvider)
