#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Business (custom) Provider examples satisfy the same Provider contracts as the
default registries -- no agent.md/SKILL.md/mcp.yaml required (contract)."""

import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.registry.mcp import MCPServerSpec
from linktools.ai.registry.skill import SkillSpec

from ._assertions import assert_spec_provider_contract


def _agent(i):
    return AgentSpec(id=i, name=i, model=ModelPolicy(primary="m"),
                     instructions=PromptSpec(instructions="hi"))


class _BusinessAgentProvider:
    async def list_ids(self):
        return ("audit-reviewer",)

    async def get(self, agent_id):
        if agent_id != "audit-reviewer":
            raise KeyError(agent_id)
        return _agent(agent_id)


class _BusinessSkillProvider:
    async def list_ids(self):
        return ("sql",)

    async def get(self, sid):
        if sid != "sql":
            raise KeyError(sid)
        return SkillSpec(id=sid, name=sid, description="d", instructions="x")


class _BusinessMcpProvider:
    async def list_ids(self):
        return ("risk",)

    async def get(self, sid):
        if sid != "risk":
            raise KeyError(sid)
        return MCPServerSpec(id=sid, name=sid, transport="stdio", command_or_url="python -m r",
                             command=("python", "-m", "r"))


@pytest.mark.asyncio
async def test_business_agent_provider_contract():
    await assert_spec_provider_contract(
        _BusinessAgentProvider(), sample_id="audit-reviewer", expected_type=AgentSpec)


@pytest.mark.asyncio
async def test_business_skill_provider_contract():
    await assert_spec_provider_contract(
        _BusinessSkillProvider(), sample_id="sql", expected_type=SkillSpec)


@pytest.mark.asyncio
async def test_business_mcp_provider_contract():
    await assert_spec_provider_contract(
        _BusinessMcpProvider(), sample_id="risk", expected_type=MCPServerSpec)
