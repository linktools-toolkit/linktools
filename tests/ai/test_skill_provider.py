#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillProvider (contract): prompt catalog for skill:*, list/read tools, and an
authorization boundary that never leaks unauthorized skill content."""

import pytest

from linktools.ai.agent.tool.exposure import ToolExposurePolicy
from linktools.ai.agent.assembly.provider import AgentFeatureContext
from linktools.ai.agent.assembly.models import AgentFeatureRef
from linktools.ai.errors import SkillNotFoundError
from linktools.ai.agent.skill import SkillProvider


class _Spec:
    def __init__(self, name, description, instructions, metadata=None):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.metadata = metadata or {}


class _SkillSrc:
    def __init__(self, skills):
        self._skills = skills

    async def list_ids(self):
        return tuple(self._skills.keys())

    async def get(self, skill_id):
        if skill_id not in self._skills:
            raise KeyError(skill_id)
        return self._skills[skill_id]


def _src():
    return _SkillSrc(
        {
            "sql-analysis": _Spec(
                "sql-analysis",
                "Analyze SQL logic.",
                "FULL SQL INSTRUCTIONS",
                metadata={"tags": ["audit"]},
            ),
            "incident-review": _Spec(
                "incident-review", "Review incidents.", "FULL INCIDENT INSTRUCTIONS"
            ),
        }
    )


def _ctx():
    return AgentFeatureContext(
        agent_id="a1",
        execution_id="e1",
        root_execution_id="e1",
        parent_execution_id=None,
        session_id="s1",
        tenant_id="t1",
        user_id="u1",
        workspace=None,
        sandbox=None,
    )


@pytest.mark.asyncio
async def test_skill_wildcard_injects_catalog_without_full_content():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(AgentFeatureRef("skill", "*"), _ctx())
    catalog = bundle.prompt_sections["skills"]
    assert "sql-analysis" in catalog and "incident-review" in catalog
    # Full content is NOT injected into the prompt.
    assert "FULL SQL INSTRUCTIONS" not in catalog
    names = {md.descriptor.name for md in bundle.tools}
    assert {"list_skills", "read_skill"} <= names


@pytest.mark.asyncio
async def test_skill_tools_preserve_feature_config_identity():
    provider = SkillProvider(_src())
    ref = AgentFeatureRef(
        "skill",
        "sql-analysis",
        config={"mode": "strict"},
    )
    bundle = await provider.resolve(ref, _ctx())
    assert all(tool.descriptor.feature == ref for tool in bundle.tools)


@pytest.mark.asyncio
async def test_skill_wildcard_read_skill_allowed_for_all():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(AgentFeatureRef("skill", "*"), _ctx())
    read_fn = next(
        md.handler
        for md in bundle.tools
        if md.descriptor.name == "read_skill"
    )
    out = await read_fn("sql-analysis")
    assert out["content"] == "FULL SQL INSTRUCTIONS"


@pytest.mark.asyncio
async def test_skill_single_id_only_authorized_for_that_skill():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(AgentFeatureRef("skill", "sql-analysis"), _ctx())
    list_fn = next(
        md.handler
        for md in bundle.tools
        if md.descriptor.name == "list_skills"
    )
    listing = await list_fn()
    ids = {s["id"] for s in listing["skills"]}
    assert ids == {"sql-analysis"}


@pytest.mark.asyncio
async def test_skill_unauthorized_read_does_not_leak():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(AgentFeatureRef("skill", "sql-analysis"), _ctx())
    read_fn = next(
        md.handler
        for md in bundle.tools
        if md.descriptor.name == "read_skill"
    )
    with pytest.raises(SkillNotFoundError):
        await read_fn("incident-review")


@pytest.mark.asyncio
async def test_skill_single_does_not_inject_catalog():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(AgentFeatureRef("skill", "sql-analysis"), _ctx())
    assert "skills" not in bundle.prompt_sections  # no catalog for single-id


@pytest.mark.asyncio
async def test_skill_catalog_is_owned_by_provider():
    ctx = _ctx()
    provider = SkillProvider(_src())
    bundle = await provider.resolve(AgentFeatureRef("skill", "*"), ctx)
    assert "skills" in bundle.prompt_sections
    assert any(
        md.descriptor.name == "list_skills"
        for md in bundle.tools
    )


@pytest.mark.asyncio
async def test_skill_list_filters_by_query():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(AgentFeatureRef("skill", "*"), _ctx())
    list_fn = next(
        md.handler
        for md in bundle.tools
        if md.descriptor.name == "list_skills"
    )
    listing = await list_fn(query="sql")
    assert {s["id"] for s in listing["skills"]} == {"sql-analysis"}
