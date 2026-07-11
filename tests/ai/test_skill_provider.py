#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillProvider (contract): prompt catalog for skill:*, list/read tools, and an
authorization boundary that never leaks unauthorized skill content."""

import pytest

from linktools.ai.capability import CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.errors import SkillNotFoundError
from linktools.ai.skill import SkillProvider


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
    return _SkillSrc({
        "sql-analysis": _Spec("sql-analysis", "Analyze SQL logic.", "FULL SQL INSTRUCTIONS",
                              metadata={"tags": ["audit"]}),
        "incident-review": _Spec("incident-review", "Review incidents.", "FULL INCIDENT INSTRUCTIONS"),
    })


def _ctx():
    return CapabilityContext(agent_id="a1", exposure_policy=CapabilityToolExposurePolicy())


@pytest.mark.asyncio
async def test_skill_wildcard_injects_catalog_without_full_content():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(CapabilityRef("skill", "*"), _ctx())
    catalog = bundle.prompt_sections["skills"]
    assert "sql-analysis" in catalog and "incident-review" in catalog
    # Full content is NOT injected into the prompt.
    assert "FULL SQL INSTRUCTIONS" not in catalog
    assert "list_skills" in bundle.toolsets[0].tools
    assert "read_skill" in bundle.toolsets[0].tools


@pytest.mark.asyncio
async def test_skill_wildcard_read_skill_allowed_for_all():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(CapabilityRef("skill", "*"), _ctx())
    read_fn = bundle.toolsets[0].tools["read_skill"].function
    out = await read_fn("sql-analysis")
    assert out["content"] == "FULL SQL INSTRUCTIONS"


@pytest.mark.asyncio
async def test_skill_single_id_only_authorized_for_that_skill():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(CapabilityRef("skill", "sql-analysis"), _ctx())
    list_fn = bundle.toolsets[0].tools["list_skills"].function
    listing = await list_fn()
    ids = {s["id"] for s in listing["skills"]}
    assert ids == {"sql-analysis"}


@pytest.mark.asyncio
async def test_skill_unauthorized_read_does_not_leak():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(CapabilityRef("skill", "sql-analysis"), _ctx())
    read_fn = bundle.toolsets[0].tools["read_skill"].function
    with pytest.raises(SkillNotFoundError):
        await read_fn("incident-review")


@pytest.mark.asyncio
async def test_skill_single_does_not_inject_catalog():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(CapabilityRef("skill", "sql-analysis"), _ctx())
    assert "skills" not in bundle.prompt_sections  # no catalog for single-id


@pytest.mark.asyncio
async def test_skill_catalog_disabled_when_prompt_catalog_off():
    ctx = CapabilityContext(agent_id="a1",
                            exposure_policy=CapabilityToolExposurePolicy(expose_prompt_catalog=False))
    provider = SkillProvider(_src())
    bundle = await provider.resolve(CapabilityRef("skill", "*"), ctx)
    assert "skills" not in bundle.prompt_sections
    # Tools are still exposed (they are Level-1 discovery, gated separately).
    assert "list_skills" in bundle.toolsets[0].tools


@pytest.mark.asyncio
async def test_skill_list_filters_by_query():
    provider = SkillProvider(_src())
    bundle = await provider.resolve(CapabilityRef("skill", "*"), _ctx())
    list_fn = bundle.toolsets[0].tools["list_skills"].function
    listing = await list_fn(query="sql")
    assert {s["id"] for s in listing["skills"]} == {"sql-analysis"}
