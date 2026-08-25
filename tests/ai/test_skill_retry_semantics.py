#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill tool regression for model-correctable catalog misses."""

import pytest
from linktools.ai.capability import SkillCapability, SkillCatalogSnapshot, SkillDescriptor
from linktools.ai.spec import SkillSpec
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


async def test_missing_skill_id_is_model_retry() -> None:
    catalog = SkillCatalogSnapshot(
        (SkillDescriptor("known", 1, "Known skill"),),
        (SkillSpec("known", content="instructions"),),
    )
    capability = SkillCapability(catalog, id="linktools-skill")
    toolset = capability.get_toolset()
    tool = toolset.tools["load_skill"]

    with pytest.raises(ModelRetry, match="skill not found"):
        await tool.function(_context(), "missing")
