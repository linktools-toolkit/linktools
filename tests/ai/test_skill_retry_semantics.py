#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill tool regression for model-correctable catalog misses."""

import pytest
from linktools.ai.capability import SkillCapability, SkillCatalogSnapshot, SkillDescriptor
from linktools.ai.spec import SkillSpec
from pydantic_ai.exceptions import ModelRetry

pytestmark = pytest.mark.asyncio


async def test_missing_skill_id_is_model_retry() -> None:
    catalog = SkillCatalogSnapshot(
        (SkillDescriptor("known", 1, "Known skill"),),
        (SkillSpec("known", content="instructions"),),
    )
    capability = SkillCapability(catalog, id="linktools-skill")
    toolset = capability.get_toolset()

    with pytest.raises(ModelRetry, match="skill not found"):
        await toolset.load_skill(None, "missing")
