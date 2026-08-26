#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill tool regression for model-correctable catalog misses."""

import pytest
from linktools.ai.asset import AssetRef
from linktools.ai.capability import CapabilityMaterializationContext, bind_skill_capability
from linktools.ai.core import ResourceKind, ResourceRef
from linktools.ai.spec import SkillSpec
from linktools.ai.workspace import trusted_workspace_principal
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


async def test_missing_skill_id_is_model_retry(tmp_path) -> None:
    binding = bind_skill_capability(
        (AssetRef("skill", "known"),),
        (SkillSpec("known", content="instructions"),),
    )
    context = CapabilityMaterializationContext(
        trusted_workspace_principal("tenant"),
        ResourceRef(ResourceKind.EXECUTION, "execution", "tenant"),
        tmp_path,
    )
    capabilities = await binding.materialize(context)
    assert len(capabilities) == 1
    toolset = capabilities[0].get_toolset()
    assert toolset is not None
    run_context = _context()
    tools = await toolset.get_tools(run_context)

    with pytest.raises(ModelRetry, match="skill not found"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "missing"},
            run_context,
            tools["load_skill"],
        )
