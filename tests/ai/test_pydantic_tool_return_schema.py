#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic tool return-schema regressions for runtime capabilities."""

import warnings

from linktools.ai.capability import (
    SkillCapability,
    SkillDefinition,
    SkillSourceRegistry,
    SubagentCapability,
)
from linktools.ai.core import JsonValue
from linktools.ai.runtime._skill_adapter import _PydanticSkillCapability
from linktools.ai.runtime._subagent_adapter import _PydanticSubagentCapability
from linktools.ai.spec import SkillSpec, SubagentRef


def test_pydantic_capability_tool_return_schemas_are_constrained() -> None:
    async def delegate(
        _ref: SubagentRef,
        _task: str,
        *,
        invocation_id: str,
    ) -> dict[str, JsonValue]:
        assert invocation_id
        return {
            "execution_id": "child",
            "status": "SUCCEEDED",
            "output": {"value": True},
        }

    skill = _PydanticSkillCapability(
        SkillCapability(
            (SkillDefinition(SkillSpec("skill", content="instructions")),),
            SkillSourceRegistry(),
        )
    )
    subagent = _PydanticSubagentCapability(
        SubagentCapability((SubagentRef("agent", "child"),), delegate)
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"Could not generate return schema.*",
            category=UserWarning,
        )
        assert skill.get_toolset() is not None
        assert subagent.get_toolset() is not None
