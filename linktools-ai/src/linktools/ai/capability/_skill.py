#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution capability for one compiler-selected immutable Skill set."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import FunctionToolset

from ..errors import AIError, ErrorCode
from ..spec import SkillSpec
from ._context import RunContext
from ._names import SKILL_TOOL_NAMES


@dataclass
class SkillCapability(AbstractCapability[RunContext[object]]):
    skills: "tuple[SkillSpec, ...]"
    id: str = "linktools-skill"

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.skills, key=lambda item: item.id))
        ids = tuple(item.id for item in ordered)
        if len(ids) != len(set(ids)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self.skills = ordered

    def get_instructions(
        self,
    ) -> "Callable[[PydanticRunContext[RunContext[object]]], Awaitable[str | None]]":
        async def render(ctx: PydanticRunContext[RunContext[object]]) -> "str | None":
            del ctx
            if not self.skills:
                return None
            lines = [
                "The following skills are available for this agent run.",
                "Use the `load_skill` tool to load the full instructions for a skill when it is relevant.",
            ]
            lines.extend(f"- {item.id}: Available skill {item.id}" for item in self.skills)
            return "\n".join(lines)

        return render

    def get_toolset(self) -> "FunctionToolset[RunContext[object]]":
        by_id = {item.id: item for item in self.skills}
        toolset = FunctionToolset[RunContext[object]](id=self.id)

        @toolset.tool
        async def list_skills(
            ctx: PydanticRunContext[RunContext[object]],
        ) -> "list[dict[str, str]]":
            """List skills available for this agent run."""
            del ctx
            return [
                {"id": item.id, "description": f"Available skill {item.id}"}
                for item in self.skills
            ]

        @toolset.tool
        async def load_skill(
            ctx: PydanticRunContext[RunContext[object]],
            skill_id: str,
        ) -> "dict[str, str]":
            """Load the full instructions for a selected skill."""
            del ctx
            specification = by_id.get(skill_id)
            if specification is None:
                raise ModelRetry("skill not found")
            return {"id": specification.id, "content": specification.content}

        return toolset

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


__all__ = ["SKILL_TOOL_NAMES", "SkillCapability"]
