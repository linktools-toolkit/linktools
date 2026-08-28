#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI adapter for the vendor-neutral Skill capability."""

from collections.abc import Awaitable, Callable
from typing import cast

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import FunctionToolset

from ..capability import RunContext, SkillCapability
from ..errors import AIError, ErrorCode

_SKILL_CAPABILITY_ID = "linktools-skill"
_MODEL_CORRECTABLE_ERRORS = frozenset(
    {
        ErrorCode.CAPABILITY_RESOLUTION_INVALID,
        ErrorCode.REQUEST_FIELD_INVALID,
        ErrorCode.ASSET_PATH_OUTSIDE_ROOT,
        ErrorCode.ASSET_NOT_FOUND,
        ErrorCode.ASSET_CODEC_UNKNOWN,
    }
)


class _PydanticSkillCapability(AbstractCapability[RunContext[object]]):
    def __init__(self, capability: SkillCapability) -> None:
        if not isinstance(capability, SkillCapability):
            raise TypeError("capability must be SkillCapability")
        self.id = _SKILL_CAPABILITY_ID
        self._capability = capability

    def get_instructions(
        self,
    ) -> "Callable[[PydanticRunContext[RunContext[object]]], Awaitable[str | None]]":
        async def render(ctx: PydanticRunContext[RunContext[object]]) -> "str | None":
            del ctx
            return self._capability.instructions()

        return render

    def get_toolset(self) -> "FunctionToolset[RunContext[object]]":
        toolset = FunctionToolset[RunContext[object]](id=self.id)

        @toolset.tool
        async def list_skills(
            ctx: PydanticRunContext[RunContext[object]],
        ) -> "list[dict[str, str]]":
            """List skills available for this agent run."""
            del ctx
            return await self._capability.list_skills()

        @toolset.tool
        async def load_skill(
            ctx: PydanticRunContext[RunContext[object]],
            skill_id: str,
            path: "str | None" = None,
        ) -> "dict[str, str | list[str]]":
            """Load skill instructions or one relative text resource."""
            del ctx
            try:
                return cast(
                    dict[str, str | list[str]],
                    await self._capability.load_skill(skill_id, path),
                )
            except AIError as error:
                if error.code not in _MODEL_CORRECTABLE_ERRORS:
                    raise
                raise ModelRetry(_skill_retry_message(error.code)) from error

        return toolset

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


def _skill_retry_message(code: ErrorCode) -> str:
    if code is ErrorCode.CAPABILITY_RESOLUTION_INVALID:
        return "skill not found"
    if code is ErrorCode.ASSET_NOT_FOUND:
        return "skill resource not found"
    if code is ErrorCode.ASSET_CODEC_UNKNOWN:
        return "skill resource is not UTF-8 text"
    return "skill resource path is invalid"


__all__ = ["_PydanticSkillCapability"]
