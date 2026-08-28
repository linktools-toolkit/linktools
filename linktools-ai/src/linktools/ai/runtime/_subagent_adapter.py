#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI adapter for the vendor-neutral Subagent capability."""

from collections.abc import Awaitable, Callable

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import FunctionToolset

from ..capability import RunContext, SubagentCapability
from ..core import JsonValue
from ..errors import AIError, ErrorCode

_SUBAGENT_CAPABILITY_ID = "linktools-subagent"
_MODEL_CORRECTABLE_ERRORS = frozenset(
    {
        ErrorCode.CAPABILITY_RESOLUTION_INVALID,
        ErrorCode.REQUEST_FIELD_INVALID,
    }
)


class _PydanticSubagentCapability(AbstractCapability[RunContext[object]]):
    def __init__(self, capability: SubagentCapability) -> None:
        if not isinstance(capability, SubagentCapability):
            raise TypeError("capability must be SubagentCapability")
        self.id = _SUBAGENT_CAPABILITY_ID
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
        async def list_subagents(
            ctx: PydanticRunContext[RunContext[object]],
        ) -> "list[dict[str, str]]":
            """List subagents available for this agent run."""
            del ctx
            return await self._capability.list_subagents()

        @toolset.tool
        async def delegate_task(
            ctx: PydanticRunContext[RunContext[object]],
            subagent_id: str,
            task: str,
        ) -> "dict[str, JsonValue]":
            """Delegate one task to a selected subagent."""
            if not ctx.tool_call_id:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            try:
                return await self._capability.delegate_task(
                    subagent_id,
                    task,
                    invocation_id=ctx.tool_call_id,
                )
            except AIError as error:
                if error.code not in _MODEL_CORRECTABLE_ERRORS:
                    raise
                raise ModelRetry("requested subagent or task is invalid") from error

        return toolset

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


__all__ = ["_PydanticSubagentCapability"]
