#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI adapter for the vendor-neutral Subagent capability."""

import json
from collections.abc import Awaitable, Callable
from typing import cast

from pydantic import JsonValue as PydanticJsonValue
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import FunctionToolset

from ..capability import RunContext, SubagentCapability
from ..errors import AIError, ErrorCode

_SUBAGENT_CAPABILITY_ID = "linktools-subagent"
_MODEL_RETRY_ERRORS = frozenset(
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
        ) -> "dict[str, PydanticJsonValue]":
            """Delegate one task to a selected subagent."""
            if not ctx.tool_call_id:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            try:
                return cast(
                    dict[str, PydanticJsonValue],
                    await self._capability.delegate_task(
                        subagent_id,
                        task,
                        invocation_id=ctx.tool_call_id,
                    ),
                )
            except AIError as error:
                if error.code in _MODEL_RETRY_ERRORS:
                    raise ModelRetry(_subagent_retry_message(error)) from error
                if error.code is ErrorCode.TOOL_EXECUTION_FAILED:
                    raise ToolFailed(_subagent_failure_message(error)) from error
                raise

        return toolset

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


def _subagent_retry_message(error: AIError) -> str:
    if error.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID:
        subagent_id = error.safe_details.get("subagent_id")
        target = (
            f"subagent {subagent_id!r}"
            if isinstance(subagent_id, str) and subagent_id
            else "requested subagent"
        )
        return f"{target} is not available; call list_subagents and choose an available subagent"
    return "requested subagent or task is invalid"


def _subagent_failure_message(error: AIError) -> str:
    return "subagent execution failed: " + json.dumps(
        dict(error.safe_details),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = ["_PydanticSubagentCapability"]
