#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness Planning adapter for the Runtime-owned plan store."""

from collections.abc import Callable
from typing import Any, cast

from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai_harness.planning import (
    PlanItem as HarnessPlanItem,
    Planning,
    TaskStatus,
)

from ..capability import ToolCallRejected, tool_semantic_metadata
from ..errors import AIError, ErrorCode
from ._harness import HarnessPlanStoreAdapter

_PLANNING_CAPABILITY_ID = "linktools.ai.planning"
_PLANNING_TOOL_NAME = "write_plan"
_INVALID_PLAN = "The plan content is invalid. Correct the plan and retry."
_FLAT_PLAN = (
    "Subtasks and dependencies are not enabled. Use a flat plan and retry."
)
_PLAIN_PLAN_STATUSES = frozenset(
    {
        TaskStatus.pending,
        TaskStatus.in_progress,
        TaskStatus.completed,
        TaskStatus.cancelled,
    }
)


class _PlanningToolset(AbstractToolset[None]):
    """Raise LinkTools signals for invalid plans before Harness writes them."""

    def __init__(self, wrapped: AbstractToolset[None]) -> None:
        self._wrapped = wrapped
        self._raw_tools: dict[str, ToolsetTool[None]] = {}

    @property
    def id(self) -> str | None:
        return self._wrapped.id

    async def __aenter__(self) -> "_PlanningToolset":
        self._wrapped = await self._wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self._wrapped.__aexit__(*args)

    async def get_tools(
        self,
        ctx: PydanticRunContext[None],
    ) -> dict[str, ToolsetTool[None]]:
        raw_tools = await self._wrapped.get_tools(ctx)
        result: dict[str, ToolsetTool[None]] = {}
        self._raw_tools = {}
        for name, raw_tool in raw_tools.items():
            result[name] = ToolsetTool(
                toolset=self,
                tool_def=raw_tool.tool_def,
                max_retries=raw_tool.max_retries,
                args_validator=raw_tool.args_validator,
                args_validator_func=raw_tool.args_validator_func,
            )
            self._raw_tools[name] = raw_tool
        return result

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: PydanticRunContext[None],
        tool: ToolsetTool[None],
    ) -> Any:
        raw_tool = self._raw_tools.get(name)
        if raw_tool is None or tool.toolset is not self:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if name == _PLANNING_TOOL_NAME:
            _validate_model_plan(tool_args.get("items"))
        return await self._wrapped.call_tool(name, tool_args, ctx, raw_tool)


def _validate_model_plan(items: object) -> None:
    if not isinstance(items, list):
        raise ToolCallRejected(_INVALID_PLAN)
    identifiers: list[str] = []
    for item in items:
        if not isinstance(item, HarnessPlanItem):
            raise ToolCallRejected(_INVALID_PLAN)
        identifiers.append(item.id)
        if item.parent_id is not None or item.depends_on:
            raise ToolCallRejected(_FLAT_PLAN)
        if item.status not in _PLAIN_PLAN_STATUSES:
            raise ToolCallRejected(_FLAT_PLAN)
        if not isinstance(item.content, str) or not item.content.strip():
            raise ToolCallRejected(_INVALID_PLAN)
    if len(identifiers) != len(set(identifiers)):
        raise ToolCallRejected(_INVALID_PLAN)


class HarnessPlanning(Planning[None]):
    """Expose only the Runtime-supported Planning tool with its semantics."""

    def get_toolset(self) -> AbstractToolset[None] | None:
        toolset = super().get_toolset()
        if toolset is not None:
            tool = toolset.tools[_PLANNING_TOOL_NAME]
            tool.metadata = tool_semantic_metadata(
                base=tool.metadata,
                plan_safe=True,
                compaction_keep_result=True,
            )
            return _PlanningToolset(cast(AbstractToolset[None], toolset))
        return None


def build_harness_planning(
    store_resolver: Callable[
        [PydanticRunContext[None]],
        HarnessPlanStoreAdapter,
    ],
) -> HarnessPlanning:
    """Build the Runtime-owned Planning capability."""
    return HarnessPlanning(
        id=_PLANNING_CAPABILITY_ID,
        tools=(_PLANNING_TOOL_NAME,),
        store_resolver=store_resolver,
    )


__all__ = ["HarnessPlanning", "build_harness_planning"]
