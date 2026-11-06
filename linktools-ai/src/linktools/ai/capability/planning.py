#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning tools borrowing the host-owned plan store."""

from collections.abc import Callable
from typing import cast

from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai_harness.planning import (
    PlanItem as HarnessPlanItem,
    Planning,
    PlanStore,
    TaskStatus,
)

from ._tool_signal import ToolCallRetry
from ._tool_metadata import tool_metadata

_PLANNING_CAPABILITY_ID = "linktools.ai.planning"
_PLANNING_TOOL_NAME = "write_plan"
_INVALID_PLAN = "The plan content is invalid. Correct the plan and retry."
_EMPTY_PLAN_CONTENT = (
    "Each plan item must have non-empty content. Fill every item and retry."
)
_DUPLICATE_PLAN_IDS = "Plan item ids must be unique. Remove duplicate ids and retry."
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


def _validate_model_plan(ctx: PydanticRunContext[None], items: object) -> None:
    """Reject unsupported plans before the native tool accesses its run store."""
    del ctx
    if not isinstance(items, list):
        raise ToolCallRetry(_INVALID_PLAN)
    identifiers: list[str] = []
    for item in items:
        if not isinstance(item, HarnessPlanItem):
            raise ToolCallRetry(_INVALID_PLAN)
        identifiers.append(item.id)
        if item.parent_id is not None or item.depends_on:
            raise ToolCallRetry(_FLAT_PLAN)
        if item.status not in _PLAIN_PLAN_STATUSES:
            raise ToolCallRetry(_FLAT_PLAN)
        if not isinstance(item.content, str) or not item.content.strip():
            raise ToolCallRetry(_EMPTY_PLAN_CONTENT)
    if len(identifiers) != len(set(identifiers)):
        raise ToolCallRetry(_DUPLICATE_PLAN_IDS)


class HarnessPlanning(Planning[None]):
    """Expose only the Runtime-supported Planning tool with its semantics."""

    @classmethod
    def bind(cls, store_resolver: Callable[[PydanticRunContext[None]], PlanStore]) -> "HarnessPlanning":
        """Bind the supported tool surface to an execution's native store resolver."""
        return cls(id=_PLANNING_CAPABILITY_ID, tools=(_PLANNING_TOOL_NAME,), store_resolver=store_resolver)

    def get_toolset(self) -> AbstractToolset[None] | None:
        toolset = super().get_toolset()
        if toolset is not None:
            tool = toolset.tools[_PLANNING_TOOL_NAME]
            tool.args_validator = _validate_model_plan
            tool.metadata = tool_metadata(
                base=tool.metadata,
                plan_safe=True,
                compaction_keep_result=True,
            )
            return cast(AbstractToolset[None], toolset)
        return None


__all__ = ["HarnessPlanning"]
