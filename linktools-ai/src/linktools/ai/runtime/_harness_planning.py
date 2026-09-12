#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness Planning adapter for the Runtime-owned plan store."""

from collections.abc import Callable

from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai_harness.planning import Planning

from ..capability import tool_semantic_metadata
from ._harness import HarnessPlanStoreAdapter

_PLANNING_CAPABILITY_ID = "linktools.ai.planning"
_PLANNING_TOOL_NAME = "write_plan"


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
        return toolset

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
