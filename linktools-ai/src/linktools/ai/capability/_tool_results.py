#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host-defined result projection before Runtime tool settlement."""

from typing import Generic, TypeVar, TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability

from ._context import AgentContext

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pydantic_ai.tools import ToolDefinition

AppT = TypeVar("AppT")


class ToolResultsCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Project successful live tool results before their first durable receipt.

    Register with ``scope='agent'`` to bind the behavior by id and revision.
    Selected instances run in capability order. Completed-operation replay and
    final response fixtures do not run these projections again. Returning a
    reference requires an independently configured, authorized input loader.
    This hook does not invoke tools or change their external-effect policies.
    """

    async def process_tool_result(
        self,
        context: AgentContext[AppT],
        tool: "ToolDefinition",
        arguments: "Mapping[str, object]",
        result: object,
    ) -> object:
        """Return a projection, or preserve the original result unchanged."""
        return result


__all__ = ["ToolResultsCapability"]
