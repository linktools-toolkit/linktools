#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host-defined result projection after successful raw tool execution."""

from types import MappingProxyType
from typing import Generic, TypeVar, TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability

from ._context import AgentContext
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pydantic_ai.tools import ToolDefinition

AppT = TypeVar("AppT")


class ToolResultsCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Project successful live tool results through their execution owner.

    Register with ``scope='agent'`` to bind the behavior by id and revision.
    For native SDK execution, materialize the CapabilityContribution; mounting
    this callback alone does not install an execution boundary. Runtime invokes
    the same projection before the first durable receipt.
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

    @classmethod
    async def process_results(
        cls,
        capabilities: "Sequence[ToolResultsCapability[AppT]]",
        context: AgentContext[AppT],
        tool: "ToolDefinition",
        arguments: "Mapping[str, object]",
        result: object,
    ) -> object:
        """Project one raw success; projection failure cannot make it safe to retry."""
        try:
            frozen_arguments = MappingProxyType(dict(arguments))
            for capability in capabilities:
                result = await capability.process_tool_result(context, tool, frozen_arguments, result)
        except Exception as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        return result


__all__ = ["ToolResultsCapability"]
