#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Own the final conversion of LinkTools tool signals to Pydantic controls."""

from linktools.core import environ
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
)
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.tools import ToolDefinition

from ..capability import AgentContext, ToolCallFailed, ToolCallRetry

_OWNS_PYDANTIC_TOOL_CONTROL = True
_logger = environ.get_logger("ai.runtime.pydantic_tool_control")


def build_model_retry(message: str) -> ModelRetry:
    """Construct the Pydantic retry control for one LinkTools retry signal."""
    return ModelRetry(message)


def build_tool_failed(message: str) -> ToolFailed:
    """Construct the Pydantic failure control for one LinkTools failure."""
    return ToolFailed(message)


class PydanticToolControlCapability(AbstractCapability[AgentContext[object]]):
    """Convert only LinkTools model-facing signals at the outer boundary."""

    def __init__(self) -> None:
        self.id = "linktools.ai.pydantic-tool-control"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    async def on_tool_execute_error(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> None:
        del ctx, args
        if isinstance(error, ToolCallRetry):
            _logger.debug(
                "converting tool call retry: tool=%s call=%s",
                tool_def.name,
                call.tool_call_id,
            )
            raise build_model_retry(error.message) from error
        if isinstance(error, ToolCallFailed):
            _logger.debug(
                "converting failed tool call: tool=%s call=%s",
                tool_def.name,
                call.tool_call_id,
            )
            raise build_tool_failed(error.message) from error
        raise error


__all__ = [
    "PydanticToolControlCapability",
    "build_model_retry",
    "build_tool_failed",
]
