#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Own the final conversion of LinkTools tool signals to Pydantic controls."""

from linktools.core import environ
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    RawToolArgs,
    ValidatedToolArgs,
    WrapToolValidateHandler,
)
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.tools import ToolDefinition

from ._context import AgentContext
from ._tool_signal import ToolCallFailed, ToolCallRetry

_OWNS_PYDANTIC_TOOL_CONTROL = True
_logger = environ.get_logger("ai.capability.pydantic_tool_control")


class PydanticToolControlCapability(AbstractCapability[AgentContext[object]]):
    """Convert LinkTools tool feedback at the native SDK's outer boundary.

    Add once when composing a standalone Pydantic Agent with LinkTools tools.
    Runtime installs this capability itself. It owns no resources and neither
    authorizes tool calls nor changes their external-effect or retry policies.
    Unrelated exceptions and cancellation retain their original semantics.
    """

    def __init__(self) -> None:
        self.id = "linktools.ai.pydantic-tool-control"

    @classmethod
    def build_model_retry(cls, message: str) -> ModelRetry:
        """Construct SDK retry feedback, including for deferred recovery."""
        return ModelRetry(message)

    @classmethod
    def build_tool_failed(cls, message: str) -> ToolFailed:
        """Construct SDK failed-call feedback, including for deferred recovery."""
        return ToolFailed(message)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    async def wrap_tool_validate(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        handler: WrapToolValidateHandler,
    ) -> ValidatedToolArgs:
        del ctx, call, tool_def
        try:
            return await handler(args)
        except ToolCallRetry as error:
            raise self.build_model_retry(error.message) from error
        except ToolCallFailed as error:
            raise self.build_tool_failed(error.message) from error

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
            raise self.build_model_retry(error.message) from error
        if isinstance(error, ToolCallFailed):
            _logger.debug(
                "converting failed tool call: tool=%s call=%s",
                tool_def.name,
                call.tool_call_id,
            )
            raise self.build_tool_failed(error.message) from error
        raise error


__all__ = ["PydanticToolControlCapability"]
