#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiddlewareCapability: adapts a MiddlewarePipeline into a real pydantic-ai
AbstractCapability so the four model/tool hooks fire during agent.run().

current_context is a MUTABLE field (per-Run injection, same pattern/caveat as
tool/capability.py's PolicyCapability: AgentRunner sets it before each
agent.run() and clears it after; concurrent Runs sharing one CompiledAgent
would race on it -- known, out-of-scope this phase).

Middlewares observe tool calls this phase; they do not mutate args
(before_tool_execute returns `args` unchanged)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability

from ..policy.engine import ToolContext, ToolRequest
from .pipeline import MiddlewarePipeline

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import ToolDefinition


@dataclass
class MiddlewareCapability(AbstractCapability[None]):
    pipeline: MiddlewarePipeline
    current_context: "ToolContext | None" = None

    def _context_or_default(self) -> ToolContext:
        return self.current_context or ToolContext(run_id="unknown", session_id="unknown")

    async def before_model_request(self, ctx: "RunContext[Any]", request_context: "ModelRequestContext") -> "ModelRequestContext":
        return await self.pipeline.run_before_model(self._context_or_default(), request_context)

    async def after_model_request(self, ctx: "RunContext[Any]", *, request_context: "ModelRequestContext", response: "ModelResponse") -> "ModelResponse":
        return await self.pipeline.run_after_model(self._context_or_default(), response)

    async def before_tool_execute(self, ctx: "RunContext[Any]", *, call: "ToolCallPart", tool_def: "ToolDefinition", args: Any) -> Any:
        request = ToolRequest(tool_name=tool_def.name, arguments=args if isinstance(args, dict) else {})
        await self.pipeline.run_before_tool(self._context_or_default(), request)
        return args

    async def after_tool_execute(self, ctx: "RunContext[Any]", *, call: "ToolCallPart", tool_def: "ToolDefinition", args: Any, result: Any) -> Any:
        request = ToolRequest(tool_name=tool_def.name, arguments=args if isinstance(args, dict) else {})
        return await self.pipeline.run_after_tool(self._context_or_default(), request, result)

    async def on_tool_execute_error(self, ctx: "RunContext[Any]", *, call: "ToolCallPart", tool_def: "ToolDefinition", args: Any, error: Exception) -> None:
        await self.pipeline.run_on_error(self._context_or_default(), error)
        raise error


def build_middleware_capability(pipeline: MiddlewarePipeline) -> MiddlewareCapability:
    return MiddlewareCapability(pipeline=pipeline)
