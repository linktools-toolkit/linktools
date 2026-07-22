#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiddlewareCapability: adapts a MiddlewarePipeline into a real pydantic-ai
AbstractCapability so the four model/tool hooks fire during agent.run().

The per-Run ToolContext arrives via pydantic-ai dependency injection (same
pattern as tool/capability.py's PolicyCapability): the runner passes
``deps=AgentDependencies(tool_context=...)`` to ``agent.pydantic_agent.run()`` /
``.iter()`` and each hook reads it off ``ctx.deps.tool_context``. No mutable
per-Run field, so a single capability instance is safe to reuse across many
concurrent Runs sharing one CompiledAgent.

Middlewares observe tool calls; they do not mutate args
(before_tool_execute returns `args` unchanged)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability

from ..governance.policy.engine import ToolRequest
from .pipeline import MiddlewarePipeline

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import ToolDefinition


@dataclass
class MiddlewareCapability(AbstractCapability[None]):
    pipeline: MiddlewarePipeline

    async def before_model_request(
        self, ctx: "RunContext[Any]", request_context: "ModelRequestContext"
    ) -> "ModelRequestContext":
        return await self.pipeline.run_before_model(
            ctx.deps.tool_context, request_context
        )

    async def after_model_request(
        self,
        ctx: "RunContext[Any]",
        *,
        request_context: "ModelRequestContext",
        response: "ModelResponse",
    ) -> "ModelResponse":
        return await self.pipeline.run_after_model(ctx.deps.tool_context, response)

    async def before_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
    ) -> Any:
        request = ToolRequest(
            tool_name=tool_def.name, arguments=args if isinstance(args, dict) else {}
        )
        await self.pipeline.run_before_tool(ctx.deps.tool_context, request)
        return args

    async def after_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
        result: Any,
    ) -> Any:
        request = ToolRequest(
            tool_name=tool_def.name, arguments=args if isinstance(args, dict) else {}
        )
        return await self.pipeline.run_after_tool(
            ctx.deps.tool_context, request, result
        )

    async def on_tool_execute_error(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
        error: Exception,
    ) -> None:
        await self.pipeline.run_on_error(ctx.deps.tool_context, error)
        raise error


def build_middleware_capability(pipeline: MiddlewarePipeline) -> MiddlewareCapability:
    return MiddlewareCapability(pipeline=pipeline)
