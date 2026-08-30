#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI bridge for runtime observation middleware."""

from collections.abc import Sequence
from typing import Any, NoReturn

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext, ToolDefinition

from ..core import Principal
from ..observe import (
    Middleware,
    MiddlewarePipeline,
    RunContext as ObservedRunContext,
    context_for,
)


class _ObservationalMiddlewareCapability(AbstractCapability[None]):
    def __init__(
        self,
        pipeline: MiddlewarePipeline,
        context: ObservedRunContext,
    ) -> None:
        if not isinstance(pipeline, MiddlewarePipeline):
            raise TypeError("pipeline must be MiddlewarePipeline")
        if not isinstance(context, ObservedRunContext):
            raise TypeError("context must be observe.RunContext")
        self._pipeline = pipeline
        self._context = context

    async def before_run(self, ctx: RunContext[None]) -> None:
        del ctx
        await self._pipeline.before_run(self._context)

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        await self._pipeline.before_model(self._context)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del ctx, request_context
        await self._pipeline.after_model(self._context)
        return response

    async def before_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del ctx, call, tool_def
        await self._pipeline.before_tool(self._context)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        del ctx, call, tool_def, args
        await self._pipeline.after_tool(self._context)
        return result

    async def on_run_error(
        self,
        ctx: RunContext[None],
        *,
        error: BaseException,
    ) -> NoReturn:
        del ctx
        await self._pipeline.on_error(self._context, error)
        raise error

    async def after_run(
        self,
        ctx: RunContext[None],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        del ctx
        await self._pipeline.after_run(self._context)
        return result


def _build_middleware_pipeline(
    middleware: Sequence[Middleware],
) -> MiddlewarePipeline:
    return MiddlewarePipeline(middleware)


def _require_middleware_pipeline(value: object) -> MiddlewarePipeline:
    if not isinstance(value, MiddlewarePipeline):
        raise TypeError("middleware must be MiddlewarePipeline")
    return value


def _observational_middleware_capability(
    pipeline: MiddlewarePipeline,
    *,
    principal: Principal,
    execution_id: str,
    session_id: str | None,
    run_id: str,
    agent_id: str,
) -> _ObservationalMiddlewareCapability:
    return _ObservationalMiddlewareCapability(
        pipeline,
        context_for(principal, execution_id, session_id, run_id, agent_id),
    )
