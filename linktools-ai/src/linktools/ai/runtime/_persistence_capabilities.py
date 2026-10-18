#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native recovery and trace capabilities with isolated per-run bindings."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_ai.capabilities import AbstractCapability, AgentNode, CapabilityOrdering, NodeResult
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..capability import AgentContext
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from ._run_hooks import AgentRunCapture

AppT = TypeVar("AppT")


class CheckpointCapability(AbstractCapability[AgentContext[AppT]]):
    """Confirm recovery context at model and tool safety boundaries.

    Register at Runtime scope. Runtime owns checkpoint transactions and producer
    fencing; each run binds its own capture without modifying the declaration.
    """

    id: str | None = "runtime.checkpoint"
    _capture: AgentRunCapture | None = None

    def bind(self, capture: AgentRunCapture) -> CheckpointCapability[AppT]:
        """Retain this capability's native type and configuration for one run."""
        bound = copy(self)
        bound._capture = capture
        return bound

    def _bound_capture(self) -> AgentRunCapture:
        if self._capture is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._capture

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_model_request(
        self, ctx: PydanticRunContext[AgentContext[AppT]], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        capture = self._bound_capture()
        await capture.before_model_request(ctx, request_context)
        await capture.save_checkpoint(ctx, messages=ctx.messages, state="complete")
        return request_context

    async def after_node_run(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *,
        node: AgentNode[AgentContext[AppT]], result: NodeResult[AgentContext[AppT]],
    ) -> NodeResult[AgentContext[AppT]]:
        return await self._bound_capture().checkpoint_after_node(ctx, node=node, result=result)

    async def after_run(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        await self._bound_capture().capture_final_context(ctx, result=result)
        return result

    async def on_run_error(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, error: BaseException,
    ) -> AgentRunResult[Any]:
        await self._bound_capture().capture_interrupted_context(ctx)
        raise error


class TraceCapability(AbstractCapability[AgentContext[AppT]]):
    """Retain model, transcript and tool evidence without enabling recovery.

    Native hooks collect this run's observations. Existing history boundaries
    retain their batching behavior; Runtime seals the final execution tail.
    """

    id: str | None = "runtime.trace"
    _capture: AgentRunCapture | None = None

    def bind(self, capture: AgentRunCapture) -> TraceCapability[AppT]:
        """Retain this capability's native type and configuration for one run."""
        bound = copy(self)
        bound._capture = capture
        return bound

    def _bound_capture(self) -> AgentRunCapture:
        if self._capture is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._capture

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_run(self, ctx: PydanticRunContext[AgentContext[AppT]]) -> None:
        capture = self._bound_capture()
        await capture.before_run(ctx)
        await capture.recorder.record_event("AGENT_RUN_STARTED", ctx.run_step)

    async def after_run(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        capture = self._bound_capture()
        await capture.capture_final_context(ctx, result=result)
        await capture.recorder.record_event(
            "AGENT_RUN_INTERRUPTED" if isinstance(result.output, DeferredToolRequests)
            else "AGENT_RUN_SUCCEEDED", ctx.run_step,
        )
        return result

    async def on_run_error(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, error: BaseException,
    ) -> AgentRunResult[Any]:
        capture = self._bound_capture()
        await capture.capture_interrupted_context(ctx)
        await capture.recorder.record_event("AGENT_RUN_FAILED", ctx.run_step, error=repr(error))
        raise error

    async def before_tool_execute(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, call: ToolCallPart,
        tool_def: ToolDefinition, args: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._bound_capture().record_tool_start(ctx, call=call, tool_def=tool_def, args=args)

    async def after_tool_execute(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, call: ToolCallPart,
        tool_def: ToolDefinition, args: dict[str, Any], result: Any,
    ) -> Any:
        return await self._bound_capture().record_tool_result(
            ctx, call=call, tool_def=tool_def, args=args, result=result,
        )

    async def on_tool_execute_error(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, call: ToolCallPart,
        tool_def: ToolDefinition, args: dict[str, Any], error: Exception,
    ) -> Any:
        return await self._bound_capture().record_tool_error(
            ctx, call=call, tool_def=tool_def, args=args, error=error,
        )


__all__ = ["CheckpointCapability", "TraceCapability"]
