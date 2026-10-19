#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native recovery and trace capabilities with isolated per-run bindings."""

from __future__ import annotations

import asyncio
from time import monotonic_ns
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_ai import CallToolsNode, ModelRequestNode
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..capability import AgentContext
from ..errors import AIError, ErrorCode
from ._journal import (
    DURATION_NS_METADATA_KEY,
    MODEL_REQUEST_SEQ_METADATA_KEY,
    await_request_handoff,
)
from .state._step_contracts import TOOL_ERROR_CODE_METADATA_KEY

if TYPE_CHECKING:
    from ._run_hooks import AgentRunCapture

AppT = TypeVar("AppT")


class CheckpointService(AbstractCapability[AgentContext[AppT]]):
    """Confirm recovery context at model and tool safety boundaries.

    Register at Runtime scope. Runtime owns checkpoint transactions and producer
    fencing; each run binds its own capture without modifying the declaration.
    """

    def __init__(self, capture: "AgentRunCapture") -> None:
        self._capture = capture

    id: str | None = "runtime.checkpoint"
    _capture: AgentRunCapture | None = None

    def _bound_capture(self) -> AgentRunCapture:
        if self._capture is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._capture

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_model_request(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        capture = self._bound_capture()
        await capture.context.save_checkpoint(
            ctx, messages=ctx.messages, state="complete"
        )
        return request_context

    async def after_node_run(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        node: AgentNode[AgentContext[AppT]],
        result: NodeResult[AgentContext[AppT]],
    ) -> NodeResult[AgentContext[AppT]]:
        if isinstance(node, ModelRequestNode):
            response = None
            if isinstance(result, CallToolsNode):
                response = result.model_response
            elif ctx.messages and isinstance(ctx.messages[-1], ModelResponse):
                response = ctx.messages[-1]
            if response is not None:
                # The recorder captured this response with its request identity.
                # It must be recoverable before any tool effect.
                await self._bound_capture().context.save_checkpoint(
                    ctx, messages=ctx.messages, state="complete"
                )
        if isinstance(node, CallToolsNode):
            pending = result.request if isinstance(result, ModelRequestNode) else None
            await self._bound_capture().context.save_checkpoint(
                ctx,
                messages=ctx.messages,
                pending=pending,
                state="complete",
            )
        return result

    async def after_run(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        capture = self._bound_capture()
        await capture.after_run(ctx, result=result)
        await capture.context.save_checkpoint(
            ctx,
            messages=result.all_messages(),
            state="interrupted"
            if isinstance(result.output, DeferredToolRequests)
            else "complete",
        )
        return result

    async def on_run_error(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        capture = self._bound_capture()
        capture.recorder.finish_transcript(interrupted=True)
        await capture.context.save_checkpoint(
            ctx, messages=capture.live_messages or ctx.messages, state="interrupted"
        )
        raise error


class TraceService(AbstractCapability[AgentContext[AppT]]):
    """Retain model, transcript and tool evidence without enabling recovery.

    Native hooks collect this run's observations. Existing history boundaries
    retain their batching behavior; Runtime seals the final execution tail.
    """

    def __init__(self, capture: "AgentRunCapture") -> None:
        self._capture = capture
        self._tool_started_ns = {}

    id: str | None = "runtime.trace"
    _capture: AgentRunCapture | None = None
    _tool_started_ns: dict[str, int]

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
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        capture = self._bound_capture()
        await capture.after_run(ctx, result=result)
        await capture.context.save_checkpoint(
            ctx,
            messages=result.all_messages(),
            state="interrupted"
            if isinstance(result.output, DeferredToolRequests)
            else "complete",
        )
        await capture.recorder.record_event(
            "AGENT_RUN_INTERRUPTED"
            if isinstance(result.output, DeferredToolRequests)
            else "AGENT_RUN_SUCCEEDED",
            ctx.run_step,
        )
        return result

    async def on_run_error(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        capture = self._bound_capture()
        capture.recorder.finish_transcript(interrupted=True)
        await capture.context.save_checkpoint(
            ctx, messages=capture.live_messages or ctx.messages, state="interrupted"
        )
        await capture.recorder.record_event(
            "AGENT_RUN_FAILED", ctx.run_step, error=repr(error)
        )
        raise error

    def _tool_request_metadata(self, tool_call_id: str) -> dict[str, str]:
        sequence = self._bound_capture().recorder.model_request_seq_for_tool_call(
            tool_call_id
        )
        return (
            {} if sequence is None else {MODEL_REQUEST_SEQ_METADATA_KEY: str(sequence)}
        )

    async def before_tool_execute(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        self._bound_capture()
        if call.tool_call_id in self._tool_started_ns:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._tool_started_ns[call.tool_call_id] = monotonic_ns()
        await self._bound_capture().recorder.record_tool_start(call, ctx.run_step)
        return args

    async def after_tool_execute(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        del args
        self._bound_capture()
        started_ns = self._tool_started_ns.pop(call.tool_call_id, None)
        if started_ns is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        metadata = self._tool_request_metadata(call.tool_call_id)
        metadata[DURATION_NS_METADATA_KEY] = str(max(0, monotonic_ns() - started_ns))
        self._bound_capture().recorder.remember_tool_execution(
            call.tool_call_id,
            metadata=metadata,
        )
        return result

    async def on_tool_execute_error(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        del args
        self._bound_capture()
        started_ns = self._tool_started_ns.pop(call.tool_call_id, None)
        if started_ns is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        metadata = self._tool_request_metadata(call.tool_call_id)
        metadata[DURATION_NS_METADATA_KEY] = str(max(0, monotonic_ns() - started_ns))
        if isinstance(error, AIError):
            metadata[TOOL_ERROR_CODE_METADATA_KEY] = error.code.value

        async def record_failure() -> None:
            await self._bound_capture().recorder.record_event(
                "TOOL_CALL_FAILED",
                ctx.run_step,
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                error=repr(error),
                metadata=metadata,
            )
            await self._bound_capture().recorder.commit_history_boundary()

        interrupted = await await_request_handoff(record_failure())
        if interrupted:
            raise asyncio.CancelledError from error
        raise error


__all__ = ["CheckpointService", "TraceService"]
