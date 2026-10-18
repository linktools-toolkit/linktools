#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public persistence capabilities with isolated Runtime service bindings."""

from __future__ import annotations

import math
from copy import copy
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.tools import ToolDefinition

from ..core import DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL
from ..errors import AIError, ErrorCode
from ._context import AgentContext

if TYPE_CHECKING:
    from ..runtime import RuntimeStorageRoute

AppT = TypeVar("AppT")


class ConversationCapability(AbstractCapability[AgentContext[AppT]]):
    """Retain actual context for continuation and execution forks.

    Runtime confirms the candidate projection with the terminal execution.
    A complete configured cache may support a pending turn; the core owns
    admission eligibility, effects, fencing and durable reconciliation."""

    id = "runtime.conversation"
    _service: AbstractCapability[AgentContext[AppT]] | None = None

    def __init__(
        self, *, storage: RuntimeStorageRoute | None = None, cache: object | None = None
    ) -> None:
        self._storage = storage
        self._cache = cache

    @property
    def storage(self) -> RuntimeStorageRoute | None:
        return self._storage

    @property
    def cache(self) -> object | None:
        return self._cache

    def bind(
        self, service: AbstractCapability[AgentContext[AppT]]
    ) -> ConversationCapability[AppT]:
        bound = copy(self)
        bound._service = service
        return bound

    def _bound_service(self) -> AbstractCapability[AgentContext[AppT]]:
        if self._service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._service

    async def after_run(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        return await self._bound_service().after_run(ctx, result=result)

    async def on_run_error(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, error: BaseException
    ) -> AgentRunResult[Any]:
        return await self._bound_service().on_run_error(ctx, error=error)


class CheckpointCapability(AbstractCapability[AgentContext[AppT]]):
    """Confirm recovery context at model and tool safety boundaries.

    Register at Runtime scope. Runtime owns checkpoint transactions and producer
    fencing; each run binds its own capture without modifying the declaration."""

    id = "runtime.checkpoint"
    _service: AbstractCapability[AgentContext[AppT]] | None = None

    def __init__(self, *, storage: RuntimeStorageRoute | None = None) -> None:
        self._storage = storage

    @property
    def storage(self) -> RuntimeStorageRoute | None:
        return self._storage

    def bind(
        self, service: AbstractCapability[AgentContext[AppT]]
    ) -> CheckpointCapability[AppT]:
        bound = copy(self)
        bound._service = service
        return bound

    def _bound_service(self) -> AbstractCapability[AgentContext[AppT]]:
        if self._service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._service

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_model_request(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await self._bound_service().before_model_request(ctx, request_context)

    async def after_node_run(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        node: AgentNode[AgentContext[AppT]],
        result: NodeResult[AgentContext[AppT]],
    ) -> NodeResult[AgentContext[AppT]]:
        return await self._bound_service().after_node_run(ctx, node=node, result=result)

    async def after_run(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        return await self._bound_service().after_run(ctx, result=result)

    async def on_run_error(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, error: BaseException
    ) -> AgentRunResult[Any]:
        return await self._bound_service().on_run_error(ctx, error=error)


class TraceCapability(AbstractCapability[AgentContext[AppT]]):
    """Retain model, transcript and tool evidence without enabling recovery.

    A write-capable cache owns optional observation batching. Without one,
    observation boundaries await publication. Runtime seals the final tail.
    """

    id = "runtime.trace"
    _service: AbstractCapability[AgentContext[AppT]] | None = None

    def __init__(
        self, *, storage: RuntimeStorageRoute | None = None, cache: object | None = None,
        batch_size: int = DEFAULT_TRACE_BATCH_SIZE,
        flush_interval: float = DEFAULT_TRACE_FLUSH_INTERVAL,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (isinstance(flush_interval, bool) or not isinstance(flush_interval, (int, float))
                or not math.isfinite(flush_interval) or flush_interval <= 0):
            raise ValueError("flush_interval must be finite and positive")
        methods = tuple(getattr(cache, name, None) for name in ("append", "read_batch", "ack"))
        if any(method is not None for method in methods) and not all(callable(method) for method in methods):
            raise TypeError("cache must implement append, read_batch and ack together")
        self._batch_size = batch_size
        self._flush_interval = float(flush_interval)
        self._storage = storage
        self._cache = cache

    @property
    def storage(self) -> RuntimeStorageRoute | None:
        return self._storage

    @property
    def cache(self) -> object | None:
        return self._cache

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def flush_interval(self) -> float:
        return self._flush_interval

    def bind(
        self, service: AbstractCapability[AgentContext[AppT]]
    ) -> TraceCapability[AppT]:
        bound = copy(self)
        bound._service = service
        return bound

    def _bound_service(self) -> AbstractCapability[AgentContext[AppT]]:
        if self._service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._service

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_run(self, ctx: PydanticRunContext[AgentContext[AppT]]) -> None:
        return await self._bound_service().before_run(ctx)

    async def after_run(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        return await self._bound_service().after_run(ctx, result=result)

    async def on_run_error(
        self, ctx: PydanticRunContext[AgentContext[AppT]], *, error: BaseException
    ) -> AgentRunResult[Any]:
        return await self._bound_service().on_run_error(ctx, error=error)

    async def before_tool_execute(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._bound_service().before_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args
        )

    async def after_tool_execute(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        return await self._bound_service().after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result=result
        )

    async def on_tool_execute_error(
        self,
        ctx: PydanticRunContext[AgentContext[AppT]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        return await self._bound_service().on_tool_execute_error(
            ctx, call=call, tool_def=tool_def, args=args, error=error
        )


__all__ = ["ConversationCapability", "CheckpointCapability", "TraceCapability"]
