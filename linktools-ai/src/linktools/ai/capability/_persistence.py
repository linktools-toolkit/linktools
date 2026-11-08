#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public persistence capabilities with isolated Runtime service bindings."""

from __future__ import annotations

from copy import copy
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.tools import ToolDefinition

from ..core import CommitPolicy, DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL
from ..errors import AIError, ErrorCode
from ._context import AgentContext


AppT = TypeVar("AppT")


class TraceCapability(AbstractCapability[AgentContext[AppT]]):
    """Retain model, transcript and tool evidence without enabling recovery.

    Omitted batch timing resolves to 128 entries and 3 seconds at construction.
    A borrowed write buffer supports optional batch publication. Without one,
    batch observation boundaries await publication. Terminal mode retains observations
    until Runtime seals the final tail or a required safety boundary publishes them.

    Physical routes and caches belong to PersistenceStorage. Trace routes must
    share the execution StateStorageGroup. Storage caches do not supply the
    independent ``commit.buffer`` policy.
    """

    id = "runtime.trace"
    _service: AbstractCapability[AgentContext[AppT]] | None = None
    _flush: Callable[[], Awaitable[None]] | None = None
    _model_request_seq: Callable[[str], int | None] | None = None

    def __init__(
        self, *,
        commit: CommitPolicy = CommitPolicy.batch(),
        record_http_bodies: bool = False,
    ) -> None:
        if not isinstance(record_http_bodies, bool):
            raise TypeError("record_http_bodies must be bool")
        self._record_http_bodies = record_http_bodies
        if not isinstance(commit, CommitPolicy) or commit.mode not in {"sync", "batch", "terminal"}:
            raise ValueError("Trace requires sync, batch or terminal commit")
        self._commit = commit.resolve_batch(
            batch_size=DEFAULT_TRACE_BATCH_SIZE, flush_interval=DEFAULT_TRACE_FLUSH_INTERVAL,
        )

    @property
    def record_http_bodies(self) -> bool:
        """Whether supported HTTP attempts retain their unredacted wire bodies."""
        return self._record_http_bodies

    @property
    def commit(self) -> CommitPolicy:
        return self._commit

    def bind(
        self, service: AbstractCapability[AgentContext[AppT]], *,
        flush: Callable[[], Awaitable[None]] | None = None,
        model_request_seq: Callable[[str], int | None] | None = None,
    ) -> TraceCapability[AppT]:
        bound = copy(self)
        bound._service = service
        bound._flush = flush
        bound._model_request_seq = model_request_seq
        return bound

    @classmethod
    def for_context(cls, ctx: PydanticRunContext[AgentContext[AppT]]) -> TraceCapability[AppT]:
        """Get this run's bound Trace feature from the native SDK registry."""
        capability = ctx.capabilities.get(cls.id)
        if not isinstance(capability, cls):
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
        return capability

    async def flush(self) -> None:
        """Publish this run's captured Trace frontier to its source archive.

        Success is a source-confirmed visibility barrier, not buffer admission.
        Future observations keep the configured batching policy. This does not
        capture hooks that have not run, commit business state, or drain other runs.
        """
        if self._flush is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._flush()

    def model_request_seq_for_tool_call(self, tool_call_id: str) -> int | None:
        """Return the captured originating request, including restored calls."""
        return None if self._model_request_seq is None else self._model_request_seq(tool_call_id)

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


__all__ = ["TraceCapability"]
