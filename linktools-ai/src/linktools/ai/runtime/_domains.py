#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plural Runtime domain entry points."""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from ..core import Principal
from ._input_capture import CaptureInputRequest, ExecutionInputCaptureRef, RuntimeInputCaptures
from .recovery import (
    ExecutionRecoveryEffect,
    ResolveToolEffectRequest,
    ToolEffectResolutionResult,
)
from .service_api import (
    ApprovalDecisionRequest,
    ApprovalDecisionResult,
    ApprovalView,
    ArtifactDownload,
    ArtifactView,
    CancelExecutionRequest,
    CancelExecutionResult,
    CloseSessionRequest,
    CompareEvaluationRequest,
    CreateSessionRequest,
    EvaluationComparison,
    EvaluationHandle,
    EvaluationView,
    ExecutionEvent,
    ExecutionHandle,
    ExecutionHistoryItem,
    ExecutionResult,
    ExecutionTraceItem,
    ExecutionTreeEvent,
    ExecutionView,
    ExternalCallView,
    ExternalResolution,
    ExternalSupplyRequest,
    ExternalSupplyResult,
    ForkExecutionRequest,
    ForkSessionRequest,
    ListExecutionRequest,
    ListSessionRequest,
    ModelInteractionItem,
    Page,
    ReplayEvaluationRequest,
    RetryExecutionRequest,
    SessionHistoryItem,
    SessionTurn,
    SessionView,
    StartEvaluationRequest,
    TranscriptItem,
    UpdateSessionRequest,
    UsageReadCutoff,
    ExecutionService,
    SessionService,
)

if TYPE_CHECKING:
    from pydantic import BaseModel
    from ._agent import Agent, Execution, Session
    from ._metrics import MetricBufferStatus, MetricFlushResult
    from ._runtime_service import Runtime

AppT = TypeVar("AppT")


class RuntimeAgents(Generic[AppT]):
    def __init__(self, get_agent: Callable[[str], "Agent[AppT]"]) -> None:
        self._get_agent = get_agent

    def get(self, agent_id: str = "default") -> "Agent[AppT]":
        return self._get_agent(agent_id)


class RuntimeExecutions(Generic[AppT]):
    def __init__(
        self,
        service: ExecutionService,
        get_execution: Callable[[str, Principal | None], Awaitable["Execution[AppT]"]],
        input_captures: RuntimeInputCaptures | None = None,
    ) -> None:
        self._service = service
        self._get_execution = get_execution
        self._input_captures = input_captures

    async def get(
        self,
        execution_id: str,
        *,
        principal: Principal | None = None,
    ) -> "Execution[AppT]":
        return await self._get_execution(execution_id, principal)

    async def capture_input(self, execution_id: str, request: CaptureInputRequest) -> ExecutionInputCaptureRef:
        if self._input_captures is None:
            from ..errors import AIError, ErrorCode
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._input_captures.capture_input(execution_id, request)


    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return await self._service.inspect(execution_id, principal=principal)

    async def list(self, request: ListExecutionRequest) -> Page[ExecutionView]:
        return await self._service.list(request)

    async def list_children(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionView, ...]:
        return await self._service.list_children(execution_id, principal=principal)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        return await self._service.result(execution_id, principal=principal)

    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: float | None = None) -> ExecutionResult:
        return await self._service.wait(execution_id, principal=principal, timeout_seconds=timeout_seconds)

    async def retry(self, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        return await self._service.retry(execution_id, request)

    async def fork(self, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        return await self._service.fork(execution_id, request)

    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
        return await self._service.cancel(execution_id, request)

    async def recovery_effects(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionRecoveryEffect, ...]:
        return await self._service.recovery_effects(execution_id, principal=principal)

    async def resolve_tool_effect(self, execution_id: str, request: ResolveToolEffectRequest) -> ToolEffectResolutionResult:
        return await self._service.resolve_tool_effect(execution_id, request)

    async def recover(self, execution_id: str, *, principal: Principal) -> ExecutionHandle:
        return await self._service.recover(execution_id, principal=principal)

    async def trace(self, execution_id: str, *, principal: Principal, cursor: str | None = None, include_content: bool = False, limit: int = 100) -> Page[ExecutionTraceItem]:
        return await self._service.trace(execution_id, principal=principal, cursor=cursor, include_content=include_content, limit=limit)

    async def transcript(self, execution_id: str, *, principal: Principal, cursor: str | None = None, include_content: bool = False, limit: int = 100) -> Page[TranscriptItem]:
        return await self._service.transcript(execution_id, principal=principal, cursor=cursor, include_content=include_content, limit=limit)

    async def history(self, execution_id: str, *, principal: Principal, cursor: str | None = None, include_content: bool = False, limit: int = 100) -> Page[ExecutionHistoryItem]:
        return await self._service.history(execution_id, principal=principal, cursor=cursor, include_content=include_content, limit=limit)

    async def model_interactions(self, execution_id: str, *, principal: Principal, cursor: str | None = None, include_content: bool = False, limit: int = 100, cutoffs: tuple[UsageReadCutoff, ...] | None = None) -> Page[ModelInteractionItem]:
        return await self._service.model_interactions(execution_id, principal=principal, cursor=cursor, include_content=include_content, limit=limit, cutoffs=cutoffs)

class RuntimeSessions(Generic[AppT]):
    def __init__(
        self,
        service: SessionService,
        get_session: Callable[[str, Principal | None], Awaitable["Session[AppT]"]],
    ) -> None:
        self._service = service
        self._get_session = get_session

    async def get(self, session_id: str, *, principal: Principal | None = None) -> "Session[AppT]":
        return await self._get_session(session_id, principal)

    async def create(self, agent_id: str, request: CreateSessionRequest) -> SessionView:
        return await self._service.create(agent_id, request)

    async def reconcile(self, session_id: str, *, principal: Principal) -> SessionView:
        return await self._service.reconcile(session_id, principal=principal)

    async def list(self, request: ListSessionRequest) -> Page[SessionView]:
        return await self._service.list(request)

    async def history(self, session_id: str, *, principal: Principal, cursor: str | None = None, limit: int = 100) -> Page[SessionHistoryItem]:
        return await self._service.history(session_id, principal=principal, cursor=cursor, limit=limit)

    async def timeline(self, session_id: str, *, principal: Principal, cursor: str | None = None, limit: int = 100) -> Page[SessionTurn]:
        return await self._service.timeline(session_id, principal=principal, cursor=cursor, limit=limit)

    async def fork(self, agent_id: str, session_id: str, request: ForkSessionRequest) -> SessionView:
        return await self._service.fork(agent_id, session_id, request)

    async def update(self, agent_id: str, session_id: str, request: UpdateSessionRequest) -> SessionView:
        return await self._service.update(agent_id, session_id, request)

    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        return await self._service.close(session_id, request)


class RuntimeMetrics:
    def __init__(
        self,
        status: Callable[[], "MetricBufferStatus"],
        flush: Callable[..., Awaitable["MetricFlushResult"]],
    ) -> None:
        self._status = status
        self._flush = flush

    def status(self) -> "MetricBufferStatus":
        return self._status()

    async def flush(self, *, timeout_seconds: float = 5.0) -> "MetricFlushResult":
        return await self._flush(timeout_seconds=timeout_seconds)


__all__ = ["RuntimeAgents", "RuntimeExecutions", "RuntimeMetrics", "RuntimeSessions"]
