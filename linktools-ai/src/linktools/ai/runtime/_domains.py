#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plural Runtime domain entry points."""

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Generic, TypeVar

from ..core import JsonValue, Principal
from ..errors import AIError
from ..task import TaskBindingContract, TaskEffectResolution
from ..agent import AgentBindingContract
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
    ExecutionRequest,
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
    ResumeSessionRequest,
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
    ) -> None:
        self._service = service
        self._get_execution = get_execution

    async def get(
        self,
        execution_id: str,
        *,
        principal: Principal | None = None,
    ) -> "Execution[AppT]":
        return await self._get_execution(execution_id, principal)

    async def acquire_dependency_hold(self, execution_id: str, *, tenant_id: str, hold_id: str) -> bool:
        return await self._service.acquire_dependency_hold(execution_id, tenant_id=tenant_id, hold_id=hold_id)

    async def release_dependency_hold(self, execution_id: str, *, tenant_id: str, hold_id: str) -> None:
        await self._service.release_dependency_hold(execution_id, tenant_id=tenant_id, hold_id=hold_id)

    async def request_terminal_handoff(self, execution_id: str, *, tenant_id: str) -> None:
        await self._service.request_terminal_handoff(execution_id, tenant_id=tenant_id)

    async def start(self, binding_digest: str, request: ExecutionRequest, *, dependency_hold_id: str | None = None, binding_contract: AgentBindingContract | None = None) -> ExecutionHandle:
        return await self._service.start(binding_digest, request, dependency_hold_id=dependency_hold_id, binding_contract=binding_contract)

    async def start_task(self, binding: "TaskBindingContract", *, principal: Principal, input: Mapping[str, JsonValue], idempotency_key: str, correlation: Mapping[str, str | int]) -> ExecutionHandle:
        return await self._service.start_task(binding, principal=principal, input=input, idempotency_key=idempotency_key, correlation=correlation)

    async def claim_task_attempt(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return await self._service.claim_task_attempt(execution_id, principal=principal)

    async def schedule_task_retry(self, execution_id: str, *, principal: Principal, error_code: str) -> ExecutionView:
        return await self._service.schedule_task_retry(execution_id, principal=principal, error_code=error_code)

    async def defer_task_input(self, execution_id: str, *, principal: Principal, wait_id: str) -> ExecutionView:
        return await self._service.defer_task_input(execution_id, principal=principal, wait_id=wait_id)

    async def supply_task_input(self, execution_id: str, *, principal: Principal, value: JsonValue) -> ExecutionView:
        return await self._service.supply_task_input(execution_id, principal=principal, value=value)

    async def resume_task_not_applied(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return await self._service.resume_task_not_applied(execution_id, principal=principal)

    async def resolve_task_effect(self, execution_id: str, *, principal: Principal, resolution: TaskEffectResolution) -> ExecutionView:
        return await self._service.resolve_task_effect(execution_id, principal=principal, resolution=resolution)

    async def complete_task(self, execution_id: str, *, principal: Principal, output: JsonValue) -> ExecutionResult:
        return await self._service.complete_task(execution_id, principal=principal, output=output)

    async def fail_task(self, execution_id: str, *, principal: Principal, error: "AIError") -> ExecutionResult:
        return await self._service.fail_task(execution_id, principal=principal, error=error)

    async def require_task_recovery(self, execution_id: str, *, principal: Principal, error_code: str) -> ExecutionView:
        return await self._service.require_task_recovery(execution_id, principal=principal, error_code=error_code)

    async def cancel_task(self, execution_id: str, *, principal: Principal) -> CancelExecutionResult:
        return await self._service.cancel_task(execution_id, principal=principal)

    async def resolve_existing(self, binding_digest: str, request: ExecutionRequest, *, binding_contract: AgentBindingContract | None = None) -> ExecutionHandle | None:
        return await self._service.resolve_existing(binding_digest, request, binding_contract=binding_contract)

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return await self._service.inspect(execution_id, principal=principal)

    async def list(self, request: ListExecutionRequest) -> Page[ExecutionView]:
        return await self._service.list(request)

    async def list_children(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionView, ...]:
        return await self._service.list_children(execution_id, principal=principal)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        return await self._service.result(execution_id, principal=principal)

    async def result_payload_size(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> int:
        return await self._service.result_payload_size(
            execution_id,
            principal=principal,
        )

    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: float | None = None) -> ExecutionResult:
        return await self._service.wait(execution_id, principal=principal, timeout_seconds=timeout_seconds)

    async def run(self, binding_digest: str, request: ExecutionRequest, *, timeout_seconds: float | None = None, binding_contract: AgentBindingContract | None = None) -> ExecutionResult:
        return await self._service.run(binding_digest, request, timeout_seconds=timeout_seconds, binding_contract=binding_contract)

    async def retry(self, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        return await self._service.retry(execution_id, request)

    async def fork(self, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        return await self._service.fork(execution_id, request)

    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
        return await self._service.cancel(execution_id, request)

    async def cancel_handle(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
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

    async def resume(self, agent_id: str, binding_digest: str, session_id: str, request: ResumeSessionRequest, *, binding_contract: AgentBindingContract | None = None) -> ExecutionHandle:
        return await self._service.resume(agent_id, binding_digest, session_id, request, binding_contract=binding_contract)

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
