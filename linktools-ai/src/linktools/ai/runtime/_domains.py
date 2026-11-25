#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plural Runtime domain entry points."""

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, overload

from ..session import SessionTurnNotice, SessionAdmission, PublishedSessionTurn, SessionService
from ..agent import AgentBindingContract
from ..core import (
    AuthorizationAction, BudgetUsage, JsonValue, Principal, PromptLimits, ResourceKind, ResourceRef, normalize_thinking,
    validate_agent_id, validate_idempotency_key, validate_resource_id,
)
from ..errors import AIError, ErrorCode
from ..persistence.turns import SessionTurnKey
from ..observe import WaitResult, RuntimeMetricsControl
from ..inputs import CaptureInputRequest, ExecutionInputCaptureRef, InputCaptures
from ..execution import ExecutionRecoveryEffect, ResolveToolEffectRequest, ToolEffectResolutionResult
from ..execution import ApprovalDecisionRequest, ApprovalDecisionResult, ApprovalView, ArtifactDownload, ArtifactView, CancelExecutionRequest, CancelExecutionResult, ExecutionEvent, ExecutionHistoryItem, ExecutionResult, ExecutionTraceItem, ExecutionTreeEvent, ExecutionObservationEvent, ExecutionView, ExternalCallView, ExternalResolution, ExternalSupplyRequest, ExternalSupplyResult, ForkExecutionRequest, ListExecutionRequest, ModelInteractionItem, Page, RetryExecutionRequest, UsageReadCutoff, ExecutionService
from ..session import CloseSessionRequest, CreateSessionRequest, ForkSessionRequest, ListSessionRequest, SessionDetail, SessionHistoryItem, SessionTurn, SessionView, UpdateSessionRequest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from ..core import ThinkingValue
    from ..model import UserPromptInput
    from pydantic import BaseModel
    from ._agent import Agent, Session
    from ..execution import Execution
    from ..observe import MetricRecorderStatus, MetricFlushResult
    from ._runtime_service import Runtime
    from ._regenerable_turn import _RegenerableTurnRunner
    from ._agent_executor import AgentEmission

AppT = TypeVar("AppT")


class RuntimeAgents(Generic[AppT]):
    def __init__(self, get_agent: Callable[[str], "Agent[AppT]"], restore_agent: Callable[[AgentBindingContract], "Agent[AppT]"]) -> None:
        self._get_agent = get_agent
        self._restore_agent = restore_agent

    def get(self, agent_id: str = "default") -> "Agent[AppT]":
        return self._get_agent(agent_id)

    def restore(self, binding: AgentBindingContract) -> "Agent[AppT]":
        """Keep recorded behavior; resolve current resources when observed or started."""
        return self._restore_agent(binding)


class RuntimeExecutions(Generic[AppT]):
    def __init__(
        self,
        service: ExecutionService,
        get_execution: Callable[[str, Principal | None], Awaitable["Execution[AppT]"]],
        input_captures: InputCaptures | None = None,
        *,
        bind_execution: Callable[[str, Principal], "Execution[AppT]"],
        ensure_open: Callable[[], None] = lambda: None,
    ) -> None:
        self._service = service
        self._get_execution = get_execution
        self._bind_execution = bind_execution
        self._input_captures = input_captures
        self._ensure_open = ensure_open

    async def get(
        self,
        execution_id: str,
        *,
        principal: Principal | None = None,
    ) -> "Execution[AppT]":
        return await self._get_execution(execution_id, principal)

    async def capture_input(self, execution_id: str, request: CaptureInputRequest) -> ExecutionInputCaptureRef:
        self._ensure_open()
        if self._input_captures is None:
            from ..errors import AIError, ErrorCode
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._input_captures.capture_input(execution_id, request)

    async def budget_usage(self, execution_id: str, *, principal: Principal) -> BudgetUsage | None:
        self._ensure_open()
        return await self._service.budget_usage(execution_id, principal=principal)

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return await self._service.inspect(execution_id, principal=principal)

    async def list(self, request: ListExecutionRequest) -> Page[ExecutionView]:
        return await self._service.list(request)

    async def list_children(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionView, ...]:
        return await self._service.list_children(execution_id, principal=principal)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        return await self._service.result(execution_id, principal=principal)

    @overload
    async def wait(
        self, execution_id: str, *, principal: Principal,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
        include_model_interactions: Literal[False] = False,
    ) -> WaitResult[ExecutionResult]: ...

    @overload
    async def wait(
        self, execution_id: str, *, principal: Principal,
        on_event: Callable[[ExecutionObservationEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
        include_model_interactions: Literal[True],
    ) -> WaitResult[ExecutionResult]: ...

    @overload
    async def wait(
        self, execution_id: str, *, principal: Principal,
        on_event: Callable[[ExecutionTreeEvent | ExecutionObservationEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
        include_model_interactions: bool,
    ) -> WaitResult[ExecutionResult]: ...

    async def wait(
        self, execution_id: str, *, principal: Principal,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | Callable[[ExecutionObservationEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
        include_model_interactions: bool = False,
    ) -> WaitResult[ExecutionResult]:
        execution = await self.get(execution_id, principal=principal)
        return await execution.wait(
            on_event=on_event, cursor=cursor, include_event_content=include_event_content,
            timeout_seconds=timeout_seconds, close_timeout_seconds=close_timeout_seconds,
            raise_on_stream_error=raise_on_stream_error,
            include_model_interactions=include_model_interactions,
        )

    async def retry(self, execution_id: str, request: RetryExecutionRequest) -> "Execution[AppT]":
        self._ensure_open()
        handle = await self._service.retry(execution_id, request)
        return self._bind_execution(handle.execution_id, request.principal)

    async def fork(self, execution_id: str, request: ForkExecutionRequest) -> "Execution[AppT]":
        self._ensure_open()
        handle = await self._service.fork(execution_id, request)
        return self._bind_execution(handle.execution_id, request.principal)

    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
        self._ensure_open()
        return await self._service.cancel(execution_id, request)

    async def recovery_effects(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionRecoveryEffect, ...]:
        return await self._service.recovery_effects(execution_id, principal=principal)

    async def resolve_tool_effect(self, execution_id: str, request: ResolveToolEffectRequest) -> ToolEffectResolutionResult:
        self._ensure_open()
        return await self._service.resolve_tool_effect(execution_id, request)

    async def recover(
        self,
        execution_id: str,
        *,
        principal: Principal,
        idempotency_key: str | None = None,
    ) -> "Execution[AppT]":
        """Take over recoverable work after its previous executor stopped.

        Reusing a key observes the same recovery attempt. Without a key each
        invocation can supersede an earlier attempt. A deferred execution resumes
        only when its persisted approvals and external results are complete. An
        incomplete frontier stays waiting without consuming the key, so the same
        key can be retried after its results arrive.
        """
        self._ensure_open()
        handle = await self._service.recover(
            execution_id, principal=principal, idempotency_key=idempotency_key,
        )
        return self._bind_execution(handle.execution_id, principal)

    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> Page[ExecutionTraceItem]:
        return await self._service.trace(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
        )

    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> Page[ExecutionHistoryItem]:
        return await self._service.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
            message_seq=message_seq,
            part_index=part_index,
        )

    async def model_interactions(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        include_content: bool = False,
        include_client_requests: bool = False,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        limit: int = 100,
        cutoffs: tuple[UsageReadCutoff, ...] | None = None,
    ) -> Page[ModelInteractionItem]:
        return await self._service.model_interactions(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            **({"include_client_requests": include_client_requests} if include_client_requests is not False else {}),
            **({"agent_run_seq": agent_run_seq} if agent_run_seq is not None else {}),
            **({"model_request_seq": model_request_seq} if model_request_seq is not None else {}),
            limit=limit,
            cutoffs=cutoffs,
        )

async def _discard_turn_emission(_event: "AgentEmission") -> None:
    pass


class RuntimeSessions(Generic[AppT]):
    def __init__(
        self,
        service: SessionService,
        get_session: Callable[[str, Principal | None], Awaitable["Session[AppT]"]],
        *,
        start_session: Callable[..., Awaitable["Execution[AppT]"]],
        bind_session: Callable[[SessionView, Principal], "Session[AppT]"],
        ensure_open: Callable[[], None] = lambda: None,
        turn_runner: "_RegenerableTurnRunner | None" = None,
        runtime: "Runtime[AppT] | None" = None,
    ) -> None:
        self._start_session = start_session
        self._service = service
        self._get_session = get_session
        self._bind_session = bind_session
        self._ensure_open = ensure_open
        self._turn_runner = turn_runner
        self._runtime = runtime

    @property
    def supports_turn_sessions(self) -> bool:
        return self._turn_runner is not None and self._runtime is not None

    def _require_turn_sessions(self) -> "tuple[_RegenerableTurnRunner, Runtime[AppT]]":
        self._ensure_open()
        runner, runtime = self._turn_runner, self._runtime
        if runner is None or runtime is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return runner, runtime

    async def create_turn_session(
        self, session_id: str, agent_id: str, *, principal: Principal,
        metadata: "Mapping[str, JsonValue] | None" = None, idempotency_key: str | None = None,
    ) -> dict[str, JsonValue]:
        runner, runtime = self._require_turn_sessions()
        resolver = runtime._binding_resolver
        if resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_resource_id(session_id)
        validate_agent_id(agent_id)
        if idempotency_key is not None:
            validate_idempotency_key(idempotency_key)
        await self._service.require_available()
        return await runner.create_session(
            session_id, agent_id, principal=principal, tenant_id=runtime.tenant_id,
            authorization=runtime._authorization, resolver=resolver,
            metadata={} if metadata is None else metadata, request_key=idempotency_key,
        )

    async def update_turn_session(
        self, session_id: str, *, principal: Principal, expected_revision: int,
        metadata: "Mapping[str, JsonValue]",
    ) -> bool:
        return await self._mutate_turn_session(
            session_id, principal=principal, expected_revision=expected_revision,
            action=AuthorizationAction.SESSION_UPDATE, metadata=metadata,
        )

    async def close_turn_session(
        self, session_id: str, *, principal: Principal, expected_revision: int,
    ) -> bool:
        return await self._mutate_turn_session(
            session_id, principal=principal, expected_revision=expected_revision,
            action=AuthorizationAction.SESSION_CLOSE,
        )

    async def _mutate_turn_session(
        self, session_id: str, *, principal: Principal, expected_revision: int,
        action: AuthorizationAction, metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> bool:
        _, runtime = self._require_turn_sessions()
        validate_resource_id(session_id)
        if type(expected_revision) is not int or expected_revision < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if principal.tenant_id != runtime.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        store = runtime._state.session_turns
        if store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        row = await store.get_session(session_id)
        if row is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        owner = row["owner_principal_id"].decode("utf-8")
        await runtime._authorization.authorize(
            principal, action, ResourceRef(ResourceKind.SESSION, session_id, runtime.tenant_id, owner),
        )
        if action is AuthorizationAction.SESSION_CLOSE:
            return await store.close_session(
                session_id, owner_principal_id=owner, expected_revision=expected_revision,
            )
        return await store.update_session(
            session_id, owner_principal_id=owner, expected_revision=expected_revision,
            config={**row["config_json"], "metadata": dict(metadata or {})},
        )

    async def start_turn(
        self, session_id: str, prompt: str, *, principal: Principal,
        idempotency_key: str, thinking: JsonValue = None,
    ) -> str:
        runner, runtime = self._require_turn_sessions()
        resolver = runtime._binding_resolver
        if resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_resource_id(session_id)
        validate_idempotency_key(idempotency_key)
        if not isinstance(prompt, str) or not prompt.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "A plain text prompt is required")
        await self._service.require_available()
        return await runner.start_session(
            session_id, prompt=prompt, principal=principal, namespace=runtime.namespace,
            tenant_id=runtime.tenant_id, app=runtime.app, authorization=runtime._authorization,
            resolver=resolver, request_key=idempotency_key, limits=PromptLimits(),
            event_sink=_discard_turn_emission, thinking=normalize_thinking(thinking),
        )

    async def cancel_turn(self, session_id: str, turn_id: str, *, principal: Principal) -> bool:
        runner, runtime = self._require_turn_sessions()
        validate_resource_id(session_id)
        validate_resource_id(turn_id)
        return await runner.cancel_session_turn(
            SessionTurnKey(session_id, turn_id), principal=principal,
            tenant_id=runtime.tenant_id, authorization=runtime._authorization,
        )

    async def recover_turn(self, session_id: str, turn_id: str, *, principal: Principal) -> bool:
        runner, runtime = self._require_turn_sessions()
        resolver = runtime._binding_resolver
        if resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_resource_id(session_id)
        validate_resource_id(turn_id)
        return await runner.recover_session_turn(
            SessionTurnKey(session_id, turn_id), principal=principal, namespace=runtime.namespace,
            tenant_id=runtime.tenant_id, app=runtime.app, authorization=runtime._authorization,
            resolver=resolver, limits=PromptLimits(), event_sink=_discard_turn_emission,
        )

    def observe_turn_lifecycle(
        self, callback: Callable[[dict[str, JsonValue]], None],
    ) -> Callable[[], None]:
        """Subscribe a trusted transport to non-content turn milestones."""
        runner, _ = self._require_turn_sessions()
        return runner.observe_lifecycle(callback)

    def invalidate_turn_session(self, session_id: str, turn_id: str, *, revision: int) -> None:
        """Wake an authorized observer after another producer changes this turn."""
        runner, _ = self._require_turn_sessions()
        validate_resource_id(session_id)
        validate_resource_id(turn_id)
        runner.invalidate_session(session_id, turn_id, revision=revision)

    def observe_turn_session(
        self, session_id: str, callback: Callable[[dict[str, JsonValue]], None],
    ) -> Callable[[], None]:
        """Register a trusted host callback. Authorize before forwarding any frame."""
        runner, _ = self._require_turn_sessions()
        validate_resource_id(session_id)
        return runner.observe(session_id, callback)

    def observe_turn_lifecycle(
        self, callback: Callable[[dict[str, JsonValue]], None],
    ) -> Callable[[], None]:
        """Register a trusted host carrier for body-free committed notifications."""
        runner, _ = self._require_turn_sessions()
        return runner.observe_lifecycle(callback)

    def invalidate_turn_session(self, session_id: str, turn_id: str, *, revision: int) -> None:
        """Notify any local observers to re-read the authoritative session."""
        runner, _ = self._require_turn_sessions()
        validate_resource_id(session_id)
        validate_resource_id(turn_id)
        if type(revision) is not int or revision < 1:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        runner.invalidate_session(session_id, turn_id, revision=revision)

    async def get(self, session_id: str, *, principal: Principal | None = None) -> "Session[AppT]":
        return await self._get_session(session_id, principal)

    async def start(
        self, session_id: str, user_prompt: "UserPromptInput", *, principal: Principal,
        idempotency_key: str | None = None, output: "type[BaseModel] | None" = None,
        correlation: "Mapping[str, object] | None" = None, files: "Sequence[str]" = (),
        planning: bool | None = None, thinking: "ThinkingValue | None" = None,
    ) -> "Execution[AppT]":
        """Start a turn on an existing authorized session using its current Agent."""
        return await self._start_session(session_id, user_prompt, principal=principal,
            idempotency_key=idempotency_key, output=output, correlation=correlation,
            files=files, planning=planning, thinking=thinking)

    async def inspect(self, session_id: str, *, principal: Principal) -> SessionView:
        """Read current session state without constructing a bound handle."""
        self._ensure_open()
        return await self._service.get(session_id, principal=principal)

    @contextmanager
    def capture_published_turn(
        self, session_id: str, execution_id: str, *, principal: Principal,
    ) -> Iterator[Callable[[], Awaitable[PublishedSessionTurn | None]]]:
        """Enroll before draining the local execution stream, then read after wait.

        None means no complete publication was captured. Fetch the exact
        committed turn through History when needed. This is not current state.
        """
        self._ensure_open()
        with self._service.capture_published_turn(session_id, execution_id, principal=principal) as captured:
            async def read() -> PublishedSessionTurn | None:
                self._ensure_open()
                return await captured()
            yield read

    def observe_changes(self, session_id: str, on_invalidate: Callable[[], None]) -> Callable[[], None]:
        """Subscribe a trusted host to this Runtime's local session invalidations.

        No initial event, snapshot, authorization grant, external notification
        or expiry timer is provided. Read current state through the authorized
        API when needed. The synchronous callback must schedule rather than
        perform I/O. Unsubscribe when done; Runtime close removes subscriptions.
        """
        self._ensure_open()
        return self._service.observe_changes(session_id, on_invalidate)

    def observe_turns(
        self, handler: Callable[[SessionTurnNotice], None],
    ) -> Callable[[], None]:
        """Subscribe a trusted host to turn milestones produced by this Runtime.

        Best-effort notices carry confirmed admission and release or display-only
        tool progress. They omit prompts, raw tool payloads, and model output.
        The synchronous callback must only schedule work, not do I/O; the host
        owns transport. No replay, remote subscription, or authorization is
        provided. Read current state through the authorized API as needed.
        """
        self._ensure_open()
        return self._service.observe_turns(handler)

    def invalidate(self, session_id: str) -> None:
        """Wake existing local watches to reread current authorized session state."""
        self._ensure_open()
        self._service.invalidate(session_id)

    def watch(self, session_id: str, *, principal: Principal) -> AsyncIterator[SessionView]:
        """Observe this Runtime's session changes; external writes need explicit refresh."""
        self._ensure_open()
        return self._service.watch(session_id, principal=principal)

    async def lookup_admission(
        self, session_id: str, *, principal: Principal, idempotency_key: str,
    ) -> SessionAdmission | None:
        """Read an accepted request without starting or reconciling execution.

        None means no retained admission. An uncertain admission raises
        EXECUTION_START_UNKNOWN; retry with the same key, never a fresh key.
        Callers must still compare their original request with the frozen input.
        """
        self._ensure_open()
        return await self._service.lookup_admission(
            session_id, principal=principal, idempotency_key=idempotency_key,
        )

    async def create(self, agent_id: str, request: CreateSessionRequest) -> "Session[AppT]":
        view = await self.create_view(agent_id, request)
        return self._bind_session(view, request.principal)

    async def create_view(self, agent_id: str, request: CreateSessionRequest) -> SessionView:
        """Return the authorized creation result without a separate session read."""
        self._ensure_open()
        return await self._service.create(agent_id, request)

    async def reconcile(self, session_id: str, *, principal: Principal) -> SessionView:
        self._ensure_open()
        return await self._service.reconcile(session_id, principal=principal)

    async def wait_inactive(
        self, session_id: str, *, principal: Principal, timeout_seconds: float | None = None,
        observation_timeout_seconds: float | None = None,
    ) -> SessionView:
        """Wait for a non-consuming, point-in-time inactive session observation."""
        self._ensure_open()
        return await self._service.wait_inactive(
            session_id, principal=principal, timeout_seconds=timeout_seconds,
            observation_timeout_seconds=observation_timeout_seconds,
        )

    async def refresh(self, session_id: str, *, principal: Principal, cursor: str | None = None, limit: int = 100, max_age: float = 0) -> SessionDetail:
        """Refresh a session; opt in to bounded stale display with positive max_age."""
        self._ensure_open()
        return await self._service.refresh(session_id, principal=principal, cursor=cursor, limit=limit, max_age=max_age)

    async def list(self, request: ListSessionRequest) -> Page[SessionView]:
        return await self._service.list(request)

    async def history(self, session_id: str, *, principal: Principal, cursor: str | None = None, limit: int = 100) -> Page[SessionHistoryItem]:
        return await self._service.history(session_id, principal=principal, cursor=cursor, limit=limit)

    async def timeline(self, session_id: str, *, principal: Principal, cursor: str | None = None, limit: int = 100) -> Page[SessionTurn]:
        return await self._service.timeline(session_id, principal=principal, cursor=cursor, limit=limit)

    async def fork(self, agent_id: str, session_id: str, request: ForkSessionRequest) -> "Session[AppT]":
        self._ensure_open()
        view = await self._service.fork(agent_id, session_id, request)
        return self._bind_session(view, request.principal)

    async def update(self, agent_id: str, session_id: str, request: UpdateSessionRequest) -> SessionView:
        self._ensure_open()
        return await self._service.update(agent_id, session_id, request)

    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        self._ensure_open()
        return await self._service.close(session_id, request)


class RuntimeMetrics(RuntimeMetricsControl):
    """Aggregate metrics for producers created by this Runtime, not borrowed peers."""


__all__ = ["RuntimeAgents", "RuntimeExecutions", "RuntimeMetrics", "RuntimeSessions"]
