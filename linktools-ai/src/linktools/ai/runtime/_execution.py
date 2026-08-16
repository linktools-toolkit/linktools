#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution query API and the persistence-backed default service."""

import asyncio
import json
import re
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from functools import wraps
from typing import Protocol

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    StopReason,
    UsageMetrics,
    canonical_sha256,
    idempotency_key_digest as compute_idempotency_key_digest,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, read_object
from .service_api import (
    CancelExecutionRequest,
    CancelExecutionResult,
    ExecutionHandle,
    ExecutionHistoryItem,
    ExecutionHistoryReader,
    ExecutionRequest,
    ExecutionResult,
    ExecutionTraceItem,
    ExecutionView,
    ForkExecutionRequest,
    RetryExecutionRequest,
    TranscriptItem,
)
from .state._contracts import (
    ExecutionCancelRequestCommit,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartUnknownCommit,
    ExecutionState,
    ExecutionTerminalCommit,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationTerminalUpdate,
    ResultRecord,
    SessionRepository,
)
from .state._plan import RuntimeDomain

_logger = environ.get_logger("ai.runtime.execution")


def _consumed_query(method: "Callable[..., object]") -> "Callable[..., object]":
    @wraps(method)
    async def wrapped(self: "DefaultExecutionService", execution_id: str, *args: object, principal: Principal, **kwargs: object) -> object:
        async with self._execution_consumer(execution_id, principal.tenant_id):
            result = await method(self, execution_id, *args, principal=principal, **kwargs)
            await self._request_handoff_if_terminal(execution_id, principal.tenant_id)
            return result

    return wrapped


def _observed_query(method: "Callable[..., object]") -> "Callable[..., object]":
    @wraps(method)
    async def wrapped(self: "DefaultExecutionService", execution_id: str, *args: object, principal: Principal, **kwargs: object) -> object:
        async with self._execution_consumer(execution_id, principal.tenant_id):
            return await method(self, execution_id, *args, principal=principal, **kwargs)

    return wrapped


class _ExecutionReleaseCallback(Protocol):
    async def __call__(self, execution_id: str, *, tenant_id: str) -> None: ...


class _ExecutionTerminalVerifier(Protocol):
    async def __call__(self, execution: ExecutionRecord, status: ExecutionStatus, required_step_run_id: "str | None") -> None: ...


class _SubagentCancellation(Protocol):
    async def cancel_children(self, parent_execution_id: str, principal: Principal) -> None: ...


async def _no_release_terminal(execution_id: str, *, tenant_id: str) -> None:
    del execution_id, tenant_id


async def _missing_terminal_verifier(execution: ExecutionRecord, status: ExecutionStatus, required_step_run_id: "str | None") -> None:
    del execution, status, required_step_run_id
    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


@dataclass
class _ExecutionHandoffState:
    active_consumers: int = 0
    dependency_holds: set[str] = field(default_factory=set)
    release_requested: bool = False
    release_in_progress: bool = False


class ExecutionQueryApi(Protocol):
    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView: ...
    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult: ...
    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> ExecutionResult: ...
    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[ExecutionTraceItem]': ...
    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[TranscriptItem]': ...
    async def history(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[ExecutionHistoryItem]': ...


class ExecutionApi(ExecutionQueryApi, Protocol):
    async def run(self, request: ExecutionRequest) -> ExecutionHandle: ...
    async def run_and_wait(self, request: ExecutionRequest, *, timeout_seconds: "float | None" = None) -> ExecutionResult: ...
    async def retry(self, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle: ...
    async def fork(self, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle: ...
    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult: ...


class ExecutionBackend(Protocol):
    async def start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None: ...
    async def cancel(self, execution: ExecutionRecord) -> "CancelEffectOutcome": ...


class CancelEffectOutcome(StrEnum):
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"


class DefaultExecutionService:
    """Apply execution authorization and state transitions before launching work."""

    def __init__(
        self,
        state: ExecutionState,
        object_store: ObjectStore,
        authorization: AuthorizationPolicy,
        *,
        sessions: SessionRepository,
        backend: "ExecutionBackend | None" = None,
        operation_ids: "Callable[[], str] | None" = None,
        history_reader: ExecutionHistoryReader,
        release_terminal: _ExecutionReleaseCallback | None = None,
        terminal_verifier: "_ExecutionTerminalVerifier | None" = None,
    ) -> None:
        self._state = state
        self._object_store = object_store
        self._sessions = sessions
        self._authorization = authorization
        self._backend = backend
        self._operation_ids = operation_ids or (lambda: uuid.uuid4().hex)
        self._history_reader = history_reader
        self._release_terminal = release_terminal or _no_release_terminal
        self._terminal_verifier = terminal_verifier or _missing_terminal_verifier
        self._subagent_cancellation: _SubagentCancellation | None = None
        self._session_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._session_locks_guard = asyncio.Lock()
        self._handoff_states: dict[tuple[str, str], _ExecutionHandoffState] = {}
        self._handoff_condition = asyncio.Condition()

    def bind_backend(self, backend: ExecutionBackend) -> None:
        if backend is None:
            raise ValueError("execution backend is required")
        if self._backend is not None:
            raise RuntimeError("execution backend is already bound")
        self._backend = backend

    def bind_subagent_cancellation(self, cancellation: _SubagentCancellation) -> None:
        if self._subagent_cancellation is not None:
            raise RuntimeError("subagent cancellation is already bound")
        self._subagent_cancellation = cancellation

    async def _acquire_dependency_hold(self, execution_id: str, *, tenant_id: str, hold_id: str) -> None:
        if not hold_id:
            raise ValueError("execution dependency hold id is required")
        async with self._handoff_condition:
            state = self._handoff_states.setdefault((tenant_id, execution_id), _ExecutionHandoffState())
            while state.release_in_progress:
                await self._handoff_condition.wait()
            state.dependency_holds.add(hold_id)

    async def _release_dependency_hold(self, execution_id: str, *, tenant_id: str, hold_id: str) -> None:
        owner = False
        state: _ExecutionHandoffState | None = None
        async with self._handoff_condition:
            state = self._handoff_states.get((tenant_id, execution_id))
            if state is None:
                return
            state.dependency_holds.discard(hold_id)
            owner = self._claim_cleanup_locked(state)
            if not owner and not state.release_requested and not state.dependency_holds and state.active_consumers == 0:
                self._handoff_states.pop((tenant_id, execution_id), None)
        if owner and state is not None:
            await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def _request_terminal_handoff(self, execution_id: str, *, tenant_id: str) -> None:
        owner = False
        state: _ExecutionHandoffState
        async with self._handoff_condition:
            state = self._handoff_states.setdefault((tenant_id, execution_id), _ExecutionHandoffState())
            state.release_requested = True
            owner = self._claim_cleanup_locked(state)
        if owner:
            await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def _request_handoff_if_terminal(self, execution_id: str, tenant_id: str) -> None:
        current = await self._state.executions.get(execution_id, tenant_id=tenant_id)
        if current is not None and current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            await self._request_terminal_handoff(execution_id, tenant_id=tenant_id)

    def _claim_cleanup_locked(self, state: _ExecutionHandoffState) -> bool:
        if state.active_consumers == 0 and not state.dependency_holds and state.release_requested and not state.release_in_progress:
            state.release_in_progress = True
            return True
        return False

    async def _run_handoff_cleanup(self, execution_id: str, tenant_id: str, state: _ExecutionHandoffState) -> None:
        try:
            await self._release_terminal(execution_id, tenant_id=tenant_id)
        except BaseException:
            async with self._handoff_condition:
                state.release_in_progress = False
                state.release_requested = True
                self._handoff_condition.notify_all()
            _logger.error("execution handoff cleanup failed: execution=%s", execution_id, exc_info=environ.debug)
            return
        async with self._handoff_condition:
            if self._handoff_states.get((tenant_id, execution_id)) is state:
                self._handoff_states.pop((tenant_id, execution_id), None)
            self._handoff_condition.notify_all()

    @asynccontextmanager
    async def _execution_consumer(self, execution_id: str, tenant_id: str):
        async with self._handoff_condition:
            state = self._handoff_states.setdefault((tenant_id, execution_id), _ExecutionHandoffState())
            while state.release_in_progress:
                await self._handoff_condition.wait()
            state.active_consumers += 1
        try:
            yield
        finally:
            owner = False
            async with self._handoff_condition:
                state.active_consumers -= 1
                if state.active_consumers < 0:
                    raise RuntimeError("execution consumer count underflow")
                owner = self._claim_cleanup_locked(state)
                if not owner and not state.release_requested and not state.dependency_holds and state.active_consumers == 0:
                    self._handoff_states.pop((tenant_id, execution_id), None)
            if owner:
                await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def run(self, binding_digest: str, request: ExecutionRequest) -> ExecutionHandle:
        return await self._start(binding_digest, request, scope="execution.run")

    async def run_for_session(self, binding_digest: str, session_id: str, request: ExecutionRequest) -> ExecutionHandle:
        if not session_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._start(binding_digest, request, session_id=session_id, scope="session.resume")

    async def start_subagent(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        parent_execution_id: str,
        root_execution_id: str,
    ) -> ExecutionHandle:
        return await self._start(
            binding_digest,
            request,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id,
            lineage_kind=ExecutionLineageKind.SUBAGENT,
            scope="execution.subagent",
        )

    async def list_children(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionView, ...]:
        async with self._execution_consumer(execution_id, principal.tenant_id):
            await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
            children = await self._state.executions.list_children(
                execution_id,
                tenant_id=principal.tenant_id,
            )
            return tuple(ExecutionView(child.execution_id, child.status) for child in children)

    async def _start(self, binding_digest: str, request: ExecutionRequest, *, session_id: "str | None" = None, source_execution_id: "str | None" = None, base_execution_id: "str | None" = None, parent_execution_id: "str | None" = None, root_execution_id: "str | None" = None, lineage_kind: ExecutionLineageKind = ExecutionLineageKind.RUN, scope: str = "execution.run") -> ExecutionHandle:
        if session_id is None:
            return await self._start_unlocked(binding_digest, request, session_id=session_id, source_execution_id=source_execution_id, base_execution_id=base_execution_id, parent_execution_id=parent_execution_id, root_execution_id=root_execution_id, lineage_kind=lineage_kind, scope=scope)
        lock = await self._session_lock(request.principal.tenant_id, session_id)
        async with lock:
            return await self._start_unlocked(binding_digest, request, session_id=session_id, source_execution_id=source_execution_id, base_execution_id=base_execution_id, parent_execution_id=parent_execution_id, root_execution_id=root_execution_id, lineage_kind=lineage_kind, scope=scope)

    async def _start_unlocked(self, binding_digest: str, request: ExecutionRequest, *, session_id: "str | None" = None, source_execution_id: "str | None" = None, base_execution_id: "str | None" = None, parent_execution_id: "str | None" = None, root_execution_id: "str | None" = None, lineage_kind: ExecutionLineageKind = ExecutionLineageKind.RUN, scope: str = "execution.run") -> ExecutionHandle:
        if re.fullmatch(r"[0-9a-f]{64}", binding_digest) is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if request.idempotency_key is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        conversation_run_id = None
        if session_id is not None and source_execution_id is None:
            session = await self._sessions.get(session_id, tenant_id=request.principal.tenant_id)
            if session is None:
                raise AIError(ErrorCode.SESSION_NOT_FOUND)
            conversation_run_id = None if session.continuation is None else session.continuation.step_run_id
            base_execution_id = None
            lineage_kind = ExecutionLineageKind.SESSION_RESUME
        execution_id = self._operation_ids()
        resource = ResourceRef(ResourceKind.EXECUTION, execution_id, request.principal.tenant_id)
        await self._authorization.authorize(request.principal, AuthorizationAction.EXECUTION_RUN, resource)
        idempotency_key_digest = compute_idempotency_key_digest(request.idempotency_key)
        request_digest = _request_digest(request, binding_digest, session_id=session_id, source_execution_id=source_execution_id, base_execution_id=base_execution_id, parent_execution_id=parent_execution_id, root_execution_id=root_execution_id, lineage_kind=lineage_kind)
        existing = await self._state.idempotency.get(
            scope,
            idempotency_key_digest,
            tenant_id=request.principal.tenant_id,
        )
        if existing is not None:
            if existing.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if existing.status is IdempotencyStatus.START_UNKNOWN:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            if existing.status is IdempotencyStatus.RESERVED:
                pending = await self._state.executions.get(existing.resource_id, tenant_id=request.principal.tenant_id)
                if pending is not None and pending.status is ExecutionStatus.STARTED:
                    return ExecutionHandle(existing.resource_id)
                if pending is None or pending.status is not ExecutionStatus.PENDING_START:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    await self._state.executions.claim_start(
                        ExecutionStartClaim(
                            existing.resource_id,
                            request.principal.tenant_id,
                            pending.revision,
                            pending.event_sequence,
                            scope,
                            idempotency_key_digest,
                            request_digest,
                            datetime.now(timezone.utc),
                        )
                    )
                except AIError as error:
                    current = await self._state.executions.get(existing.resource_id, tenant_id=request.principal.tenant_id)
                    if error.code is not ErrorCode.STORAGE_CONFLICT or current is None or current.status is not ExecutionStatus.STARTED:
                        raise
                    return ExecutionHandle(existing.resource_id)
                await self._launch_claim(
                    request,
                    existing.resource_id,
                    request.principal.tenant_id,
                    scope,
                    idempotency_key_digest,
                )
                return ExecutionHandle(existing.resource_id)
            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_START_PERSISTENCE_FAILED)
            if existing.status is IdempotencyStatus.CANCELLED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_CANCELLED)
            return ExecutionHandle(existing.resource_id)
        if session_id is not None:
            active = await self._state.executions.list_by_session(session_id, tenant_id=request.principal.tenant_id)
            if any(item.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} for item in active):
                raise AIError(ErrorCode.SESSION_BUSY)
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id=execution_id,
            tenant_id=request.principal.tenant_id,
            session_id=session_id,
            binding_digest=binding_digest,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id or execution_id,
            status=ExecutionStatus.PENDING_START,
            revision=0,
            event_sequence=0,
            error_code=None,
            safe_error_details={},
            created_at=now,
            updated_at=now,
            source_execution_id=source_execution_id,
            base_execution_id=base_execution_id,
            lineage_kind=lineage_kind,
            agent_run_sequence=0,
            memory_scope=request.memory_scope,
            conversation_step_run_id=conversation_run_id,
        )
        reservation = await self._state.executions.reserve_start(
            ExecutionStartReservation(
                execution,
                IdempotencyRecord(
                    tenant_id=request.principal.tenant_id,
                    runtime_domain=RuntimeDomain.EXECUTION,
                    scope=scope,
                    idempotency_key_digest=idempotency_key_digest,
                    request_digest=request_digest,
                    resource_kind=ResourceKind.EXECUTION,
                    resource_id=execution_id,
                    status=IdempotencyStatus.RESERVED,
                    result_digest=None,
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                ),
            )
        )
        if not reservation.created:
            if reservation.idempotency.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if reservation.execution.status is ExecutionStatus.PENDING_START and reservation.idempotency.status is IdempotencyStatus.RESERVED:
                try:
                    await self._state.executions.claim_start(
                        ExecutionStartClaim(
                            reservation.execution.execution_id,
                            reservation.execution.tenant_id,
                            reservation.execution.revision,
                            reservation.execution.event_sequence,
                            scope,
                            idempotency_key_digest,
                            request_digest,
                            now,
                        )
                    )
                except AIError as error:
                    current = await self._state.executions.get(reservation.execution.execution_id, tenant_id=reservation.execution.tenant_id)
                    if error.code is not ErrorCode.STORAGE_CONFLICT or current is None or current.status is not ExecutionStatus.STARTED:
                        raise
                    return ExecutionHandle(reservation.execution.execution_id)
                await self._launch_claim(
                    request,
                    reservation.execution.execution_id,
                    reservation.execution.tenant_id,
                    scope,
                    idempotency_key_digest,
                )
            elif reservation.execution.status is ExecutionStatus.STARTED and reservation.idempotency.status is IdempotencyStatus.STARTED:
                return ExecutionHandle(reservation.execution.execution_id)
            elif reservation.execution.status is ExecutionStatus.START_UNKNOWN or reservation.idempotency.status is IdempotencyStatus.START_UNKNOWN:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ExecutionHandle(reservation.execution.execution_id)
        execution_id = reservation.execution.execution_id
        try:
            await self._state.executions.claim_start(
                ExecutionStartClaim(
                    execution_id,
                    request.principal.tenant_id,
                    0,
                    0,
                    scope,
                    idempotency_key_digest,
                    request_digest,
                    now,
                )
            )
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                raise
            raise AIError(ErrorCode.EXECUTION_START_PERSISTENCE_FAILED) from error
        await self._launch_claim(request, execution_id, request.principal.tenant_id, scope, idempotency_key_digest)
        _logger.info("execution started: execution=%s scope=%s", execution_id, scope)
        return ExecutionHandle(execution_id)

    async def _session_lock(self, tenant_id: str, session_id: str) -> asyncio.Lock:
        key = tenant_id, session_id
        async with self._session_locks_guard:
            lock = self._session_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[key] = lock
            return lock

    async def _launch_claim(
        self,
        request: ExecutionRequest,
        execution_id: str,
        tenant_id: str,
        scope: str,
        idempotency_key_digest: str,
    ) -> None:
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            launch_record = await self._state.executions.get(execution_id, tenant_id=tenant_id)
            if launch_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._backend.start(request, launch_record)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            current = await self._state.executions.get(execution_id, tenant_id=tenant_id)
            if current is not None and current.status is ExecutionStatus.STARTED:
                identity = await self._state.idempotency.get(scope, idempotency_key_digest, tenant_id=tenant_id)
                if identity is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._state.executions.mark_start_unknown(
                    ExecutionStartUnknownCommit(
                        execution_id,
                        tenant_id,
                        current.revision,
                        current.event_sequence,
                        scope,
                        idempotency_key_digest,
                        identity.request_digest,
                        datetime.now(timezone.utc),
                    )
                )
            _logger.error("execution start outcome unknown: execution=%s", execution_id)
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN) from error

    @_observed_query
    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return ExecutionView(execution.execution_id, execution.status)

    @_consumed_query
    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        if execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        result = await self._state.executions.get_result(execution_id, tenant_id=principal.tenant_id)
        if result is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            if result.object_ref is not None or any(value is not None for value in (result.output_schema_id, result.output_schema_revision, result.output_schema_fingerprint)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ExecutionResult(
                execution.execution_id,
                execution.status,
                None,
                result.output_schema_id,
                result.output_schema_revision,
                result.output_schema_fingerprint,
                result.usage,
            )
        if result.object_ref is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            payload = await read_object(
                self._object_store,
                result.object_ref.key,
                expected_digest=result.object_ref.digest,
                expected_size=result.object_ref.size,
            )
            if result.output_schema_id == "text":
                output: JsonValue = payload.decode("utf-8")
            else:
                decoded = json.loads(payload.decode("utf-8"))
                output = decoded
            return ExecutionResult(
                execution.execution_id,
                execution.status,
                output,
                result.output_schema_id,
                result.output_schema_revision,
                result.output_schema_fingerprint,
                result.usage,
            )
        except AIError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    @_consumed_query
    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> ExecutionResult:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def poll() -> ExecutionResult:
            while True:
                view = await self.inspect(execution_id, principal=principal)
                if view.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    return await self.result(execution_id, principal=principal)
                await asyncio.sleep(0.05)

        try:
            return await asyncio.wait_for(poll(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE, "execution wait timed out") from error

    async def run_and_wait(self, binding_digest: str, request: ExecutionRequest, *, timeout_seconds: "float | None" = None) -> ExecutionResult:
        handle = await self.run(binding_digest, request)
        return await self.wait(handle.execution_id, principal=request.principal, timeout_seconds=timeout_seconds)

    async def retry(self, binding_digest: str, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding_digest:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        retry_request = ExecutionRequest(
            user_prompt=request.user_prompt,
            principal=request.principal,
            idempotency_key=request.idempotency_key,
            memory_scope=previous.memory_scope,
        )
        return await self._start(binding_digest, retry_request, session_id=previous.session_id, parent_execution_id=previous.parent_execution_id, root_execution_id=previous.root_execution_id, scope="execution.retry", source_execution_id=previous.execution_id, lineage_kind=ExecutionLineageKind.RETRY, base_execution_id=previous.base_execution_id)

    async def fork(self, binding_digest: str, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding_digest:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        fork_request = ExecutionRequest(
            user_prompt=request.user_prompt,
            principal=request.principal,
            idempotency_key=request.idempotency_key,
            memory_scope=previous.memory_scope,
        )
        return await self._start(binding_digest, fork_request, session_id=previous.session_id, parent_execution_id=previous.parent_execution_id, root_execution_id=previous.root_execution_id, scope="execution.fork", source_execution_id=previous.execution_id, lineage_kind=ExecutionLineageKind.FORK, base_execution_id=previous.execution_id)

    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
        async with self._execution_consumer(execution_id, request.principal.tenant_id):
            result = await self._cancel(execution_id, request)
            await self._request_handoff_if_terminal(execution_id, request.principal.tenant_id)
            return result

    async def _cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
        execution = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_CANCEL)
        operation_digest = canonical_sha256(
            {
                "action": "execution.cancel",
                "principal": principal_identity_payload(request.principal),
                "execution_id": execution_id,
                "force": request.force,
            }
        )
        operation_id = compute_idempotency_key_digest(request.idempotency_key)
        operation = await self._state.operations.get(
            operation_id,
            tenant_id=request.principal.tenant_id,
        )
        if operation is not None:
            if operation.request_digest != operation_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                return CancelExecutionResult(execution_id, execution.status is ExecutionStatus.CANCELLED)
            if operation.status is OperationStatus.EFFECT_UNKNOWN:
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                return CancelExecutionResult(execution_id, False)
            if operation.status is OperationStatus.FAILED:
                raise _stable_operation_error(operation.error_code)
        else:
            now = datetime.now(timezone.utc)
            operation = OperationLedgerInput(
                operation_id,
                request.principal.tenant_id,
                ResourceKind.EXECUTION,
                execution_id,
                execution_id,
                OperationKind.EXECUTION_CANCEL,
                OperationStatus.PENDING,
                operation_digest,
                None,
                None,
                None,
                True,
                now,
                now,
            )
            operation = await self._state.operations.append(operation)
        if execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
            if resolved is not None:
                return resolved
        try:
            try:
                cancelling = await self._state.executions.request_cancel(
                    ExecutionCancelRequestCommit(
                        execution_id=execution_id,
                        tenant_id=request.principal.tenant_id,
                        expected_revision=execution.revision,
                        expected_event_sequence=execution.event_sequence,
                        operation_id=operation.operation_id,
                        requested_at=datetime.now(timezone.utc),
                    )
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                raise
            if self._backend is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if self._subagent_cancellation is not None:
                await self._subagent_cancellation.cancel_children(cancelling.execution_id, request.principal)
            outcome = await self._backend.cancel(cancelling)
            if outcome is CancelEffectOutcome.UNKNOWN:
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                current = await self._state.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.status is OperationStatus.PENDING:
                    try:
                        current = await self._state.operations.compare_and_swap(
                            operation.operation_id,
                            tenant_id=request.principal.tenant_id,
                            expected_status=OperationStatus.PENDING,
                            next_record=_operation_status(current, OperationStatus.EFFECT_UNKNOWN),
                        )
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_CONFLICT:
                            raise
                        resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                        if resolved is not None:
                            return resolved
                        current = await self._state.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
                        if current is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.status is OperationStatus.EFFECT_UNKNOWN:
                    _logger.warning("execution cancellation effect unknown: execution=%s operation=%s", execution_id, operation.operation_id)
                    return CancelExecutionResult(execution_id, False)
                if current.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                    latest = await self._state.executions.get(execution_id, tenant_id=request.principal.tenant_id)
                    if latest is None or latest.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    return CancelExecutionResult(execution_id, latest.status is ExecutionStatus.CANCELLED)
                if current.status is OperationStatus.FAILED:
                    raise _stable_operation_error(current.error_code)
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            cancelling_current = await self._state.executions.get(execution_id, tenant_id=request.principal.tenant_id)
            if cancelling_current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
            if resolved is not None:
                return resolved
            if cancelling_current.status is not ExecutionStatus.CANCELLING:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            now = datetime.now(timezone.utc)
            terminal = _next_execution(cancelling_current, ExecutionStatus.CANCELLED, now, error_code=None, terminal_event=True)
            result = ResultRecord(
                execution_id,
                request.principal.tenant_id,
                None,
                None,
                None,
                None,
                StopReason.CANCELLED,
                UsageMetrics(),
                now,
            )
            idempotency_records = await self._state.idempotency.list_by_resource(ResourceKind.EXECUTION, execution_id, tenant_id=request.principal.tenant_id)
            if len(idempotency_records) > 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            identity = idempotency_records[0] if idempotency_records else None
            idempotency = (
                None
                if identity is None
                else IdempotencyTerminalUpdate(
                    identity.scope,
                    identity.idempotency_key_digest,
                    identity.status,
                    IdempotencyStatus.CANCELLED,
                    identity.request_digest,
                    None,
                    None,
                )
            )
            current_operation = await self._state.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
            if current_operation is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current_operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                latest = await self._state.executions.get(execution_id, tenant_id=request.principal.tenant_id)
                if latest is None or latest.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return CancelExecutionResult(execution_id, latest.status is ExecutionStatus.CANCELLED)
            if current_operation.status is OperationStatus.FAILED:
                raise _stable_operation_error(current_operation.error_code)
            if current_operation.status not in {OperationStatus.PENDING, OperationStatus.EFFECT_UNKNOWN}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            operation_update = OperationTerminalUpdate(operation.operation_id, current_operation.status, OperationStatus.SUCCEEDED, execution_id, None, None)
            await self._terminal_verifier(cancelling_current, ExecutionStatus.CANCELLED, None)
            try:
                await self._state.executions.commit_terminal(
                    ExecutionTerminalCommit(
                        expected_revision=cancelling_current.revision,
                        expected_event_sequence=cancelling_current.event_sequence,
                        execution=terminal,
                        result=result,
                        terminal_event_type=ExecutionEventType.EXECUTION_CANCELLED,
                        terminal_event_payload={},
                        idempotency=idempotency,
                        operation=operation_update,
                    )
                )
            except AIError as error:
                if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.EXECUTION_RESULT_CONFLICT}:
                    raise
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                raise
            _logger.info("execution cancelled: execution=%s operation=%s", execution_id, operation.operation_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
            if resolved is not None:
                return resolved
            if isinstance(error, AIError) and error.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.EXECUTION_RESULT_CONFLICT}:
                raise
            error_code = error.code.value if isinstance(error, AIError) else ErrorCode.STORAGE_UNAVAILABLE.value
            try:
                current = await self._state.operations.get(operation.operation_id, tenant_id=request.principal.tenant_id)
                if current is not None and current.status is OperationStatus.PENDING:
                    await self._state.operations.compare_and_swap(
                        operation.operation_id,
                        tenant_id=request.principal.tenant_id,
                        expected_status=OperationStatus.PENDING,
                        next_record=_operation_failure(current, error_code),
                    )
            except Exception:
                _logger.error(
                    "execution cancellation ledger update failed: execution=%s operation=%s",
                    execution_id,
                    operation.operation_id,
                    exc_info=True,
                )
            raise
        return CancelExecutionResult(execution_id, True)

    async def _resolve_cancel_race(
        self,
        execution_id: str,
        tenant_id: str,
        operation: OperationLedgerRecord,
    ) -> "CancelExecutionResult | None":
        current = await self._state.executions.get(execution_id, tenant_id=tenant_id)
        if current is None or current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return None
        current_operation = await self._state.operations.get(operation.operation_id, tenant_id=tenant_id)
        if current_operation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for attempt in range(2):
            if current_operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                break
            if current_operation.status is OperationStatus.FAILED:
                raise _stable_operation_error(current_operation.error_code)
            if current_operation.status not in {OperationStatus.PENDING, OperationStatus.EFFECT_UNKNOWN}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            result = await self._state.executions.get_result(execution_id, tenant_id=tenant_id)
            result_digest = None if result is None or result.object_ref is None else result.object_ref.digest
            try:
                current_operation = await self._state.operations.compare_and_swap(
                    operation.operation_id,
                    tenant_id=tenant_id,
                    expected_status=current_operation.status,
                    next_record=_operation_result(current_operation, current.execution_id, result_digest),
                )
                break
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                reloaded = await self._state.operations.get(operation.operation_id, tenant_id=tenant_id)
                if reloaded is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current_operation = reloaded
                if attempt == 1 and current_operation.status in {OperationStatus.PENDING, OperationStatus.EFFECT_UNKNOWN}:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.debug(
            "execution cancellation race resolved: execution=%s status=%s operation=%s",
            execution_id,
            current.status.value,
            operation.operation_id,
        )
        return CancelExecutionResult(execution_id, current.status is ExecutionStatus.CANCELLED)


    @_observed_query
    async def trace(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> "Page[ExecutionTraceItem]":
        record = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return await self._history_reader.trace(record.execution_id, tenant_id=record.tenant_id, cursor=cursor, limit=limit)

    @_observed_query
    async def transcript(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> Page[TranscriptItem]:
        record = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return await self._history_reader.transcript(record.execution_id, tenant_id=record.tenant_id, cursor=cursor, limit=limit)

    @_observed_query
    async def history(self, execution_id: str, *, principal: Principal, cursor: "str | None" = None, limit: int = 100) -> "Page[ExecutionHistoryItem]":
        record = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        return await self._history_reader.history(record.execution_id, tenant_id=record.tenant_id, cursor=cursor, limit=limit)

    async def _load_authorized(self, execution_id: str, principal: Principal, action: AuthorizationAction) -> ExecutionRecord:
        header = await self._state.executions.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._state.executions.get(execution_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


def _request_digest(
    request: ExecutionRequest,
    binding_digest: str,
    *,
    session_id: str | None,
    source_execution_id: str | None,
    base_execution_id: str | None,
    parent_execution_id: str | None,
    root_execution_id: str | None,
    lineage_kind: ExecutionLineageKind,
) -> str:
    return canonical_sha256(
        {
            "user_prompt": request.user_prompt,
            "binding_digest": binding_digest,
            "scope": session_id or "execution",
            "principal": principal_identity_payload(request.principal),
            "session_id": session_id,
            "source_execution_id": source_execution_id,
            "base_execution_id": base_execution_id,
            "parent_execution_id": parent_execution_id,
            "root_identity": root_execution_id or "$self",
            "lineage_kind": lineage_kind.value,
            "memory_scope_digest": None if request.memory_scope is None else canonical_sha256(request.memory_scope),
        }
    )


def _operation_result(
    operation: OperationLedgerRecord,
    result_ref: str,
    result_digest: str | None,
) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        operation.operation_id,
        operation.tenant_id,
        operation.resource_kind,
        operation.resource_id,
        operation.execution_id,
        operation.operation_kind,
        OperationStatus.SUCCEEDED,
        operation.request_digest,
        result_ref,
        result_digest,
        None,
        operation.compactable,
        operation.sequence,
        operation.created_at,
        datetime.now(timezone.utc),
    )


def _operation_failure(operation: OperationLedgerRecord, error_code: str) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        operation.operation_id,
        operation.tenant_id,
        operation.resource_kind,
        operation.resource_id,
        operation.execution_id,
        operation.operation_kind,
        OperationStatus.FAILED,
        operation.request_digest,
        None,
        None,
        error_code,
        operation.compactable,
        operation.sequence,
        operation.created_at,
        datetime.now(timezone.utc),
    )


def _operation_status(operation: OperationLedgerRecord, status: OperationStatus) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        operation.operation_id,
        operation.tenant_id,
        operation.resource_kind,
        operation.resource_id,
        operation.execution_id,
        operation.operation_kind,
        status,
        operation.request_digest,
        operation.result_ref,
        operation.result_digest,
        operation.error_code,
        operation.compactable,
        operation.sequence,
        operation.created_at,
        datetime.now(timezone.utc),
    )


def _next_execution(record: ExecutionRecord, status: ExecutionStatus, now: datetime, *, error_code: "str | None" = None, agent_run_sequence: int | None = None, terminal_event: bool = False) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=record.execution_id,
        tenant_id=record.tenant_id,
        session_id=record.session_id,
        binding_digest=record.binding_digest,
        parent_execution_id=record.parent_execution_id,
        root_execution_id=record.root_execution_id,
        source_execution_id=record.source_execution_id,
        base_execution_id=record.base_execution_id,
        lineage_kind=record.lineage_kind,
        status=status,
        revision=record.revision + 1,
        event_sequence=record.event_sequence + (1 if terminal_event else 0),
        agent_run_sequence=record.agent_run_sequence if agent_run_sequence is None else agent_run_sequence,
        error_code=error_code,
        safe_error_details=record.safe_error_details,
        created_at=record.created_at,
        updated_at=now,
        memory_scope=record.memory_scope,
        conversation_step_run_id=record.conversation_step_run_id,
    )


def _stable_idempotency_error(error_code: str | None, fallback: ErrorCode) -> AIError:
    try:
        return AIError(fallback if error_code is None else ErrorCode(error_code))
    except ValueError:
        return AIError(fallback)


def _stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["CancelEffectOutcome", "DefaultExecutionService", "ExecutionApi", "ExecutionBackend", "ExecutionQueryApi"]
