#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution query API and the persistence-backed default service."""

import asyncio
import json
import re
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from functools import wraps
from typing import Protocol

from linktools.core import environ

from ..agent import AgentBinding, AgentBindingSnapshot, AgentCatalog, AgentCompiler
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
    principal_identity_payload,
)
from ..core import (
    idempotency_key_digest as compute_idempotency_key_digest,
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
    ExecutionStartReservation,
    ExecutionStartUnknownCommit,
    ExecutionState,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
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


class _ExecutionTerminalCommitter(Protocol):
    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None,
    ) -> ExecutionTerminalCommitResult: ...


class _SubagentCancellation(Protocol):
    async def cancel_children(self, parent_execution_id: str, principal: Principal) -> None: ...


class _LaunchGate(Protocol):
    async def __call__(self, execution_id: str) -> None: ...


class _LocalExecutionWaiter(Protocol):
    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool: ...

    async def wait_terminal(self, execution_id: str, *, tenant_id: str) -> None: ...


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


@dataclass
class _SessionLockEntry:
    lock: asyncio.Lock
    ref_count: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionStartIdentity:
    scope: str
    idempotency_key_digest: str
    request_digest: str


class ExecutionBackend(Protocol):
    async def prepare_start(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        identity: ExecutionStartIdentity,
    ) -> ExecutionRecord: ...
    async def abort_start(self, execution: ExecutionRecord) -> None: ...
    async def launch(self, request: ExecutionRequest, execution: ExecutionRecord) -> None: ...

    async def commit_cancel_checkpoint(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus,
    ) -> ExecutionRecord: ...
    async def cancel(self, execution: ExecutionRecord) -> "CancelEffectOutcome": ...
    def worker_failure(self, execution_id: str, *, tenant_id: str) -> AIError | None: ...
    def worker_installed(self, execution_id: str) -> bool: ...


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
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        backend: "ExecutionBackend | None" = None,
        operation_ids: "Callable[[], str] | None" = None,
        history_reader: ExecutionHistoryReader,
        release_terminal: _ExecutionReleaseCallback | None = None,
        terminal_verifier: "_ExecutionTerminalVerifier | None" = None,
        local_waiter: "_LocalExecutionWaiter | None" = None,
    ) -> None:
        self._state = state
        self._object_store = object_store
        self._sessions = sessions
        self._catalog = catalog
        self._compiler = compiler
        self._authorization = authorization
        self._backend = backend
        self._operation_ids = operation_ids or (lambda: uuid.uuid4().hex)
        self._history_reader = history_reader
        self._release_terminal = release_terminal or _no_release_terminal
        self._terminal_verifier = terminal_verifier or _missing_terminal_verifier
        self._terminal_verifier_is_default = terminal_verifier is None
        self._terminal_committer: _ExecutionTerminalCommitter | None = None
        self._local_waiter = local_waiter
        self._subagent_cancellation: _SubagentCancellation | None = None
        self._session_locks: dict[tuple[str, str], _SessionLockEntry] = {}
        self._session_locks_guard = asyncio.Lock()
        self._handoff_states: dict[tuple[str, str], _ExecutionHandoffState] = {}
        self._handoff_condition = asyncio.Condition()

    def bind_backend(self, backend: ExecutionBackend) -> None:
        if backend is None:
            raise ValueError("execution backend is required")
        if self._backend is not None:
            raise RuntimeError("execution backend is already bound")
        self._backend = backend

    def bind_terminal_verifier(self, verifier: _ExecutionTerminalVerifier) -> None:
        if verifier is None:
            raise ValueError("terminal verifier is required")
        if not self._terminal_verifier_is_default:
            raise RuntimeError("terminal verifier is already bound")
        self._terminal_verifier = verifier
        self._terminal_verifier_is_default = False

    def bind_terminal_committer(self, committer: _ExecutionTerminalCommitter) -> None:
        if committer is None:
            raise ValueError("terminal committer is required")
        if self._terminal_committer is not None:
            raise RuntimeError("terminal committer is already bound")
        self._terminal_committer = committer

    def bind_subagent_cancellation(self, cancellation: _SubagentCancellation) -> None:
        if self._subagent_cancellation is not None:
            raise RuntimeError("subagent cancellation is already bound")
        self._subagent_cancellation = cancellation

    def bind_local_waiter(self, waiter: _LocalExecutionWaiter) -> None:
        if self._local_waiter is not None:
            raise RuntimeError("local execution waiter is already bound")
        self._local_waiter = waiter

    def _binding(
        self,
        binding_digest: str,
        snapshot: "AgentBindingSnapshot | None" = None,
    ) -> AgentBinding:
        try:
            binding = self._catalog.binding(binding_digest)
        except AIError as error:
            if error.code is not ErrorCode.AGENT_DEFINITION_UNAVAILABLE or snapshot is None:
                raise
            if snapshot.binding_digest != binding_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            binding = self._catalog.register_binding(self._compiler.restore(snapshot))
        if snapshot is not None and binding.snapshot != snapshot:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return binding

    def _validate_replayed_execution(
        self,
        execution: ExecutionRecord,
        binding: AgentBinding,
        request: ExecutionRequest,
    ) -> None:
        if (
            execution.binding_digest != binding.digest
            or execution.planning is not request.planning
            or execution.thinking is not request.thinking
            or execution.binding != binding.snapshot
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def acquire_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> None:
        if not hold_id:
            raise ValueError("execution dependency hold id is required")
        async with self._handoff_condition:
            state = self._handoff_states.setdefault((tenant_id, execution_id), _ExecutionHandoffState())
            while state.release_in_progress:
                await self._handoff_condition.wait()
            state.dependency_holds.add(hold_id)

    async def release_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> None:
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

    async def request_terminal_handoff(self, execution_id: str, *, tenant_id: str) -> None:
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
            await self.request_terminal_handoff(execution_id, tenant_id=tenant_id)

    def _claim_cleanup_locked(self, state: _ExecutionHandoffState) -> bool:
        if state.active_consumers == 0 and not state.dependency_holds and state.release_requested and not state.release_in_progress:
            state.release_in_progress = True
            return True
        return False

    async def _run_handoff_cleanup(self, execution_id: str, tenant_id: str, state: _ExecutionHandoffState) -> None:
        try:
            await self._release_terminal(execution_id, tenant_id=tenant_id)
        except BaseException as error:
            async with self._handoff_condition:
                state.release_in_progress = False
                state.release_requested = True
                self._handoff_condition.notify_all()
            if not isinstance(error, Exception):
                raise
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

    async def run_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
    ) -> ExecutionHandle:
        if not session_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._start(
            binding_digest,
            request,
            session_id=session_id,
            session_agent_id=agent_id,
            scope="session.resume",
        )

    async def _run_with_launch_gate(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        gate: _LaunchGate,
    ) -> ExecutionHandle:
        return await self._start(binding_digest, request, scope="execution.run", launch_gate=gate)

    async def _run_for_session_with_launch_gate(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        gate: _LaunchGate,
    ) -> ExecutionHandle:
        if not session_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._start(
            binding_digest,
            request,
            session_id=session_id,
            session_agent_id=agent_id,
            scope="session.resume",
            launch_gate=gate,
        )

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

    async def _start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        session_id: "str | None" = None,
        session_agent_id: "str | None" = None,
        source_execution_id: "str | None" = None,
        base_execution_id: "str | None" = None,
        conversation_step_run_id: "str | None" = None,
        parent_execution_id: "str | None" = None,
        root_execution_id: "str | None" = None,
        lineage_kind: ExecutionLineageKind = ExecutionLineageKind.RUN,
        scope: str = "execution.run",
        launch_gate: _LaunchGate | None = None,
    ) -> ExecutionHandle:
        if session_id is None:
            return await self._start_unlocked(
                binding_digest,
                request,
                session_id=session_id,
                session_agent_id=session_agent_id,
                source_execution_id=source_execution_id,
                base_execution_id=base_execution_id,
                conversation_step_run_id=conversation_step_run_id,
                parent_execution_id=parent_execution_id,
                root_execution_id=root_execution_id,
                lineage_kind=lineage_kind,
                scope=scope,
                launch_gate=launch_gate,
            )
        async with self._session_guard(request.principal.tenant_id, session_id):
            return await self._start_unlocked(
                binding_digest,
                request,
                session_id=session_id,
                session_agent_id=session_agent_id,
                source_execution_id=source_execution_id,
                base_execution_id=base_execution_id,
                conversation_step_run_id=conversation_step_run_id,
                parent_execution_id=parent_execution_id,
                root_execution_id=root_execution_id,
                lineage_kind=lineage_kind,
                scope=scope,
                launch_gate=launch_gate,
            )

    async def _start_unlocked(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        session_id: "str | None" = None,
        session_agent_id: "str | None" = None,
        source_execution_id: "str | None" = None,
        base_execution_id: "str | None" = None,
        conversation_step_run_id: "str | None" = None,
        parent_execution_id: "str | None" = None,
        root_execution_id: "str | None" = None,
        lineage_kind: ExecutionLineageKind = ExecutionLineageKind.RUN,
        scope: str = "execution.run",
        launch_gate: _LaunchGate | None = None,
    ) -> ExecutionHandle:
        if re.fullmatch(r"[0-9a-f]{64}", binding_digest) is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if request.idempotency_key is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        binding = self._binding(binding_digest)
        conversation_run_id = conversation_step_run_id
        if session_id is not None and source_execution_id is None:
            session = await self._sessions.get(session_id, tenant_id=request.principal.tenant_id)
            if session is None:
                raise AIError(ErrorCode.SESSION_NOT_FOUND)
            from ._session import _session_agent_id

            if _session_agent_id(session) != session_agent_id:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
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
                if pending is not None:
                    self._validate_replayed_execution(pending, binding, request)
                if pending is not None and pending.status is ExecutionStatus.STARTED:
                    await self._launch_started(
                        request,
                        pending,
                        scope=scope,
                        idempotency_key_digest=idempotency_key_digest,
                        launch_gate=launch_gate,
                    )
                    return ExecutionHandle(existing.resource_id)
                if pending is None or pending.status is not ExecutionStatus.PENDING_START:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._prepare_and_launch(
                    request,
                    pending,
                    scope=scope,
                    idempotency_key_digest=idempotency_key_digest,
                    request_digest=request_digest,
                    launch_gate=launch_gate,
                )
                return ExecutionHandle(existing.resource_id)
            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_START_PERSISTENCE_FAILED)
            if existing.status is IdempotencyStatus.CANCELLED:
                raise _stable_idempotency_error(existing.error_code, ErrorCode.EXECUTION_CANCELLED)
            started = await self._state.executions.get(existing.resource_id, tenant_id=request.principal.tenant_id)
            if started is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_replayed_execution(started, binding, request)
            if existing.status is IdempotencyStatus.COMPLETED:
                if started.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return ExecutionHandle(existing.resource_id)
            if existing.status is IdempotencyStatus.STARTED and started.status is ExecutionStatus.FINALIZING:
                return ExecutionHandle(existing.resource_id)
            if started.status is not ExecutionStatus.STARTED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._launch_started(
                request,
                started,
                scope=scope,
                idempotency_key_digest=idempotency_key_digest,
                launch_gate=launch_gate,
            )
            return ExecutionHandle(existing.resource_id)
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
            planning=request.planning,
            thinking=request.thinking,
            binding=binding.snapshot,
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
            self._validate_replayed_execution(reservation.execution, binding, request)
            if reservation.execution.status is ExecutionStatus.PENDING_START and reservation.idempotency.status is IdempotencyStatus.RESERVED:
                await self._prepare_and_launch(
                    request,
                    reservation.execution,
                    scope=scope,
                    idempotency_key_digest=idempotency_key_digest,
                    request_digest=request_digest,
                    launch_gate=launch_gate,
                )
            elif reservation.idempotency.status is IdempotencyStatus.STARTED and reservation.execution.status in {
                ExecutionStatus.STARTED,
                ExecutionStatus.FINALIZING,
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                await self._launch_started(
                    request,
                    reservation.execution,
                    scope=scope,
                    idempotency_key_digest=idempotency_key_digest,
                    launch_gate=launch_gate,
                )
            elif reservation.execution.status is ExecutionStatus.START_UNKNOWN or reservation.idempotency.status is IdempotencyStatus.START_UNKNOWN:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ExecutionHandle(reservation.execution.execution_id)
        execution_id = reservation.execution.execution_id
        await self._prepare_and_launch(
            request,
            reservation.execution,
            scope=scope,
            idempotency_key_digest=idempotency_key_digest,
            request_digest=request_digest,
            launch_gate=launch_gate,
        )
        _logger.info("execution started: execution=%s scope=%s", execution_id, scope)
        return ExecutionHandle(execution_id)

    @asynccontextmanager
    async def _session_guard(self, tenant_id: str, session_id: str):
        key = tenant_id, session_id
        async with self._session_locks_guard:
            entry = self._session_locks.get(key)
            if entry is None:
                entry = _SessionLockEntry(asyncio.Lock())
                self._session_locks[key] = entry
            entry.ref_count += 1
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            async with self._session_locks_guard:
                entry.ref_count -= 1
                if entry.ref_count == 0 and self._session_locks.get(key) is entry:
                    self._session_locks.pop(key)

    async def _prepare_and_launch(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        *,
        scope: str,
        idempotency_key_digest: str,
        request_digest: str,
        launch_gate: _LaunchGate | None = None,
    ) -> None:
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        identity = ExecutionStartIdentity(scope, idempotency_key_digest, request_digest)
        try:
            started = await self._backend.prepare_start(request, execution, identity)
        except AIError as error:
            if error.code in {ErrorCode.SESSION_BUSY, ErrorCode.SESSION_CONFLICT}:
                await self._reject_pending_start(
                    execution,
                    identity=identity,
                    error=error,
                )
            raise
        if started is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._launch_started(
            request,
            started,
            scope=scope,
            idempotency_key_digest=idempotency_key_digest,
            launch_gate=launch_gate,
        )

    async def _reject_pending_start(
        self,
        execution: ExecutionRecord,
        *,
        identity: ExecutionStartIdentity,
        error: AIError,
    ) -> ExecutionRecord:
        if error.code not in {ErrorCode.SESSION_BUSY, ErrorCode.SESSION_CONFLICT}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = await self._state.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        idempotency = await self._state.idempotency.get(
            identity.scope,
            identity.idempotency_key_digest,
            tenant_id=current.tenant_id,
        )
        if (
            idempotency is None
            or idempotency.runtime_domain is not RuntimeDomain.EXECUTION
            or idempotency.scope != identity.scope
            or idempotency.idempotency_key_digest != identity.idempotency_key_digest
            or idempotency.resource_kind is not ResourceKind.EXECUTION
            or idempotency.resource_id != current.execution_id
            or idempotency.tenant_id != current.tenant_id
            or idempotency.request_digest != identity.request_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            if (
                current.status is not ExecutionStatus.FAILED
                or current.error_code != error.code.value
                or idempotency.status is not IdempotencyStatus.FAILED
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            terminal = current
        else:
            if current.status is not ExecutionStatus.PENDING_START:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if idempotency.status is not IdempotencyStatus.RESERVED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            now = datetime.now(timezone.utc)
            terminal = _next_execution(
                current,
                ExecutionStatus.FAILED,
                now,
                error_code=error.code.value,
                safe_error_details=error.safe_details,
                terminal_event=True,
            )
            if self._terminal_committer is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            try:
                await self._terminal_committer.commit_terminal_checkpoint(
                    ExecutionTerminalCommit(
                        expected_revision=current.revision,
                        expected_event_sequence=current.event_sequence,
                        execution=terminal,
                        result=ResultRecord(
                            current.execution_id,
                            current.tenant_id,
                            None,
                            None,
                            None,
                            None,
                            StopReason.ERROR,
                            UsageMetrics(),
                            now,
                        ),
                        terminal_event_type=ExecutionEventType.EXECUTION_FAILED,
                        terminal_event_payload={
                            "error_code": error.code.value,
                            "safe_error_details": dict(error.safe_details),
                        },
                        idempotency=IdempotencyTerminalUpdate(
                            idempotency.scope,
                            idempotency.idempotency_key_digest,
                            idempotency.status,
                            IdempotencyStatus.FAILED,
                            idempotency.request_digest,
                            None,
                            error.code.value,
                        ),
                    ),
                    session_id=None,
                )
            except AIError as commit_error:
                if commit_error.code not in {
                    ErrorCode.STORAGE_CONFLICT,
                    ErrorCode.EXECUTION_RESULT_CONFLICT,
                }:
                    raise
                latest = await self._state.executions.get(
                    current.execution_id,
                    tenant_id=current.tenant_id,
                )
                if (
                    latest is None
                    or latest.status is not ExecutionStatus.FAILED
                    or latest.error_code != error.code.value
                ):
                    raise
                terminal = latest
        try:
            if self._backend is not None:
                await self._backend.abort_start(terminal)
        except Exception:
            _logger.error(
                "pending start cleanup failed: execution=%s",
                terminal.execution_id,
                exc_info=environ.debug,
            )
        _logger.info(
            "pending start rejected: execution=%s code=%s",
            terminal.execution_id,
            error.code.value,
        )
        return terminal

    async def _launch_started(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        *,
        scope: str,
        idempotency_key_digest: str,
        launch_gate: _LaunchGate | None = None,
    ) -> None:
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        launch_record = await self._state.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if launch_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if launch_record.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.CANCELLING,
            ExecutionStatus.FINALIZING,
        }:
            return
        if launch_record.status is ExecutionStatus.START_UNKNOWN:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        if launch_record.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            if launch_gate is not None:
                await launch_gate(launch_record.execution_id)
            await self._backend.launch(request, launch_record)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            worker_installed = False
            try:
                worker_installed = self._backend.worker_installed(launch_record.execution_id)
            except AttributeError:
                pass
            if worker_installed:
                _logger.warning(
                    "execution launch raised after worker installation: execution=%s",
                    execution.execution_id,
                    exc_info=environ.debug,
                )
                return
            current = await self._state.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
            if current is not None and current.status is ExecutionStatus.STARTED:
                identity = await self._state.idempotency.get(
                    scope,
                    idempotency_key_digest,
                    tenant_id=execution.tenant_id,
                )
                if identity is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._state.executions.mark_start_unknown(
                    ExecutionStartUnknownCommit(
                        execution.execution_id,
                        execution.tenant_id,
                        current.revision,
                        current.event_sequence,
                        scope,
                        idempotency_key_digest,
                        identity.request_digest,
                        datetime.now(timezone.utc),
                    )
                )
            _logger.error("execution start outcome unknown: execution=%s", execution.execution_id)
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN) from error

    @_observed_query
    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        if execution.status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } and self._backend is not None:
            failure = self._backend.worker_failure(
                execution_id,
                tenant_id=principal.tenant_id,
            )
            if failure is not None:
                raise failure
        return ExecutionView(execution.execution_id, execution.status)

    @_consumed_query
    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        execution = await self._load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        if execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise AIError(ErrorCode.EXECUTION_NOT_READY)
        result = await self._state.executions.get_result(execution_id, tenant_id=principal.tenant_id)
        if result is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        error_code, safe_details = _terminal_error(execution)
        if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            if result.output is not None or any(
                value is not None
                for value in (
                    result.output_schema_id,
                    result.output_schema_revision,
                    result.output_schema_fingerprint,
                )
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                return ExecutionResult(
                    execution.execution_id,
                    execution.status,
                    None,
                    result.output_schema_id,
                    result.output_schema_revision,
                    result.output_schema_fingerprint,
                    result.usage,
                    error_code,
                    safe_details,
                )
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            if result.output.kind == "inline":
                decoded = result.output.decode()
                if not isinstance(decoded, (str, dict, list, int, float, bool, type(None))):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                output = decoded
            else:
                reference = result.output.ref
                if reference is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                payload = await read_object(
                    self._object_store,
                    reference.key,
                    expected_digest=reference.digest,
                    expected_size=reference.size,
                )
                if result.output.encoding == "utf-8":
                    output = payload.decode("utf-8")
                else:
                    output = json.loads(payload.decode("utf-8"))
            return ExecutionResult(
                execution.execution_id,
                execution.status,
                output,
                result.output_schema_id,
                result.output_schema_revision,
                result.output_schema_fingerprint,
                result.usage,
                error_code,
                safe_details,
            )
        except AIError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    @_consumed_query
    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> ExecutionResult:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def wait_once() -> ExecutionResult:
            while True:
                view = await self.inspect(execution_id, principal=principal)
                waiter = self._local_waiter
                owns_execution = waiter is not None and waiter.owns_execution(
                    execution_id,
                    tenant_id=principal.tenant_id,
                )
                if view.status in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    if owns_execution:
                        _logger.debug(
                            "execution wait awaiting local terminal worker: execution=%s",
                            execution_id,
                        )
                        await waiter.wait_terminal(
                            execution_id,
                            tenant_id=principal.tenant_id,
                        )
                        continue
                    return await self.result(execution_id, principal=principal)
                if self._backend is not None:
                    failure = self._backend.worker_failure(
                        execution_id,
                        tenant_id=principal.tenant_id,
                    )
                    if failure is not None:
                        raise failure
                if owns_execution:
                    await waiter.wait_terminal(
                        execution_id,
                        tenant_id=principal.tenant_id,
                    )
                else:
                    await asyncio.sleep(1.0)

        try:
            return await asyncio.wait_for(wait_once(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.EXECUTION_WAIT_TIMEOUT) from error

    async def run_and_wait(self, binding_digest: str, request: ExecutionRequest, *, timeout_seconds: "float | None" = None) -> ExecutionResult:
        handle = await self.run(binding_digest, request)
        return await self.wait(handle.execution_id, principal=request.principal, timeout_seconds=timeout_seconds)

    async def retry(self, binding_digest: str, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding_digest:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        binding = self._binding(previous.binding_digest, previous.binding)
        if binding.digest != binding_digest:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        retry_request = ExecutionRequest(
            user_prompt=request.user_prompt,
            principal=request.principal,
            idempotency_key=request.idempotency_key,
            memory_scope=previous.memory_scope,
            planning=previous.planning,
            thinking=previous.thinking,
        )
        return await self._start(
            binding_digest,
            retry_request,
            session_id=previous.session_id,
            parent_execution_id=previous.parent_execution_id,
            root_execution_id=previous.root_execution_id,
            scope="execution.retry",
            source_execution_id=previous.execution_id,
            lineage_kind=ExecutionLineageKind.RETRY,
            base_execution_id=previous.base_execution_id,
            conversation_step_run_id=previous.conversation_step_run_id,
        )

    async def fork(self, binding_digest: str, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        previous = await self._load_authorized(execution_id, request.principal, AuthorizationAction.EXECUTION_READ)
        if previous.binding_digest != binding_digest:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        binding = self._binding(previous.binding_digest, previous.binding)
        if binding.digest != binding_digest:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        fork_request = ExecutionRequest(
            user_prompt=request.user_prompt,
            principal=request.principal,
            idempotency_key=request.idempotency_key,
            memory_scope=previous.memory_scope,
            planning=previous.planning,
            thinking=previous.thinking,
        )
        return await self._start(
            binding_digest,
            fork_request,
            session_id=previous.session_id,
            parent_execution_id=previous.parent_execution_id,
            root_execution_id=previous.root_execution_id,
            scope="execution.fork",
            source_execution_id=previous.execution_id,
            lineage_kind=ExecutionLineageKind.FORK,
            base_execution_id=previous.execution_id,
        )

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
        latest = await self._state.executions.get(
            execution_id,
            tenant_id=request.principal.tenant_id,
        )
        if latest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = latest
        prepared_before_claim = execution.status is ExecutionStatus.PENDING_START
        if execution.status is ExecutionStatus.FINALIZING:
            resolved = await self._resolve_cancel_race(
                execution_id,
                request.principal.tenant_id,
                operation,
            )
            if resolved is not None:
                return resolved
        if execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
            if resolved is not None:
                return resolved
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            try:
                cancelling = await self._backend.commit_cancel_checkpoint(
                    ExecutionCancelRequestCommit(
                        execution_id=execution_id,
                        tenant_id=request.principal.tenant_id,
                        expected_revision=execution.revision,
                        expected_event_sequence=execution.event_sequence,
                        operation_id=operation.operation_id,
                        requested_at=datetime.now(timezone.utc),
                    ),
                    expected_status=execution.status,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                raise
            if cancelling.status is not ExecutionStatus.CANCELLING:
                resolved = await self._resolve_cancel_race(
                    execution_id,
                    request.principal.tenant_id,
                    operation,
                )
                if resolved is not None:
                    return resolved
                raise AIError(ErrorCode.STORAGE_CONFLICT)
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
            terminal = _next_execution(
                cancelling_current,
                ExecutionStatus.CANCELLED,
                now,
                error_code=ErrorCode.EXECUTION_CANCELLED.value,
                safe_error_details={},
                terminal_event=True,
            )
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
            terminal_commit = ExecutionTerminalCommit(
                expected_revision=cancelling_current.revision,
                expected_event_sequence=cancelling_current.event_sequence,
                execution=terminal,
                result=result,
                terminal_event_type=ExecutionEventType.EXECUTION_CANCELLED,
                terminal_event_payload={
                    "error_code": ErrorCode.EXECUTION_CANCELLED.value,
                    "safe_error_details": {},
                },
                idempotency=idempotency,
                operation=operation_update,
            )
            if self._terminal_committer is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            try:
                await self._terminal_committer.commit_terminal_checkpoint(
                    terminal_commit,
                    session_id=cancelling_current.session_id,
                )
            except AIError as error:
                if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.EXECUTION_RESULT_CONFLICT}:
                    raise
                resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
                if resolved is not None:
                    return resolved
                raise
            if prepared_before_claim:
                try:
                    await self._backend.abort_start(terminal)
                except Exception:
                    _logger.error(
                        "pending cancellation cleanup failed: execution=%s",
                        execution_id,
                        exc_info=environ.debug,
                    )
            _logger.info("execution cancelled: execution=%s operation=%s", execution_id, operation.operation_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            resolved = await self._resolve_cancel_race(execution_id, request.principal.tenant_id, operation)
            if resolved is not None:
                return resolved
            if isinstance(error, AIError) and error.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.EXECUTION_RESULT_CONFLICT}:
                raise
            error_code = error.code.value if isinstance(error, AIError) else ErrorCode.INTERNAL_ERROR.value
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
                _logger.exception(
                    "execution cancellation ledger update failed: execution=%s operation=%s",
                    execution_id,
                    operation.operation_id,
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
        if current is None or current.status not in {
            ExecutionStatus.FINALIZING,
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
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
            if current.status is ExecutionStatus.FINALIZING:
                result_ref = current.execution_id
                result_digest = None
            else:
                result = await self._state.executions.get_result(execution_id, tenant_id=tenant_id)
                result_ref = current.execution_id
                result_digest = None if result is None or result.output is None else result.output.digest
            try:
                current_operation = await self._state.operations.compare_and_swap(
                    operation.operation_id,
                    tenant_id=tenant_id,
                    expected_status=current_operation.status,
                    next_record=_operation_result(current_operation, result_ref, result_digest),
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


def _terminal_error(
    execution: ExecutionRecord,
) -> "tuple[str | None, Mapping[str, JsonValue]]":
    details = dict(execution.safe_error_details)
    if execution.status is ExecutionStatus.SUCCEEDED:
        if execution.error_code is not None or details:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return None, {}
    if execution.status is ExecutionStatus.CANCELLED:
        if execution.error_code != ErrorCode.EXECUTION_CANCELLED.value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return execution.error_code, details
    if execution.status is not ExecutionStatus.FAILED or execution.error_code is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(execution.error_code)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if code is ErrorCode.EXECUTION_CANCELLED:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return code.value, details


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
            "planning": request.planning,
            "thinking": request.thinking,
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


def _next_execution(
    record: ExecutionRecord,
    status: ExecutionStatus,
    now: datetime,
    *,
    error_code: "str | None" = None,
    safe_error_details: "Mapping[str, JsonValue] | None" = None,
    agent_run_sequence: int | None = None,
    terminal_event: bool = False,
) -> ExecutionRecord:
    return replace(
        record,
        status=status,
        revision=record.revision + 1,
        event_sequence=record.event_sequence + (1 if terminal_event else 0),
        agent_run_sequence=record.agent_run_sequence if agent_run_sequence is None else agent_run_sequence,
        error_code=error_code,
        safe_error_details=(
            record.safe_error_details
            if safe_error_details is None
            else dict(safe_error_details)
        ),
        updated_at=now,
    )


def _stable_idempotency_error(error_code: str | None, fallback: ErrorCode) -> AIError:
    if error_code is None:
        return AIError(fallback)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["CancelEffectOutcome", "DefaultExecutionService", "ExecutionBackend"]
