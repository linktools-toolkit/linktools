#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference runtime persistence for in-memory and filesystem backends."""

import copy
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone

from linktools.core import environ

from ...core import (
    ApprovalDecision,
    ApprovalStatus,
    ExecutionEventType,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    JsonValue,
    OperationStatus,
    Page,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    TaskStatus,
    ToolOperationStatus,
    canonical_sha256,
    validate_lease_owner,
    validate_observation_payload,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ...errors import AIError, ErrorCode
from ...storage import (
    ObjectRef,
)
from ...task import TaskGraph, TaskGraphView, TaskTerminalRecord
from .._tool import ToolOperationRecord, ToolStateRepository
from ._contracts import (
    ApprovalRecord,
    ArtifactRecord,
    ArtifactState,
    ConversationCursor,
    ConversationState,
    EvaluationRecord,
    EvaluationState,
    ExecutionCancelRequestCommit,
    ExecutionEventRecord,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionStartUnknownCommit,
    ExecutionState,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExternalCallRecord,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    MemoryRecord,
    MemoryState,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationTerminalUpdate,
    RecoveryCheckpoint,
    RecoveryState,
    ResultRecord,
    RuntimeRepository,
    SessionRecord,
    TaskLease,
    TaskNodeView,
    TaskState,
)
from ._plan import RuntimeDomain
from ._transaction import RuntimeTransactionCoordinator, TransactionHub

_logger = environ.get_logger("ai.runtime.state.memory")


class _Base:
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        self._namespace = namespace
        self._closed = False
        self._lock = coordinator
        self._on_change: Callable[[], None] | None = None
        self._refresh_source: Callable[[], None] | None = None

    @property
    def namespace(self) -> str:
        return self._namespace

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    def _mark_changed(self) -> None:
        self._lock.mark_changed()
        if self._on_change is not None:
            self._on_change()

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._refresh_source is not None:
            self._refresh_source()

    def _check_tenant(self, tenant_id: str) -> None:
        validate_tenant_id(tenant_id)


class RuntimeTransactionBinding:
    """Bind adapter callbacks before repositories become usable."""

    def __init__(
        self,
        *,
        commit_callback: Callable[[RuntimeDomain], Awaitable[None] | None] | None = None,
        rollback_callback: Callable[[frozenset[RuntimeDomain]], Awaitable[None] | None] | None = None,
    ) -> None:
        self.components: tuple[RuntimeRepository, ...] = ()
        self.commit_callback = commit_callback
        self.rollback_callback = rollback_callback

    def snapshot(self) -> object:
        return _capture_runtime_snapshot(self.components)

    def restore(self, snapshot: object) -> None:
        _restore_runtime_snapshot(snapshot)

    async def commit(self, domain: RuntimeDomain) -> None:
        if self.commit_callback is not None:
            value = self.commit_callback(domain)
            if inspect.isawaitable(value):
                await value

    async def rollback(self, domains: frozenset[RuntimeDomain]) -> None:
        if self.rollback_callback is not None:
            value = self.rollback_callback(domains)
            if inspect.isawaitable(value):
                await value

    def bind_components(self, components: tuple[RuntimeRepository, ...]) -> None:
        self.components = components


class _SessionRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], SessionRecord] = {}

    async def create(self, record: SessionRecord) -> SessionRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            key = (record.tenant_id, record.session_id)
            if key in self._records:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            self._records[key] = record
            self._mark_changed()
            return record

    async def get_header(self, session_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, session_id))
        return None if record is None else ResourceRef(ResourceKind.SESSION, session_id, tenant_id, record.owner_principal_id)

    async def get(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, session_id))

    async def list(self, *, tenant_id: str, owner_principal_id: str | None = None) -> tuple[SessionRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        values = tuple(record for record in self._records.values() if record.tenant_id == tenant_id and (owner_principal_id is None or record.owner_principal_id == owner_principal_id))
        return tuple(sorted(values, key=lambda record: record.session_id))

    async def compare_and_swap(self, session_id: str, *, tenant_id: str, expected_revision: int, next_record: SessionRecord) -> SessionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, session_id)
            current = self._records.get(key)
            if current is None or current.revision != expected_revision:
                raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
            if next_record.revision != expected_revision + 1 or next_record.tenant_id != tenant_id:
                raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
            self._records[key] = next_record
            self._mark_changed()
            return next_record

    async def advance_continuation(self, session_id: str, *, tenant_id: str, expected: "ConversationCursor | None", next_cursor: ConversationCursor) -> SessionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._records.get((tenant_id, session_id))
            if current is None or current.continuation != expected or current.status is not SessionStatus.OPEN:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(current, continuation=next_cursor, revision=current.revision + 1, updated_at=datetime.now(timezone.utc))
            self._records[(tenant_id, session_id)] = updated
            self._mark_changed()
            return updated

class _ExecutionRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], ExecutionRecord] = {}
        self._idempotency: _IdempotencyRepository | None = None
        self._events: _EventRepository | None = None
        self._operations: _OperationRepository | None = None
        self._terminal: _TerminalCommitRepository | None = None

    def bind_start_repositories(self, idempotency: "_IdempotencyRepository", events: "_EventRepository", operations: "_OperationRepository") -> None:
        self._idempotency = idempotency
        self._events = events
        self._operations = operations

    def bind_terminal_repository(self, terminal: "_TerminalCommitRepository") -> None:
        self._terminal = terminal

    async def create(self, record: ExecutionRecord) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            key = (record.tenant_id, record.execution_id)
            if key in self._records:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[key] = record
            self._mark_changed()
            return record

    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, execution_id))
        return None if record is None else ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, execution_id))

    async def compare_and_swap(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: ExecutionRecord) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None or current.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.revision != expected_revision + 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[key] = next_record
            self._mark_changed()
            return next_record

    async def list_by_session(self, session_id: str, *, tenant_id: str, statuses: frozenset[ExecutionStatus] | None = None) -> tuple[ExecutionRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        values = tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.session_id == session_id)
        if statuses is not None:
            values = tuple(item for item in values if item.status in statuses)
        return tuple(sorted(values, key=lambda item: item.execution_id))

    async def list_children(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        values = tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.parent_execution_id == execution_id)
        return tuple(sorted(values, key=lambda item: item.execution_id))

    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(claim.tenant_id)
        if self._idempotency is None or self._events is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async with self._lock:
            key = (claim.tenant_id, claim.execution_id)
            current = self._records.get(key)
            identity = await self._idempotency.get(claim.scope, claim.key_digest, tenant_id=claim.tenant_id)
            if current is None or identity is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if current.status is not ExecutionStatus.PENDING_START or current.revision != claim.expected_revision or current.event_sequence != claim.expected_event_sequence or current.agent_run_sequence != 0:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if identity.status is not IdempotencyStatus.RESERVED or identity.runtime_domain is not RuntimeDomain.EXECUTION or identity.resource_kind is not ResourceKind.EXECUTION or identity.resource_id != claim.execution_id or identity.request_digest != claim.request_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            now = claim.started_at
            started = replace(current, status=ExecutionStatus.STARTED, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=now, agent_run_sequence=1)
            started_identity = replace(identity, status=IdempotencyStatus.STARTED, updated_at=now)
            event = ExecutionEventRecord(claim.execution_id, claim.tenant_id, claim.expected_event_sequence + 1, ExecutionEventType.EXECUTION_STARTED, {})
            self._records[key] = started
            self._idempotency._records[(claim.tenant_id, claim.scope, claim.key_digest)] = started_identity
            self._events._items.setdefault((claim.tenant_id, claim.execution_id), []).append(event)
            self._mark_changed()
            return started

    async def reserve_start(self, reservation: ExecutionStartReservation) -> ExecutionStartReservationResult:
        self._ensure_open()
        self._check_tenant(reservation.execution.tenant_id)
        if self._idempotency is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if reservation.execution.tenant_id != reservation.idempotency.tenant_id or reservation.execution.execution_id != reservation.idempotency.resource_id or reservation.idempotency.runtime_domain is not RuntimeDomain.EXECUTION or reservation.idempotency.resource_kind is not ResourceKind.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with self._lock:
            idempotency_key = (reservation.idempotency.tenant_id, reservation.idempotency.scope, reservation.idempotency.key_digest)
            existing_idempotency = self._idempotency._records.get(idempotency_key)
            if existing_idempotency is not None:
                if existing_idempotency.request_digest != reservation.idempotency.request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                existing_execution = self._records.get((existing_idempotency.tenant_id, existing_idempotency.resource_id))
                if existing_execution is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return ExecutionStartReservationResult(existing_execution, existing_idempotency, False)
            if (reservation.execution.tenant_id, reservation.execution.execution_id) in self._records:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[(reservation.execution.tenant_id, reservation.execution.execution_id)] = reservation.execution
            self._idempotency._records[idempotency_key] = reservation.idempotency
            self._mark_changed()
            return ExecutionStartReservationResult(reservation.execution, reservation.idempotency, True)

    async def claim_next_agent_run(self, execution_id: str, *, tenant_id: str, expected_revision: int, expected_agent_run_sequence: int) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None or current.status is not ExecutionStatus.STARTED or current.revision != expected_revision or current.agent_run_sequence != expected_agent_run_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(current, revision=current.revision + 1, agent_run_sequence=current.agent_run_sequence + 1, updated_at=datetime.now(timezone.utc))
            self._records[key] = updated
            self._mark_changed()
            return updated

    async def mark_start_unknown(self, commit: ExecutionStartUnknownCommit) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(commit.tenant_id)
        if self._idempotency is None or self._events is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async with self._lock:
            key = (commit.tenant_id, commit.execution_id)
            current = self._records.get(key)
            identity = await self._idempotency.get(commit.scope, commit.key_digest, tenant_id=commit.tenant_id)
            if current is None or identity is None or current.status is not ExecutionStatus.STARTED or current.revision != commit.expected_revision or identity.status is not IdempotencyStatus.STARTED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            unknown = replace(current, status=ExecutionStatus.START_UNKNOWN, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.occurred_at)
            self._records[key] = unknown
            self._idempotency._records[(commit.tenant_id, commit.scope, commit.key_digest)] = replace(identity, status=IdempotencyStatus.START_UNKNOWN, updated_at=commit.occurred_at)
            self._events._items.setdefault((commit.tenant_id, commit.execution_id), []).append(ExecutionEventRecord(commit.execution_id, commit.tenant_id, current.event_sequence + 1, ExecutionEventType.EXECUTION_START_UNKNOWN, {}))
            self._mark_changed()
            return unknown

    async def request_cancel(self, commit: ExecutionCancelRequestCommit) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(commit.tenant_id)
        if self._events is None or self._operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async with self._lock:
            key = (commit.tenant_id, commit.execution_id)
            current = self._records.get(key)
            operation = await self._operations.get(commit.operation_id, tenant_id=commit.tenant_id)
            events = self._events._items.get(key, ())
            if current is None or operation is None or operation.status is not OperationStatus.PENDING or operation.execution_id != commit.execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current.status is ExecutionStatus.CANCELLING and any(item.event_type is ExecutionEventType.CANCEL_REQUESTED for item in events):
                return current
            if current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} or current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(current, status=ExecutionStatus.CANCELLING, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.requested_at)
            self._records[key] = updated
            self._events._items.setdefault(key, []).append(ExecutionEventRecord(commit.execution_id, commit.tenant_id, commit.expected_event_sequence + 1, ExecutionEventType.CANCEL_REQUESTED, {}))
            self._mark_changed()
            return updated

    async def advance_event_sequence(self, execution_id: str, *, tenant_id: str, expected_sequence: int) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            sequence = current.event_sequence
            if sequence != expected_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(
                current,
                event_sequence=sequence + 1,
                revision=current.revision + 1,
                updated_at=datetime.now(timezone.utc),
            )
            self._records[key] = updated
            self._mark_changed()
            return updated

    async def commit_terminal(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult:
        if self._terminal is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._terminal.commit_terminal(commit)

    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        if self._terminal is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._terminal.get(execution_id, tenant_id=tenant_id)

    async def _commit_execution(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: ExecutionRecord) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None or current.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.tenant_id != tenant_id or next_record.execution_id != execution_id or next_record.revision != expected_revision + 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[key] = next_record
            self._mark_changed()
            return next_record


class _IdempotencyRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator, runtime_domain: RuntimeDomain) -> None:
        super().__init__(namespace, coordinator)
        self._runtime_domain = runtime_domain
        self._records: dict[tuple[str, str, str], IdempotencyRecord] = {}

    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        if record.runtime_domain is not self._runtime_domain:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if re.fullmatch(r"[0-9a-f]{64}", record.key_digest) is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        async with self._lock:
            key = (record.tenant_id, record.scope, record.key_digest)
            current = self._records.get(key)
            if current is None:
                self._records[key] = record
                self._mark_changed()
                return record
            if current.request_digest != record.request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return current

    async def get(self, scope: str, key_digest: str, *, tenant_id: str) -> IdempotencyRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, scope, key_digest))

    async def list_by_resource(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str) -> tuple[IdempotencyRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(sorted((record for record in self._records.values() if record.tenant_id == tenant_id and record.resource_kind is resource_kind and record.resource_id == resource_id), key=lambda record: (record.scope, record.key_digest)))

    async def compare_and_swap(self, scope: str, key_digest: str, *, tenant_id: str, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            store_key = (tenant_id, scope, key_digest)
            current = self._records.get(store_key)
            if current is None or current.status is not expected_status:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.tenant_id != tenant_id or next_record.key_digest != key_digest or next_record.runtime_domain is not self._runtime_domain:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[store_key] = next_record
            self._mark_changed()
            return next_record


class _EventRepository(_Base):
    def __init__(self, executions: _ExecutionRepository, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._executions = executions
        self._items: dict[tuple[str, str], list[ExecutionEventRecord]] = {}

    async def append(self, execution_id: str, *, tenant_id: str, expected_sequence: int, event_type: ExecutionEventType, payload: JsonValue) -> ExecutionEventRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        validate_observation_payload(payload)
        async with self._lock:
            items = self._items.setdefault((tenant_id, execution_id), [])
            sequence = items[-1].sequence if items else 0
            if sequence != expected_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            item = ExecutionEventRecord(execution_id, tenant_id, sequence + 1, event_type, payload)
            await self._executions.advance_event_sequence(execution_id, tenant_id=tenant_id, expected_sequence=expected_sequence)
            items.append(item)
            self._mark_changed()
            return item

    async def list(self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int) -> Page[ExecutionEventRecord]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(item for item in self._items.get((tenant_id, execution_id), ()) if item.sequence > after_sequence)
        return Page(values[:limit], str(values[limit - 1].sequence) if len(values) > limit else None)


class _TerminalCommitRepository(_Base):
    def __init__(self, executions: _ExecutionRepository, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._executions = executions
        self._results: dict[tuple[str, str], ResultRecord] = {}
        self._idempotency: _IdempotencyRepository | None = None
        self._events: _EventRepository | None = None
        self._operations: "_OperationRepository | None" = None

    def bind_terminal_repositories(self, idempotency: _IdempotencyRepository, events: _EventRepository, operations: "_OperationRepository") -> None:
        self._idempotency = idempotency
        self._events = events
        self._operations = operations

    async def commit_terminal(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult:
        self._ensure_open()
        execution = commit.execution
        self._check_tenant(execution.tenant_id)
        result = commit.result
        if (
            execution.execution_id != result.execution_id
            or execution.tenant_id != result.tenant_id
            or execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
        ):
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        _validate_terminal_result(execution, result)
        if commit.expected_event_sequence < 0 or execution.event_sequence != commit.expected_event_sequence + 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if commit.terminal_event_type not in {ExecutionEventType.EXECUTION_SUCCEEDED, ExecutionEventType.EXECUTION_FAILED, ExecutionEventType.EXECUTION_CANCELLED}:
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        if not isinstance(commit.terminal_event_payload, dict):
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        validate_observation_payload(commit.terminal_event_payload)
        if self._idempotency is None or self._events is None or self._operations is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = (execution.tenant_id, execution.execution_id)
        async with self._lock:
            current_result = self._results.get(key)
            if current_result is not None:
                current = self._executions._records.get(key)
                event = self._events._items.get(key, [])
                identity = self._find_idempotency(commit, execution)
                if (current is not None and current.status is execution.status and current_result == result and len(event) > commit.expected_event_sequence and event[commit.expected_event_sequence].event_type is commit.terminal_event_type and event[commit.expected_event_sequence].payload == commit.terminal_event_payload and self._idempotency_matches(identity, commit.idempotency)):
                    return ExecutionTerminalCommitResult(current, current_result)
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            current = self._executions._records.get(key)
            if current is None or current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            identity = self._find_idempotency(commit, execution)
            self._validate_idempotency(identity, commit.idempotency, execution)
            self._validate_operation(commit.operation, execution)
            if execution.revision != commit.expected_revision + 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._executions._records[key] = execution
            self._results[key] = result
            self._events._items.setdefault(key, []).append(ExecutionEventRecord(execution.execution_id, execution.tenant_id, commit.expected_event_sequence + 1, commit.terminal_event_type, commit.terminal_event_payload))
            if commit.idempotency is not None:
                identity_key = (execution.tenant_id, commit.idempotency.scope, commit.idempotency.key_digest)
                self._idempotency._records[identity_key] = replace(identity, status=commit.idempotency.next_status, result_digest=commit.idempotency.result_digest, error_code=commit.idempotency.error_code, updated_at=execution.updated_at)
            if commit.operation is not None:
                current_operation = self._operations._records[(execution.tenant_id, commit.operation.operation_id)]
                self._operations._records[(execution.tenant_id, commit.operation.operation_id)] = replace(current_operation, status=commit.operation.next_status, result_ref=commit.operation.result_ref, result_digest=commit.operation.result_digest, error_code=commit.operation.error_code, updated_at=execution.updated_at)
            self._mark_changed()
            return ExecutionTerminalCommitResult(execution, result)

    def _find_idempotency(self, commit: ExecutionTerminalCommit, execution: ExecutionRecord) -> IdempotencyRecord | None:
        records = tuple(record for record in self._idempotency._records.values() if record.tenant_id == execution.tenant_id and record.runtime_domain is RuntimeDomain.EXECUTION and record.resource_kind is ResourceKind.EXECUTION and record.resource_id == execution.execution_id)
        if len(records) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return records[0] if records else None

    def _validate_idempotency(self, current: IdempotencyRecord | None, update: IdempotencyTerminalUpdate | None, execution: ExecutionRecord) -> None:
        if update is None:
            if current is not None and current.status in {IdempotencyStatus.RESERVED, IdempotencyStatus.STARTED}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if current is None or current.tenant_id != execution.tenant_id or current.runtime_domain is not RuntimeDomain.EXECUTION or current.resource_kind is not ResourceKind.EXECUTION or current.resource_id != execution.execution_id or current.scope != update.scope or current.key_digest != update.key_digest or current.request_digest != update.request_digest or current.status is not update.expected_status:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _idempotency_matches(self, current: IdempotencyRecord | None, update: IdempotencyTerminalUpdate | None) -> bool:
        return update is None or (current is not None and current.status is update.next_status and current.result_digest == update.result_digest and current.error_code == update.error_code)

    def _validate_operation(self, update: OperationTerminalUpdate | None, execution: ExecutionRecord) -> None:
        if update is None:
            return
        current = self._operations._records.get((execution.tenant_id, update.operation_id))
        if current is None or current.execution_id != execution.execution_id or current.status is not update.expected_status:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def get(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._results.get((tenant_id, execution_id))


class _MemoryRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], MemoryRecord] = {}
        self._operations: _OperationRepository | None = None

    def bind_operation_repository(self, operations: "_OperationRepository") -> None:
        self._operations = operations

    async def get_header(self, memory_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, memory_id))
        return None if record is None else ResourceRef(ResourceKind.MEMORY, memory_id, tenant_id)

    async def put(self, record: MemoryRecord, *, expected_revision: int | None) -> MemoryRecord:
        stored, replayed = await self.put_with_operation(record, expected_revision=expected_revision, operation=None)
        if replayed or stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return stored

    async def put_with_operation(self, record: MemoryRecord, *, expected_revision: int | None, operation: OperationLedgerInput | None) -> "tuple[MemoryRecord | None, bool]":
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        if operation is not None and operation.tenant_id != record.tenant_id:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        async with self._lock:
            if operation is not None:
                existing = self._operation(operation)
                if existing is not None:
                    if _operation_immutable(existing) != _operation_input_immutable(operation):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    return self._records.get((record.tenant_id, record.memory_id)), True
            key = (record.tenant_id, record.memory_id)
            current = self._records.get(key)
            if current is None and expected_revision not in (None, 0):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current is not None and current.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            next_record = replace(record, revision=0 if current is None else current.revision + 1)
            self._records[key] = next_record
            if operation is not None:
                self._append_operation(operation)
            self._mark_changed()
            return next_record, False

    async def get(self, memory_id: str, *, tenant_id: str) -> MemoryRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, memory_id))

    async def list(self, *, tenant_id: str, memory_scope_key: str, cursor: "str | None", limit: int) -> "Page[MemoryRecord]":
        self._ensure_open()
        self._check_tenant(tenant_id)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(sorted((item for item in self._records.values() if item.tenant_id == tenant_id and item.memory_scope_key == memory_scope_key), key=lambda item: item.memory_id))
        start = 0 if cursor is None else next((index + 1 for index, item in enumerate(values) if item.memory_id == cursor), len(values))
        page = values[start:start + limit]
        return Page(page, page[-1].memory_id if len(values) > start + limit else None)

    async def delete(self, memory_id: str, *, tenant_id: str, expected_revision: int) -> None:
        deleted, replayed = await self.delete_with_operation(memory_id, tenant_id=tenant_id, expected_revision=expected_revision, operation=None)
        if replayed or not deleted:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def delete_with_operation(self, memory_id: str, *, tenant_id: str, expected_revision: int | None, operation: OperationLedgerInput | None) -> "tuple[bool, bool]":
        self._ensure_open()
        self._check_tenant(tenant_id)
        if operation is not None and operation.tenant_id != tenant_id:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        async with self._lock:
            if operation is not None:
                existing = self._operation(operation)
                if existing is not None:
                    if _operation_immutable(existing) != _operation_input_immutable(operation):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    return False, True
            key = (tenant_id, memory_id)
            current = self._records.get(key)
            if current is None:
                if expected_revision is not None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                deleted = False
            elif current.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            else:
                del self._records[key]
                deleted = True
            if operation is not None:
                self._append_operation(operation)
            self._mark_changed()
            return deleted, False

    def _operation(self, operation: OperationLedgerInput) -> OperationLedgerRecord | None:
        if self._operations is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._operations._records.get((operation.tenant_id, operation.operation_id))

    def _append_operation(self, record: OperationLedgerInput) -> OperationLedgerRecord:
        if self._operations is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._operations._append_locked(record, mark_changed=False)


class _ArtifactRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], ArtifactRecord] = {}

    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            current = self._records.get((record.tenant_id, record.artifact_id))
            if current is not None and current != record:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[(record.tenant_id, record.artifact_id)] = record
            self._mark_changed()
            return record

    async def get_header(self, artifact_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, artifact_id))
        return None if record is None else ResourceRef(ResourceKind.ARTIFACT, artifact_id, tenant_id)

    async def get_metadata(self, artifact_id: str, *, tenant_id: str) -> ArtifactRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, artifact_id))

    async def list_by_execution(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[ArtifactRecord]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(sorted((item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id), key=lambda item: item.artifact_id))
        start = 0 if cursor is None else next((index + 1 for index, item in enumerate(values) if item.artifact_id == cursor), len(values))
        page = values[start:start + limit]
        return Page(page, page[-1].artifact_id if len(values) > start + limit else None)


class _ApprovalRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], ApprovalRecord] = {}

    async def get_header(self, approval_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, approval_id))
        return None if record is None else ResourceRef(ResourceKind.APPROVAL, approval_id, tenant_id)

    async def create(self, record: ApprovalRecord) -> ApprovalRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            current = self._records.get((record.tenant_id, record.approval_id))
            if current is not None:
                if current == record:
                    return current
                raise AIError(ErrorCode.APPROVAL_CONFLICT)
            self._records[(record.tenant_id, record.approval_id)] = record
            self._mark_changed()
            return record

    async def get(self, approval_id: str, *, tenant_id: str) -> ApprovalRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, approval_id))

    async def decide(self, approval_id: str, *, tenant_id: str, expected_status: ApprovalStatus, idempotency_key_digest: str, decision: ApprovalDecision, principal_id: str, decision_digest: str, decided_at: datetime) -> ApprovalRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._records.get((tenant_id, approval_id))
            if current is None or current.status is not expected_status:
                if current is not None and current.idempotency_key_digest == idempotency_key_digest and current.decision is decision and current.decision_digest == decision_digest:
                    return current
                raise AIError(ErrorCode.APPROVAL_CONFLICT)
            updated = replace(current, status=ApprovalStatus.APPROVED if decision is ApprovalDecision.APPROVE else ApprovalStatus.DENIED, idempotency_key_digest=idempotency_key_digest, decision=decision, decided_by=principal_id, decision_digest=decision_digest, decided_at=decided_at)
            self._records[(tenant_id, approval_id)] = updated
            self._mark_changed()
            return updated

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id and item.status is ApprovalStatus.PENDING)


class _ExternalRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], ExternalCallRecord] = {}

    async def get_header(self, call_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, call_id))
        return None if record is None else ResourceRef(ResourceKind.EXTERNAL_CALL, call_id, tenant_id)

    async def create_call(self, record: ExternalCallRecord) -> ExternalCallRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            current = self._records.get((record.tenant_id, record.call_id))
            if current is not None:
                if current == record:
                    return current
                raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
            self._records[(record.tenant_id, record.call_id)] = record
            self._mark_changed()
            return record

    async def get(self, call_id: str, *, tenant_id: str) -> ExternalCallRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, call_id))

    async def supply(self, call_id: str, *, tenant_id: str, expected_status: ExternalCallStatus, idempotency_key_digest: str, object_ref: object, payload_digest: str, supplied_at: datetime) -> ExternalCallRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._records.get((tenant_id, call_id))
            if current is None or current.status is not expected_status:
                if current is not None and current.idempotency_key_digest == idempotency_key_digest and current.object_ref == object_ref and current.payload_digest == payload_digest:
                    return current
                raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
            updated = replace(current, status=ExternalCallStatus.SUPPLIED, idempotency_key_digest=idempotency_key_digest, object_ref=object_ref, payload_digest=payload_digest, supplied_at=supplied_at)
            self._records[(tenant_id, call_id)] = updated
            self._mark_changed()
            return updated

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ExternalCallRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id and item.status is ExternalCallStatus.PENDING)


class _RecoveryCheckpointRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], RecoveryCheckpoint] = {}

    async def create(self, record: RecoveryCheckpoint) -> RecoveryCheckpoint:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            key = record.tenant_id, record.execution_id
            current = self._records.get(key)
            if current is not None:
                if current == record:
                    return current
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[key] = record
            self._mark_changed()
            return record

    async def get(self, execution_id: str, *, tenant_id: str) -> RecoveryCheckpoint | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, execution_id))

    async def list(self, *, tenant_id: str) -> tuple[RecoveryCheckpoint, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(item for item in self._records.values() if item.tenant_id == tenant_id)

    async def compare_and_swap(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: RecoveryCheckpoint) -> RecoveryCheckpoint:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._records.get((tenant_id, execution_id))
            if current is None or current.revision != expected_revision or next_record.revision != expected_revision + 1 or next_record.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[(tenant_id, execution_id)] = next_record
            self._mark_changed()
            return next_record


class _OperationRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._runtime_domain = coordinator.owner_domain
        self._records: dict[tuple[str, str], OperationLedgerRecord] = {}
        self._counters: dict[tuple[str, str, str], int] = {}

    async def append(self, record: OperationLedgerInput) -> OperationLedgerRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            return self._append_locked(record)

    def _append_locked(self, record: OperationLedgerInput, *, mark_changed: bool = True) -> OperationLedgerRecord:
        key = (record.tenant_id, record.operation_id)
        current = self._records.get(key)
        if current is not None:
            if _operation_immutable(current) != _operation_input_immutable(record):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return current
        counter_key = (record.tenant_id, record.resource_kind.value, record.resource_id)
        sequence = self._counters.get(counter_key, 0) + 1
        created = OperationLedgerRecord(
            record.operation_id, record.tenant_id, record.resource_kind, record.resource_id,
            record.execution_id, record.operation_kind, record.status, record.request_digest,
            record.result_ref, record.result_digest, record.error_code, record.compactable,
            sequence, record.created_at, record.updated_at,
        )
        self._records[key] = created
        self._counters[counter_key] = sequence
        if mark_changed:
            self._mark_changed()
        return created

    async def get(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, operation_id))

    async def compare_and_swap(self, operation_id: str, *, tenant_id: str, expected_status: OperationStatus, next_record: OperationLedgerRecord) -> OperationLedgerRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._records.get((tenant_id, operation_id))
            if (
                current is None
                or current.status is not expected_status
                or next_record.operation_id != operation_id
                or next_record.tenant_id != tenant_id
                or next_record.resource_kind is not current.resource_kind
                or next_record.resource_id != current.resource_id
                or next_record.sequence != current.sequence
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[(tenant_id, operation_id)] = next_record
            self._mark_changed()
            return next_record

    async def list_pending(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, limit: int) -> tuple[OperationLedgerRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if limit < 1 or limit > 1000:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.resource_kind is resource_kind and item.resource_id == resource_id and item.status in {OperationStatus.PENDING, OperationStatus.RUNNING})
        return tuple(sorted(values, key=lambda item: item.sequence))[:limit]

    async def compact_terminal(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, through_sequence: int) -> str:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            values = tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.resource_kind is resource_kind and item.resource_id == resource_id and item.sequence <= through_sequence and item.compactable and item.status not in {OperationStatus.PENDING, OperationStatus.RUNNING})
            digest = canonical_sha256([asdict(item) for item in values])
            for item in values:
                self._records.pop((tenant_id, item.operation_id), None)
            self._mark_changed()
            return digest


class _TaskRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._plans: dict[tuple[str, str], TaskGraphView] = {}
        self._nodes: dict[tuple[str, str, str], TaskNodeView] = {}

    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if (tenant_id, graph_id) not in self._plans:
            return None
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def create_graph(self, graph: TaskGraph, *, tenant_id: str) -> TaskGraphView:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, graph.graph_id)
            if key in self._plans:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            view = TaskGraphView(graph.graph_id, TaskStatus.PENDING, graph.nodes)
            self._plans[key] = view
            for node in graph.nodes:
                self._nodes[(tenant_id, graph.graph_id, node.node_id)] = TaskNodeView(graph.graph_id, node.node_id, node.dependencies, TaskStatus.PENDING, None, 0, None, None, None, None)
            self._mark_changed()
            return view

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._plans.get((tenant_id, graph_id))

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._plans.get((tenant_id, graph_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            nodes = {key[2]: value for key, value in self._nodes.items() if key[:2] == (tenant_id, graph_id)}
            if current.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                changed = True
                while changed:
                    changed = False
                    for node_id in sorted(nodes):
                        node = nodes[node_id]
                        if node.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                            continue
                        dependencies = tuple(
                            nodes.get(dependency) for dependency in node.dependencies
                        )
                        if any(dependency is None for dependency in dependencies):
                            raise AIError(ErrorCode.TASK_DEPENDENCY_UNKNOWN)
                        if any(
                            dependency.status in {TaskStatus.FAILED, TaskStatus.BLOCKED}
                            for dependency in dependencies
                            if dependency is not None
                        ):
                            updated = replace(
                                node,
                                status=TaskStatus.BLOCKED,
                                error_code=ErrorCode.TASK_DEPENDENCY_FAILED.value,
                                error_digest=canonical_sha256(
                                    {
                                        "graph_id": graph_id,
                                        "node_id": node_id,
                                        "reason": "dependency_failed",
                                    }
                                ),
                            )
                        elif node.status is TaskStatus.PENDING and all(
                            dependency.status is TaskStatus.SUCCEEDED
                            for dependency in dependencies
                            if dependency is not None
                        ):
                            updated = replace(node, status=TaskStatus.READY)
                        else:
                            continue
                        if updated != node:
                            nodes[node_id] = updated
                            changed = True
                for node_id, node in nodes.items():
                    if self._nodes[(tenant_id, graph_id, node_id)] != node:
                        self._nodes[(tenant_id, graph_id, node_id)] = node
                        self._mark_changed()
                statuses = tuple(node.status for node in nodes.values())
                if not statuses:
                    status = TaskStatus.SUCCEEDED
                elif all(value in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED} for value in statuses):
                    if TaskStatus.FAILED in statuses:
                        status = TaskStatus.FAILED
                    elif TaskStatus.BLOCKED in statuses:
                        status = TaskStatus.BLOCKED
                    elif TaskStatus.CANCELLED in statuses:
                        status = TaskStatus.CANCELLED
                    else:
                        status = TaskStatus.SUCCEEDED
                elif TaskStatus.RUNNING in statuses:
                    status = TaskStatus.RUNNING
                elif TaskStatus.READY in statuses:
                    status = TaskStatus.READY
                else:
                    status = TaskStatus.PENDING
                if current.status is not status:
                    current = replace(current, status=status)
                    self._plans[(tenant_id, graph_id)] = current
                    self._mark_changed()
            else:
                for key, node in tuple(self._nodes.items()):
                    if key[:2] == (tenant_id, graph_id) and node.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                        self._nodes[key] = replace(node, status=TaskStatus.CANCELLED, owner=None, lease_expires_at=None)
                        self._mark_changed()
            _logger.debug(
                "memory task graph reconciled: graph=%s status=%s",
                graph_id,
                current.status.value,
            )
            return current

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._plans.get((tenant_id, graph_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph_status = current.status if current.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED} else TaskStatus.CANCELLED
            updated = replace(current, status=graph_status)
            self._plans[(tenant_id, graph_id)] = updated
            for key, node in tuple(self._nodes.items()):
                if key[:2] == (tenant_id, graph_id) and node.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                    self._nodes[key] = replace(node, status=TaskStatus.CANCELLED, owner=None, lease_expires_at=None)
            self._mark_changed()
            return updated

    async def claim(self, graph_id: str, node_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease:
        self._ensure_open()
        self._check_tenant(tenant_id)
        validate_lease_owner(owner)
        if not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        key = (tenant_id, graph_id, node_id)
        async with self._lock:
            plan = self._plans.get((tenant_id, graph_id))
            if plan is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if plan.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                raise AIError(ErrorCode.TASK_NOT_READY)
            current = self._nodes.get(key)
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            now = datetime.now(timezone.utc)
            reclaimable = current.status is TaskStatus.RUNNING and current.lease_expires_at is not None and current.lease_expires_at <= now
            if current.status not in {TaskStatus.PENDING, TaskStatus.READY} and not reclaimable:
                raise AIError(ErrorCode.TASK_NOT_READY)
            dependencies = tuple(self._nodes.get((tenant_id, graph_id, dependency)) for dependency in current.dependencies)
            if any(dependency is None for dependency in dependencies):
                raise AIError(ErrorCode.TASK_DEPENDENCY_UNKNOWN)
            if any(dependency.status is not TaskStatus.SUCCEEDED for dependency in dependencies if dependency is not None):
                raise AIError(ErrorCode.TASK_NOT_READY)
            fence = current.fence + 1
            expiry = now + timedelta(seconds=lease_seconds)
            self._nodes[key] = replace(current, status=TaskStatus.RUNNING, owner=owner, fence=fence, lease_expires_at=expiry)
            self._mark_changed()
            return TaskLease(graph_id, node_id, tenant_id, owner, fence, expiry)

    async def renew(self, lease: TaskLease, *, tenant_id: str, lease_seconds: int) -> TaskLease:
        self._ensure_open()
        self._check_tenant(tenant_id)
        validate_lease_owner(lease.owner)
        if lease.tenant_id != tenant_id or not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        async with self._lock:
            now = datetime.now(timezone.utc)
            current = self._nodes.get((tenant_id, lease.graph_id, lease.node_id))
            if current is None or current.status is not TaskStatus.RUNNING or current.owner != lease.owner or current.fence != lease.fence or current.lease_expires_at is None or current.lease_expires_at <= now:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            expiry = now + timedelta(seconds=lease_seconds)
            self._nodes[(tenant_id, lease.graph_id, lease.node_id)] = replace(current, lease_expires_at=expiry)
            self._mark_changed()
            return replace(lease, lease_expires_at=expiry)

    async def complete(self, lease: TaskLease, *, tenant_id: str, execution_id: "str | None", result_digest: str) -> TaskTerminalRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if lease.tenant_id != tenant_id:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        async with self._lock:
            current = self._nodes.get((tenant_id, lease.graph_id, lease.node_id))
            now = datetime.now(timezone.utc)
            if current is None or current.status is not TaskStatus.RUNNING or current.owner != lease.owner or current.fence != lease.fence or current.lease_expires_at is None or current.lease_expires_at <= now:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._nodes[(tenant_id, lease.graph_id, lease.node_id)] = replace(current, status=TaskStatus.SUCCEEDED, owner=None, result_digest=result_digest, execution_id=execution_id, lease_expires_at=None)
            self._mark_changed()
            return TaskTerminalRecord(lease.node_id, lease.owner, lease.fence, TaskStatus.SUCCEEDED, result_digest, None, None, execution_id=execution_id)

    async def fail(self, lease: TaskLease, *, tenant_id: str, error_code: str, error_digest: str) -> TaskTerminalRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if lease.tenant_id != tenant_id:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        async with self._lock:
            current = self._nodes.get((tenant_id, lease.graph_id, lease.node_id))
            now = datetime.now(timezone.utc)
            if current is None or current.status is not TaskStatus.RUNNING or current.owner != lease.owner or current.fence != lease.fence or current.lease_expires_at is None or current.lease_expires_at <= now:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._nodes[(tenant_id, lease.graph_id, lease.node_id)] = replace(current, status=TaskStatus.FAILED, owner=None, error_code=error_code, error_digest=error_digest, lease_expires_at=None)
            self._mark_changed()
            return TaskTerminalRecord(lease.node_id, lease.owner, lease.fence, TaskStatus.FAILED, None, error_code, error_digest)

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[TaskNodeView, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(sorted((item for key, item in self._nodes.items() if key[:2] == (tenant_id, graph_id)), key=lambda item: item.node_id))


class _EvaluationRepository(_Base):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], EvaluationRecord] = {}

    async def get_header(self, evaluation_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return None if (tenant_id, evaluation_id) not in self._records else ResourceRef(ResourceKind.EVALUATION, evaluation_id, tenant_id)

    async def create(self, record: EvaluationRecord) -> EvaluationRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            current = self._records.get((record.tenant_id, record.evaluation_id))
            if current is not None:
                if current == record:
                    return current
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[(record.tenant_id, record.evaluation_id)] = record
            self._mark_changed()
            return record

    async def get(self, evaluation_id: str, *, tenant_id: str) -> EvaluationRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, evaluation_id))

    async def compare_and_swap(self, evaluation_id: str, *, tenant_id: str, expected_revision: int, next_record: EvaluationRecord) -> EvaluationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._records.get((tenant_id, evaluation_id))
            if (
                current is None
                or current.revision != expected_revision
                or next_record.evaluation_id != evaluation_id
                or next_record.tenant_id != tenant_id
                or next_record.revision != expected_revision + 1
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[(tenant_id, evaluation_id)] = next_record
            self._mark_changed()
            return next_record

    async def list_by_execution(self, execution_id: str, *, tenant_id: str) -> tuple[EvaluationRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(sorted((item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id), key=lambda item: item.evaluation_id))


class _ToolRepository(_Base, ToolStateRepository):
    def __init__(self, namespace: str, coordinator: RuntimeTransactionCoordinator) -> None:
        super().__init__(namespace, coordinator)
        self._records: dict[tuple[str, str], ToolOperationRecord] = {}

    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        if re.fullmatch(r"[0-9a-f]{64}", record.idempotency_key_digest) is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        async with self._lock:
            key = (record.tenant_id, record.tool_operation_id)
            current = self._records.get(key)
            if current is not None:
                if (
                    current.step_run_id != record.step_run_id
                    or current.tool_call_id != record.tool_call_id
                    or current.idempotency_key_digest != record.idempotency_key_digest
                    or current.tool_name != record.tool_name
                    or current.arguments_digest != record.arguments_digest
                    or current.binding_fingerprint != record.binding_fingerprint
                    or current.replay_safe != record.replay_safe
                ):
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                return current
            self._records[key] = record
            self._mark_changed()
            return record

    async def get_operation(self, tool_operation_id: str, *, tenant_id: str) -> ToolOperationRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, tool_operation_id))

    async def claim(self, tool_operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        validate_lease_owner(owner)
        if not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        result: ToolOperationRecord | None = None
        effect_unknown = False
        async with self._lock:
            current = self._records.get((tenant_id, tool_operation_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
                raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
            now = datetime.now(timezone.utc)
            if current.status is ToolOperationStatus.CLAIMED and current.lease_expires_at is not None and current.lease_expires_at <= now and not current.replay_safe:
                result = replace(current, status=ToolOperationStatus.EFFECT_UNKNOWN, lease_expires_at=None)
                self._records[(tenant_id, tool_operation_id)] = result
                self._mark_changed()
                effect_unknown = True
            else:
                if current.owner is not None and current.lease_expires_at is not None and current.lease_expires_at > now:
                    raise AIError(ErrorCode.TASK_OWNER_CONFLICT)
                result = replace(current, status=ToolOperationStatus.CLAIMED, owner=owner, fence=current.fence + 1, lease_expires_at=now + timedelta(seconds=lease_seconds))
                self._records[(tenant_id, tool_operation_id)] = result
                self._mark_changed()
        if effect_unknown:
            _logger.warning("tool effect became unknown: tenant=%s tool_operation=%s", tenant_id, tool_operation_id)
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        if result is None:
            raise RuntimeError("tool claim did not produce a result")
        return result

    async def renew(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, lease_seconds: int) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = await self._require_claim(tool_operation_id, tenant_id, owner, fence)
            updated = replace(current, lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds))
            self._records[(tenant_id, tool_operation_id)] = updated
            self._mark_changed()
            return updated

    async def complete(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, result_object_ref: "ObjectRef | None") -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = await self._require_claim(tool_operation_id, tenant_id, owner, fence)
            if current.status is ToolOperationStatus.COMPLETED:
                if current.result_object_ref == result_object_ref:
                    return current
                raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
            updated = replace(current, status=ToolOperationStatus.COMPLETED, result_object_ref=result_object_ref, lease_expires_at=None)
            self._records[(tenant_id, tool_operation_id)] = updated
            self._mark_changed()
            return updated

    async def fail(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, error_code: str) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = await self._require_claim(tool_operation_id, tenant_id, owner, fence)
            updated = replace(current, status=ToolOperationStatus.FAILED, error_code=error_code, lease_expires_at=None)
            self._records[(tenant_id, tool_operation_id)] = updated
            self._mark_changed()
            return updated

    async def _require_claim(self, tool_operation_id: str, tenant_id: str, owner: str, fence: int) -> ToolOperationRecord:
        validate_lease_owner(owner)
        current = self._records.get((tenant_id, tool_operation_id))
        if current is None or current.owner != owner or current.fence != fence or current.lease_expires_at is None or current.lease_expires_at <= datetime.now(timezone.utc):
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        return current




_RuntimeStateValue = ConversationState | ExecutionState | MemoryState | ArtifactState | TaskState | EvaluationState | RecoveryState


@dataclass(frozen=True, slots=True)
class _DomainRepositoryParts:
    states: Mapping[RuntimeDomain, _RuntimeStateValue]
    components: tuple[RuntimeRepository, ...]


class _RuntimeOwnerPruner:
    def __init__(
        self,
        *,
        conversation: ConversationState,
        execution: ExecutionState,
        memory: MemoryState,
        artifact: ArtifactState,
        task: TaskState,
        evaluation: EvaluationState,
        recovery: RecoveryState,
        transient_domains: frozenset[RuntimeDomain],
    ) -> None:
        self._states = {
            RuntimeDomain.CONVERSATION: conversation,
            RuntimeDomain.EXECUTION: execution,
            RuntimeDomain.MEMORY: memory,
            RuntimeDomain.ARTIFACT: artifact,
            RuntimeDomain.TASK: task,
            RuntimeDomain.EVALUATION: evaluation,
            RuntimeDomain.RECOVERY: recovery,
        }
        self._transient_domains = transient_domains

    async def prune_execution_working(
        self,
        execution_id: str,
        tenant_id: str,
        memory_scope_key: str | None,
        candidate_step_run_ids: frozenset[str],
        *,
        domains: frozenset[RuntimeDomain] | None = None,
    ) -> None:
        active_domains = self._transient_domains if domains is None else self._transient_domains & domains
        if RuntimeDomain.MEMORY in active_domains:
            state = self._states[RuntimeDomain.MEMORY]
            records = state.records
            if isinstance(records, _MemoryRepository) and memory_scope_key is not None:
                async with records._lock:
                    memory_ids = {
                        record.memory_id
                        for record in records._records.values()
                        if record.tenant_id == tenant_id and record.memory_scope_key == memory_scope_key
                    }
                    changed = any(key[0] == tenant_id and key[1] in memory_ids for key in records._records)
                    for key in tuple(records._records):
                        if key[0] == tenant_id and key[1] in memory_ids:
                            records._records.pop(key, None)
                    changed = self._delete_operations(
                        state.operations,
                        tenant_id,
                        lambda item: item.execution_id == execution_id
                        or item.resource_kind is ResourceKind.MEMORY
                        and item.resource_id in memory_ids,
                    ) or changed
                    if changed:
                        records._mark_changed()

        if RuntimeDomain.ARTIFACT in active_domains:
            state = self._states[RuntimeDomain.ARTIFACT]
            records = state.records
            if isinstance(records, _ArtifactRepository):
                async with records._lock:
                    artifact_ids = {
                        record.artifact_id
                        for record in records._records.values()
                        if record.tenant_id == tenant_id and record.execution_id == execution_id
                    }
                    changed = any(key[0] == tenant_id and key[1] in artifact_ids for key in records._records)
                    for key in tuple(records._records):
                        if key[0] == tenant_id and key[1] in artifact_ids:
                            records._records.pop(key, None)
                    changed = self._delete_operations(
                        state.operations,
                        tenant_id,
                        lambda item: item.execution_id == execution_id
                        or item.resource_kind is ResourceKind.ARTIFACT
                        and item.resource_id in artifact_ids,
                    ) or changed
                    if changed:
                        records._mark_changed()

        if RuntimeDomain.RECOVERY in active_domains:
            state = self._states[RuntimeDomain.RECOVERY]
            if all(
                isinstance(repository, (_ApprovalRepository, _ExternalRepository, _RecoveryCheckpointRepository, _ToolRepository))
                for repository in (state.approvals, state.external_calls, state.checkpoints, state.tools)
            ):
                coordinator = state.approvals._lock
                async with coordinator:
                    approval_ids = {
                        record.approval_id
                        for record in state.approvals._records.values()
                        if record.tenant_id == tenant_id and record.execution_id == execution_id
                    }
                    call_ids = {
                        record.call_id
                        for record in state.external_calls._records.values()
                        if record.tenant_id == tenant_id and record.execution_id == execution_id
                    }
                    checkpoint_ids = {
                        record.execution_id
                        for record in state.checkpoints._records.values()
                        if record.tenant_id == tenant_id and record.execution_id == execution_id
                    }
                    tool_ids = {
                        record.tool_operation_id
                        for key, record in state.tools._records.items()
                        if key[0] == tenant_id and record.step_run_id in candidate_step_run_ids
                    }
                    changed = False
                    for key in tuple(state.approvals._records):
                        if key[0] == tenant_id and key[1] in approval_ids:
                            state.approvals._records.pop(key, None)
                            changed = True
                    for key in tuple(state.external_calls._records):
                        if key[0] == tenant_id and key[1] in call_ids:
                            state.external_calls._records.pop(key, None)
                            changed = True
                    for key in tuple(state.checkpoints._records):
                        if key[0] == tenant_id and key[1] in checkpoint_ids:
                            state.checkpoints._records.pop(key, None)
                            changed = True
                    for key, record in tuple(state.tools._records.items()):
                        if key[0] == tenant_id and record.step_run_id in candidate_step_run_ids:
                            state.tools._records.pop(key, None)
                            changed = True
                    changed = self._delete_operations(
                        state.operations,
                        tenant_id,
                        lambda item: item.execution_id == execution_id
                        or item.resource_kind in {ResourceKind.APPROVAL, ResourceKind.EXTERNAL_CALL, ResourceKind.TOOL_OPERATION}
                        and item.resource_id in approval_ids | call_ids | checkpoint_ids | tool_ids,
                    ) or changed
                    if changed:
                        state.approvals._mark_changed()

    async def prune_execution_terminal(self, execution_id: str, tenant_id: str) -> None:
        state = self._states[RuntimeDomain.EXECUTION]
        executions = state.executions
        if not isinstance(executions, _ExecutionRepository):
            return
        async with executions._lock:
            changed = executions._records.pop((tenant_id, execution_id), None) is not None
            if isinstance(executions._terminal, _TerminalCommitRepository):
                changed = executions._terminal._results.pop((tenant_id, execution_id), None) is not None or changed
            if isinstance(state.events, _EventRepository):
                changed = state.events._items.pop((tenant_id, execution_id), None) is not None or changed
            if isinstance(state.idempotency, _IdempotencyRepository):
                for key in tuple(state.idempotency._records):
                    record = state.idempotency._records[key]
                    if record.tenant_id == tenant_id and record.runtime_domain is RuntimeDomain.EXECUTION and record.resource_kind is ResourceKind.EXECUTION and record.resource_id == execution_id:
                        state.idempotency._records.pop(key, None)
                        changed = True
            changed = self._delete_operations(
                state.operations,
                tenant_id,
                lambda item: item.execution_id == execution_id
                or item.resource_kind is ResourceKind.EXECUTION
                and item.resource_id == execution_id,
            ) or changed
            if changed:
                executions._mark_changed()

    async def prune_session(self, session_id: str, tenant_id: str, continuation_step_run_id: str | None) -> bool:
        state = self._states[RuntimeDomain.CONVERSATION]
        sessions = state.sessions
        if not isinstance(sessions, _SessionRepository):
            return False
        async with sessions._lock:
            sessions._records.pop((tenant_id, session_id), None)
            self._delete_operations(
                state.operations,
                tenant_id,
                lambda item: item.resource_kind is ResourceKind.SESSION and item.resource_id == session_id,
            )
            if continuation_step_run_id is None:
                return False
            return not any(
                record.tenant_id == tenant_id
                and record.continuation is not None
                and record.continuation.step_run_id == continuation_step_run_id
                for record in sessions._records.values()
            )

    async def prune_task_graph(self, graph_id: str, tenant_id: str) -> None:
        state = self._states[RuntimeDomain.TASK]
        tasks = state.tasks
        if not isinstance(tasks, _TaskRepository):
            return
        async with tasks._lock:
            changed = tasks._plans.pop((tenant_id, graph_id), None) is not None
            for key in tuple(tasks._nodes):
                if key[:2] == (tenant_id, graph_id):
                    tasks._nodes.pop(key, None)
                    changed = True
            changed = self._delete_operations(
                state.operations,
                tenant_id,
                lambda item: item.resource_kind is ResourceKind.TASK_GRAPH and item.resource_id == graph_id,
            ) or changed
            if changed:
                tasks._mark_changed()

    async def prune_evaluation(self, evaluation_id: str, tenant_id: str) -> None:
        state = self._states[RuntimeDomain.EVALUATION]
        records = state.records
        if not isinstance(records, _EvaluationRepository):
            return
        async with records._lock:
            changed = records._records.pop((tenant_id, evaluation_id), None) is not None
            if isinstance(state.idempotency, _IdempotencyRepository):
                for key in tuple(state.idempotency._records):
                    item = state.idempotency._records[key]
                    if item.tenant_id == tenant_id and item.runtime_domain is RuntimeDomain.EVALUATION and item.resource_kind is ResourceKind.EVALUATION and item.resource_id == evaluation_id:
                        state.idempotency._records.pop(key, None)
                        changed = True
            changed = self._delete_operations(
                state.operations,
                tenant_id,
                lambda item: item.resource_kind is ResourceKind.EVALUATION and item.resource_id == evaluation_id,
            ) or changed
            if changed:
                records._mark_changed()

    async def clear_transient(self) -> None:
        for domain in RuntimeDomain:
            if domain in self._transient_domains:
                await self._clear_domain(domain)

    async def _clear_domain(self, domain: RuntimeDomain) -> None:
        state = self._states[domain]
        if domain is RuntimeDomain.CONVERSATION:
            components = (state.sessions, state.operations)
        elif domain is RuntimeDomain.EXECUTION:
            components = (state.executions, state.events, state.idempotency, state.operations)
        elif domain is RuntimeDomain.MEMORY:
            components = (state.records, state.operations)
        elif domain is RuntimeDomain.ARTIFACT:
            components = (state.records, state.operations)
        elif domain is RuntimeDomain.TASK:
            components = (state.tasks, state.operations)
        elif domain is RuntimeDomain.EVALUATION:
            components = (state.records, state.idempotency, state.operations)
        else:
            components = (state.approvals, state.external_calls, state.checkpoints, state.operations, state.tools)
        coordinator = next((component._lock for component in components if isinstance(component, _Base)), None)
        if coordinator is None:
            return
        async with coordinator:
            changed = False
            for component in components:
                changed = _clear_in_memory_component(component) or changed
            if changed:
                next(component for component in components if isinstance(component, _Base))._mark_changed()

    @staticmethod
    def _delete_operations(
        repository: object,
        tenant_id: str,
        predicate: Callable[[OperationLedgerRecord], bool],
    ) -> bool:
        if not isinstance(repository, _OperationRepository):
            return False
        changed = False
        for key, item in tuple(repository._records.items()):
            if key[0] == tenant_id and predicate(item):
                repository._records.pop(key, None)
                changed = True
        if changed:
            repository._mark_changed()
        return changed


def _clear_in_memory_component(component: object) -> bool:
    if isinstance(component, _SessionRepository):
        changed = bool(component._records)
        component._records.clear()
        return changed
    if isinstance(component, _ExecutionRepository):
        changed = bool(component._records)
        component._records.clear()
        if isinstance(component._terminal, _TerminalCommitRepository):
            changed = bool(component._terminal._results) or changed
            component._terminal._results.clear()
        return changed
    if isinstance(component, _EventRepository):
        changed = bool(component._items)
        component._items.clear()
        return changed
    if isinstance(component, _IdempotencyRepository):
        changed = bool(component._records)
        component._records.clear()
        return changed
    if isinstance(component, (_MemoryRepository, _ArtifactRepository, _ApprovalRepository, _ExternalRepository, _RecoveryCheckpointRepository, _EvaluationRepository, _ToolRepository)):
        changed = bool(component._records)
        component._records.clear()
        return changed
    if isinstance(component, _TaskRepository):
        changed = bool(component._plans or component._nodes)
        component._plans.clear()
        component._nodes.clear()
        return changed
    if isinstance(component, _OperationRepository):
        changed = bool(component._records or component._counters)
        component._records.clear()
        component._counters.clear()
        return changed
    return False


def _build_in_memory_domains(
    *,
    namespace: str,
    domains: frozenset[RuntimeDomain],
    transaction_hub: TransactionHub | None = None,
    transaction_binding: RuntimeTransactionBinding | None = None,
) -> _DomainRepositoryParts:
    validate_persistence_namespace(namespace)
    hub = transaction_hub or TransactionHub()
    binding = transaction_binding or RuntimeTransactionBinding()
    coordinators = {
        domain: RuntimeTransactionCoordinator(domain, hub=hub)
        for domain in domains
    }

    def coordinator(domain: RuntimeDomain) -> RuntimeTransactionCoordinator:
        try:
            return coordinators[domain]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    operations = {domain: _OperationRepository(namespace, coordinator(domain)) for domain in domains}
    sessions = _SessionRepository(namespace, coordinator(RuntimeDomain.CONVERSATION)) if RuntimeDomain.CONVERSATION in domains else None
    executions = _ExecutionRepository(namespace, coordinator(RuntimeDomain.EXECUTION)) if RuntimeDomain.EXECUTION in domains else None
    terminal = _TerminalCommitRepository(executions, namespace, coordinator(RuntimeDomain.EXECUTION)) if executions is not None else None
    execution_idempotency = _IdempotencyRepository(namespace, coordinator(RuntimeDomain.EXECUTION), RuntimeDomain.EXECUTION) if executions is not None else None
    events = _EventRepository(executions, namespace, coordinator(RuntimeDomain.EXECUTION)) if executions is not None else None
    tasks = _TaskRepository(namespace, coordinator(RuntimeDomain.TASK)) if RuntimeDomain.TASK in domains else None
    evaluations = _EvaluationRepository(namespace, coordinator(RuntimeDomain.EVALUATION)) if RuntimeDomain.EVALUATION in domains else None
    evaluation_idempotency = _IdempotencyRepository(namespace, coordinator(RuntimeDomain.EVALUATION), RuntimeDomain.EVALUATION) if evaluations is not None else None
    memories = _MemoryRepository(namespace, coordinator(RuntimeDomain.MEMORY)) if RuntimeDomain.MEMORY in domains else None
    artifacts = _ArtifactRepository(namespace, coordinator(RuntimeDomain.ARTIFACT)) if RuntimeDomain.ARTIFACT in domains else None
    approvals = _ApprovalRepository(namespace, coordinator(RuntimeDomain.RECOVERY)) if RuntimeDomain.RECOVERY in domains else None
    external_calls = _ExternalRepository(namespace, coordinator(RuntimeDomain.RECOVERY)) if RuntimeDomain.RECOVERY in domains else None
    checkpoints = _RecoveryCheckpointRepository(namespace, coordinator(RuntimeDomain.RECOVERY)) if RuntimeDomain.RECOVERY in domains else None
    tools = _ToolRepository(namespace, coordinator(RuntimeDomain.RECOVERY)) if RuntimeDomain.RECOVERY in domains else None

    if executions is not None and terminal is not None and execution_idempotency is not None and events is not None:
        executions.bind_start_repositories(execution_idempotency, events, operations[RuntimeDomain.EXECUTION])
        executions.bind_terminal_repository(terminal)
        terminal.bind_terminal_repositories(execution_idempotency, events, operations[RuntimeDomain.EXECUTION])
    if memories is not None:
        memories.bind_operation_repository(operations[RuntimeDomain.MEMORY])

    components: list[RuntimeRepository] = []
    for domain in RuntimeDomain:
        if domain in operations:
            components.append(operations[domain])
        domain_components: tuple[RuntimeRepository | None, ...] = {
            RuntimeDomain.CONVERSATION: (sessions,),
            RuntimeDomain.EXECUTION: (executions, terminal, execution_idempotency, events),
            RuntimeDomain.MEMORY: (memories,),
            RuntimeDomain.ARTIFACT: (artifacts,),
            RuntimeDomain.TASK: (tasks,),
            RuntimeDomain.EVALUATION: (evaluations, evaluation_idempotency),
            RuntimeDomain.RECOVERY: (approvals, external_calls, checkpoints, tools),
        }[domain]
        components.extend(item for item in domain_components if item is not None)

    states: dict[RuntimeDomain, _RuntimeStateValue] = {}
    if sessions is not None:
        states[RuntimeDomain.CONVERSATION] = ConversationState(sessions, operations[RuntimeDomain.CONVERSATION])
    if executions is not None and execution_idempotency is not None and events is not None:
        states[RuntimeDomain.EXECUTION] = ExecutionState(executions, events, execution_idempotency, operations[RuntimeDomain.EXECUTION])
    if memories is not None:
        states[RuntimeDomain.MEMORY] = MemoryState(memories, operations[RuntimeDomain.MEMORY])
    if artifacts is not None:
        states[RuntimeDomain.ARTIFACT] = ArtifactState(artifacts, operations[RuntimeDomain.ARTIFACT])
    if tasks is not None:
        states[RuntimeDomain.TASK] = TaskState(tasks, operations[RuntimeDomain.TASK])
    if evaluations is not None and evaluation_idempotency is not None:
        states[RuntimeDomain.EVALUATION] = EvaluationState(evaluations, evaluation_idempotency, operations[RuntimeDomain.EVALUATION])
    if approvals is not None and external_calls is not None and checkpoints is not None and tools is not None:
        states[RuntimeDomain.RECOVERY] = RecoveryState(approvals, external_calls, checkpoints, operations[RuntimeDomain.RECOVERY], tools)
    if frozenset(states) != domains:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    binding.bind_components(tuple(components))
    if not hub.configured:
        hub.configure(snapshot=binding.snapshot, restore=binding.restore, commit=binding.commit, rollback=binding.rollback)
    _logger.debug('in-memory runtime domains composed: namespace=%s domains=%s', namespace, sorted(domain.value for domain in domains))
    return _DomainRepositoryParts(states, tuple(components))


def _capture_runtime_snapshot(components: tuple[RuntimeRepository, ...]) -> dict[int, tuple[object, dict[str, object]]]:
    snapshot: dict[int, tuple[object, dict[str, object]]] = {}
    for component in components:
        if id(component) in snapshot:
            continue
        values: dict[str, object] = {}
        if isinstance(component, (_SessionRepository, _ExecutionRepository, _IdempotencyRepository, _MemoryRepository, _ArtifactRepository, _ApprovalRepository, _ExternalRepository, _RecoveryCheckpointRepository, _OperationRepository, _EvaluationRepository, _ToolRepository)):
            values["_records"] = copy.deepcopy(component._records)
        if isinstance(component, _OperationRepository):
            values["_counters"] = copy.deepcopy(component._counters)
        if isinstance(component, _EventRepository):
            values["_items"] = copy.deepcopy(component._items)
        if isinstance(component, _TerminalCommitRepository):
            values["_results"] = copy.deepcopy(component._results)
        if isinstance(component, _TaskRepository):
            values["_plans"] = copy.deepcopy(component._plans)
            values["_nodes"] = copy.deepcopy(component._nodes)
        if values:
            snapshot[id(component)] = (component, values)
    return snapshot


def _restore_runtime_snapshot(snapshot: object) -> None:
    if not isinstance(snapshot, dict):
        return
    for component, values in snapshot.values():
        for name, value in values.items():
            if name == "_records":
                component._records = value
            elif name == "_items":
                component._items = value
            elif name == "_results":
                component._results = value
            elif name == "_plans":
                component._plans = value
            elif name == "_nodes":
                component._nodes = value
            elif name == "_counters":
                component._counters = value


def _validate_terminal_result(execution: ExecutionRecord, result: ResultRecord) -> None:
    schema = (
        result.output_schema_id,
        result.output_schema_revision,
        result.output_schema_fingerprint,
    )
    has_schema = all(value is not None for value in schema)
    if execution.status is ExecutionStatus.SUCCEEDED and (
        not has_schema or result.object_ref is None
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if execution.status is not ExecutionStatus.SUCCEEDED and (
        has_schema or result.object_ref is not None
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _operation_immutable(record: OperationLedgerRecord) -> tuple[object, ...]:
    return (
        record.operation_id,
        record.tenant_id,
        record.resource_kind,
        record.resource_id,
        record.execution_id,
        record.operation_kind,
        record.status,
        record.request_digest,
        record.result_ref,
        record.result_digest,
        record.error_code,
        record.compactable,
    )


def _operation_input_immutable(record: OperationLedgerInput) -> tuple[object, ...]:
    return (
        record.operation_id,
        record.tenant_id,
        record.resource_kind,
        record.resource_id,
        record.execution_id,
        record.operation_kind,
        record.status,
        record.request_digest,
        record.result_ref,
        record.result_digest,
        record.error_code,
        record.compactable,
    )


__all__ = []
