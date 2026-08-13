#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference runtime persistence for in-memory and filesystem backends."""

import asyncio
import base64
import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from linktools.core import environ

from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    EvaluationStatus,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationStatus,
    Page,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    StopReason,
    TaskStatus,
    ToolOperationStatus,
    canonical_sha256,
    validate_lease_owner,
    validate_persistence_namespace,
    validate_observation_payload,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..runtime import (
    ApprovalRecord,
    ArtifactRecord,
    ArtifactStore,
    BlobRef,
    BlobStore,
    EvaluationRecord,
    EvaluationStore,
    ExecutionCancelRequestCommit,
    ExecutionEventRecord,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionStartUnknownCommit,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExecutionStore,
    ExternalResultRecord,
    RecoveryCheckpoint,
    IdempotencyRecord,
    IdempotencyRepository,
    IdempotencyTerminalUpdate,
    MemoryRecord,
    MemoryStore,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationLedgerRepository,
    OperationTerminalUpdate,
    ResultRecord,
    RuntimeStores,
    RuntimeRepository,
    RecoveryStore,
    SessionRecord,
    ConversationCursor,
    ConversationStore,
    DomainBlobStore,
    TaskLease,
    TaskNodeView,
    TaskStore,
    ToolOperationRecord,
    ToolStateStore,
)
from ..storage import (
    FilesystemLeaseCoordinator,
    FilesystemWriterLock,
    read_json,
    sync_directory,
    write_json_atomic,
    StorageDomain,
)
from ..task import TaskGraph, TaskGraphView, TaskNode, TaskTerminalRecord

_MAX_BLOB_BYTES = 2 * 1024 * 1024 * 1024
_MAX_INLINE_BLOB_BYTES = 4 * 1024 * 1024
_logger = environ.get_logger("ai.adapter.persistence")


class _SharedTransactionLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self._depth = 0
        self._dirty: set[StorageDomain] = set()
        self._commit: Callable[[StorageDomain], None] | None = None
        self._rollback: Callable[[frozenset[StorageDomain]], None] | None = None
        self._reject_cross_domain = False

    def configure(
        self,
        *,
        commit: "Callable[[StorageDomain], None] | None" = None,
        rollback: "Callable[[frozenset[StorageDomain]], None] | None" = None,
        reject_cross_domain: bool = False,
    ) -> None:
        self._commit = commit
        self._rollback = rollback
        self._reject_cross_domain = reject_cross_domain

    def mark_changed(self, domain: StorageDomain) -> None:
        self._dirty.add(domain)

    async def __aenter__(self) -> None:
        owner = asyncio.current_task()
        if owner is None:
            raise RuntimeError("runtime transaction requires an asyncio task")
        if self._owner is owner:
            self._depth += 1
            return
        await self._lock.acquire()
        self._owner = owner
        self._depth = 1
        self._dirty.clear()

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if asyncio.current_task() is not self._owner:
            raise RuntimeError("runtime transaction lock owner mismatch")
        self._depth -= 1
        if self._depth != 0:
            return
        dirty = frozenset(self._dirty)
        self._dirty.clear()
        try:
            if exc_type is not None:
                if dirty and self._rollback is not None:
                    self._rollback(dirty)
            elif len(dirty) > 1 and self._reject_cross_domain:
                if self._rollback is not None:
                    self._rollback(dirty)
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif len(dirty) == 1 and self._commit is not None:
                self._commit(next(iter(dirty)))
        finally:
            self._owner = None
            self._lock.release()


class _Base:
    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._closed = False
        self._lock = asyncio.Lock()
        self._on_change: Callable[[], None] | None = None
        self._transaction: _SharedTransactionLock | None = None
        self._owner_domain: StorageDomain | None = None
        self._refresh_source: Callable[[], None] | None = None

    @property
    def namespace(self) -> str:
        return self._namespace

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    def _mark_changed(self) -> None:
        if self._transaction is not None and self._owner_domain is not None:
            self._transaction.mark_changed(self._owner_domain)
            return
        if self._on_change is not None:
            self._on_change()

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._refresh_source is not None:
            self._refresh_source()

    def _check_tenant(self, tenant_id: str) -> None:
        validate_tenant_id(tenant_id)


_COMPONENT_DOMAINS = (
    StorageDomain.CONVERSATION,
    StorageDomain.EXECUTION,
    StorageDomain.EXECUTION,
    StorageDomain.EXECUTION,
    StorageDomain.EXECUTION,
    StorageDomain.TASK,
    StorageDomain.EVALUATION,
    StorageDomain.MEMORY,
    StorageDomain.ARTIFACT,
    StorageDomain.RECOVERY,
    StorageDomain.RECOVERY,
    StorageDomain.EXECUTION,
    StorageDomain.RECOVERY,
    None,
    StorageDomain.EVALUATION,
    StorageDomain.RECOVERY,
    StorageDomain.CONVERSATION,
    StorageDomain.MEMORY,
    StorageDomain.ARTIFACT,
    StorageDomain.TASK,
    StorageDomain.EVALUATION,
    StorageDomain.RECOVERY,
)


def _configure_transaction_component(
    component: RuntimeRepository,
    transaction: _SharedTransactionLock,
    domain: "StorageDomain | None",
) -> None:
    if isinstance(component, _Base):
        component._lock = transaction
        component._transaction = transaction
        component._owner_domain = domain
    if isinstance(component, DomainBlobRouter):
        for store_domain, store in component._stores.items():
            if isinstance(store, _Base):
                store._lock = transaction
                store._transaction = transaction
                store._owner_domain = store_domain


class _SessionRepository(_Base):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
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
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._records: dict[tuple[str, str], ExecutionRecord] = {}
        self._idempotency: _IdempotencyRepository | None = None
        self._events: _EventRepository | None = None
        self._operations: _OperationRepository | None = None

    def bind_start_repositories(self, idempotency: "_IdempotencyRepository", events: "_EventRepository", operations: "_OperationRepository") -> None:
        self._idempotency = idempotency
        self._events = events
        self._operations = operations

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
            identity = await self._idempotency.get(claim.scope, claim.key_hash, tenant_id=claim.tenant_id)
            if current is None or identity is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if current.status is not ExecutionStatus.PENDING_START or current.revision != claim.expected_revision or current.event_sequence != claim.expected_event_sequence or current.agent_run_sequence != 0:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if identity.status is not IdempotencyStatus.RESERVED or identity.execution_id != claim.execution_id or identity.request_digest != claim.request_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            now = claim.started_at
            started = replace(current, status=ExecutionStatus.STARTED, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=now, agent_run_sequence=1)
            started_identity = replace(identity, status=IdempotencyStatus.STARTED, updated_at=now)
            event = ExecutionEventRecord(claim.execution_id, claim.tenant_id, claim.expected_event_sequence + 1, ExecutionEventType.EXECUTION_STARTED, {})
            self._records[key] = started
            self._idempotency._records[(claim.tenant_id, claim.scope, claim.key_hash)] = started_identity
            self._events._items.setdefault((claim.tenant_id, claim.execution_id), []).append(event)
            self._mark_changed()
            return started

    async def reserve_start(self, reservation: ExecutionStartReservation) -> ExecutionStartReservationResult:
        self._ensure_open()
        self._check_tenant(reservation.execution.tenant_id)
        if self._idempotency is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if reservation.execution.tenant_id != reservation.idempotency.tenant_id or reservation.execution.execution_id != reservation.idempotency.execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with self._lock:
            idempotency_key = (reservation.idempotency.tenant_id, reservation.idempotency.scope, reservation.idempotency.key_hash)
            existing_idempotency = self._idempotency._records.get(idempotency_key)
            if existing_idempotency is not None:
                if existing_idempotency.request_digest != reservation.idempotency.request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                existing_execution = self._records.get((existing_idempotency.tenant_id, existing_idempotency.execution_id))
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
            identity = await self._idempotency.get(commit.scope, commit.key_hash, tenant_id=commit.tenant_id)
            if current is None or identity is None or current.status is not ExecutionStatus.STARTED or current.revision != commit.expected_revision or identity.status is not IdempotencyStatus.STARTED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            unknown = replace(current, status=ExecutionStatus.START_UNKNOWN, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.occurred_at)
            self._records[key] = unknown
            self._idempotency._records[(commit.tenant_id, commit.scope, commit.key_hash)] = replace(identity, status=IdempotencyStatus.START_UNKNOWN, updated_at=commit.occurred_at)
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

    async def advance_sequence(self, execution_id: str, *, tenant_id: str, kind: str, expected_sequence: int) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if kind != "event":
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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

    async def commit_terminal(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: ExecutionRecord) -> ExecutionRecord:
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
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._records: dict[tuple[str, str, str], IdempotencyRecord] = {}

    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        if re.fullmatch(r"[0-9a-f]{64}", record.key_hash) is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        async with self._lock:
            key = (record.tenant_id, record.scope, record.key_hash)
            current = self._records.get(key)
            if current is None:
                self._records[key] = record
                self._mark_changed()
                return record
            if current.request_digest != record.request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return current

    async def get(self, scope: str, key_hash: str, *, tenant_id: str) -> IdempotencyRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, scope, key_hash))

    async def list_by_execution(self, execution_id: str, *, tenant_id: str) -> tuple[IdempotencyRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(sorted((record for record in self._records.values() if record.tenant_id == tenant_id and record.execution_id == execution_id), key=lambda record: (record.scope, record.key_hash)))

    async def compare_and_swap(self, scope: str, key_hash: str, *, tenant_id: str, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            store_key = (tenant_id, scope, key_hash)
            current = self._records.get(store_key)
            if current is None or current.status is not expected_status:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.tenant_id != tenant_id or next_record.key_hash != key_hash:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._records[store_key] = next_record
            self._mark_changed()
            return next_record


class _EventRepository(_Base):
    def __init__(self, executions: _ExecutionRepository, namespace: str) -> None:
        super().__init__(namespace)
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
            await self._executions.advance_sequence(execution_id, tenant_id=tenant_id, kind="event", expected_sequence=expected_sequence)
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


class _ResultRepository(_Base):
    def __init__(self, executions: _ExecutionRepository, namespace: str) -> None:
        super().__init__(namespace)
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
            or result.status is not execution.status
        ):
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
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
                if (current is not None and current.status is execution.status and current.result_digest == execution.result_digest and current_result == result and len(event) > commit.expected_event_sequence and event[commit.expected_event_sequence].event_type is commit.terminal_event_type and event[commit.expected_event_sequence].payload == commit.terminal_event_payload and self._idempotency_matches(identity, commit.idempotency)):
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
                identity_key = (execution.tenant_id, commit.idempotency.scope, commit.idempotency.key_hash)
                self._idempotency._records[identity_key] = replace(identity, status=commit.idempotency.next_status, result_digest=commit.idempotency.result_digest, error_code=commit.idempotency.error_code, updated_at=execution.updated_at)
            if commit.operation is not None:
                current_operation = self._operations._records[(execution.tenant_id, commit.operation.operation_id)]
                self._operations._records[(execution.tenant_id, commit.operation.operation_id)] = replace(current_operation, status=commit.operation.next_status, result_ref=commit.operation.result_ref, result_digest=commit.operation.result_digest, error_code=commit.operation.error_code, updated_at=execution.updated_at)
            self._mark_changed()
            return ExecutionTerminalCommitResult(execution, result)

    def _find_idempotency(self, commit: ExecutionTerminalCommit, execution: ExecutionRecord) -> IdempotencyRecord | None:
        records = tuple(record for record in self._idempotency._records.values() if record.tenant_id == execution.tenant_id and record.execution_id == execution.execution_id)
        if len(records) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return records[0] if records else None

    def _validate_idempotency(self, current: IdempotencyRecord | None, update: IdempotencyTerminalUpdate | None, execution: ExecutionRecord) -> None:
        if update is None:
            if current is not None and current.status in {IdempotencyStatus.RESERVED, IdempotencyStatus.STARTED}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if current is None or current.tenant_id != execution.tenant_id or current.execution_id != execution.execution_id or current.scope != update.scope or current.key_hash != update.key_hash or current.request_digest != update.request_digest or current.status is not update.expected_status:
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
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
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

    async def list(self, *, tenant_id: str, memory_namespace_key: str, cursor: "str | None", limit: int) -> "Page[MemoryRecord]":
        self._ensure_open()
        self._check_tenant(tenant_id)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(sorted((item for item in self._records.values() if item.tenant_id == tenant_id and item.memory_namespace_key == memory_namespace_key), key=lambda item: item.memory_id))
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
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._records: dict[tuple[str, str], ArtifactRecord] = {}

    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
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
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._records: dict[tuple[str, str], ApprovalRecord] = {}

    async def get_header(self, approval_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, approval_id))
        return None if record is None else ResourceRef(ResourceKind.APPROVAL, approval_id, tenant_id)

    async def create(self, record: ApprovalRecord) -> ApprovalRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
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

    async def decide(self, approval_id: str, *, tenant_id: str, expected_status: ApprovalStatus, decision_id: str, decision: ApprovalDecision, principal_id: str, decision_digest: str, decided_at: datetime) -> ApprovalRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._records.get((tenant_id, approval_id))
            if current is None or current.status is not expected_status:
                if current is not None and current.decision_id == decision_id and current.decision is decision and current.decision_digest == decision_digest:
                    return current
                raise AIError(ErrorCode.APPROVAL_CONFLICT)
            updated = replace(current, status=ApprovalStatus.APPROVED if decision is ApprovalDecision.APPROVE else ApprovalStatus.DENIED, decision_id=decision_id, decision=decision, decided_by=principal_id, decision_digest=decision_digest, decided_at=decided_at)
            self._records[(tenant_id, approval_id)] = updated
            self._mark_changed()
            return updated

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id and item.status is ApprovalStatus.PENDING)


class _ExternalRepository(_Base):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._records: dict[tuple[str, str], ExternalResultRecord] = {}

    async def get_header(self, call_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, call_id))
        return None if record is None else ResourceRef(ResourceKind.EXTERNAL_CALL, call_id, tenant_id)

    async def create_call(self, record: ExternalResultRecord) -> ExternalResultRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        current = self._records.get((record.tenant_id, record.call_id))
        if current is not None:
            if current == record:
                return current
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        self._records[(record.tenant_id, record.call_id)] = record
        self._mark_changed()
        return record

    async def get(self, call_id: str, *, tenant_id: str) -> ExternalResultRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, call_id))

    async def supply(self, call_id: str, *, tenant_id: str, expected_status: ExternalCallStatus, result_id: str, payload_ref: str, payload_digest: str, supplied_at: datetime) -> ExternalResultRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = self._records.get((tenant_id, call_id))
        if current is None or current.status is not expected_status:
            if current is not None and current.result_id == result_id and current.payload_ref == payload_ref and current.payload_digest == payload_digest:
                return current
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        updated = replace(current, status=ExternalCallStatus.SUPPLIED, result_id=result_id, payload_ref=payload_ref, payload_digest=payload_digest, supplied_at=supplied_at)
        self._records[(tenant_id, call_id)] = updated
        self._mark_changed()
        return updated

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ExternalResultRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id and item.status is ExternalCallStatus.PENDING)


class _RecoveryCheckpointRepository(_Base):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._records: dict[tuple[str, str], RecoveryCheckpoint] = {}

    async def create(self, record: RecoveryCheckpoint) -> RecoveryCheckpoint:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        key = record.tenant_id, record.checkpoint_id
        current = self._records.get(key)
        if current is not None:
            if current == record:
                return current
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._records[key] = record
        self._mark_changed()
        return record

    async def get(self, checkpoint_id: str, *, tenant_id: str) -> RecoveryCheckpoint | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, checkpoint_id))

    async def list(self, *, tenant_id: str) -> tuple[RecoveryCheckpoint, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(item for item in self._records.values() if item.tenant_id == tenant_id)

    async def compare_and_swap(self, checkpoint_id: str, *, tenant_id: str, expected_revision: int, next_record: RecoveryCheckpoint) -> RecoveryCheckpoint:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = self._records.get((tenant_id, checkpoint_id))
        if current is None or current.revision != expected_revision or next_record.revision != expected_revision + 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._records[(tenant_id, checkpoint_id)] = next_record
        self._mark_changed()
        return next_record


class _OperationRepository(_Base):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._records: dict[tuple[str, str], OperationLedgerRecord] = {}

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
        sequence = max(
            (item.sequence for item in self._records.values() if item.tenant_id == record.tenant_id and item.resource_kind is record.resource_kind and item.resource_id == record.resource_id),
            default=0,
        ) + 1
        created = OperationLedgerRecord(
            record.operation_id, record.tenant_id, record.resource_kind, record.resource_id,
            record.execution_id, record.kind, record.status, record.request_digest,
            record.result_ref, record.result_digest, record.error_code, record.compactable,
            sequence, record.created_at, record.updated_at,
        )
        self._records[key] = created
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
        values = tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.resource_kind is resource_kind and item.resource_id == resource_id and item.sequence <= through_sequence and item.compactable and item.status not in {OperationStatus.PENDING, OperationStatus.RUNNING})
        digest = canonical_sha256([asdict(item) for item in values])
        for item in values:
            self._records.pop((tenant_id, item.operation_id), None)
        self._mark_changed()
        return digest


class _TaskRepository(_Base):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._plans: dict[tuple[str, str], TaskGraphView] = {}
        self._nodes: dict[tuple[str, str, str], TaskNodeView] = {}

    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if (tenant_id, graph_id) not in self._plans:
            return None
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def create_plan(self, graph: TaskGraph, *, tenant_id: str) -> TaskGraphView:
        self._ensure_open()
        self._check_tenant(tenant_id)
        key = (tenant_id, graph.graph_id)
        if key in self._plans:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        view = TaskGraphView(graph.graph_id, TaskStatus.PENDING, graph.nodes)
        self._plans[key] = view
        for node in graph.nodes:
            self._nodes[(tenant_id, graph.graph_id, node.task_id)] = TaskNodeView(graph.graph_id, node.task_id, node.dependencies, TaskStatus.PENDING, None, 0, None, None, None, None)
        self._mark_changed()
        return view

    async def get_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._plans.get((tenant_id, graph_id))

    async def reconcile_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            current = self._plans.get((tenant_id, graph_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            nodes = {key[2]: value for key, value in self._nodes.items() if key[:2] == (tenant_id, graph_id)}
            if current.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                for task_id, node in tuple(nodes.items()):
                    if node.status in {TaskStatus.PENDING, TaskStatus.READY}:
                        dependencies = tuple(nodes[dependency] for dependency in node.dependencies)
                        if any(dependency.status in {TaskStatus.FAILED, TaskStatus.BLOCKED} for dependency in dependencies):
                            nodes[task_id] = replace(node, status=TaskStatus.BLOCKED, error_code=ErrorCode.TASK_DEPENDENCY_FAILED.value, error_digest=canonical_sha256({"graph_id": graph_id, "task_id": task_id, "reason": "dependency_failed"}))
                        elif node.status is TaskStatus.PENDING and all(dependency.status is TaskStatus.SUCCEEDED for dependency in dependencies):
                            nodes[task_id] = replace(node, status=TaskStatus.READY)
                for task_id, node in nodes.items():
                    self._nodes[(tenant_id, graph_id, task_id)] = node
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
                current = replace(current, status=status)
                self._plans[(tenant_id, graph_id)] = current
                self._mark_changed()
            else:
                for key, node in tuple(self._nodes.items()):
                    if key[:2] == (tenant_id, graph_id) and node.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                        self._nodes[key] = replace(node, status=TaskStatus.CANCELLED, owner=None, lease_expires_at=None)
                        self._mark_changed()
            return current

    async def cancel_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
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

    async def claim(self, graph_id: str, task_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease:
        self._ensure_open()
        self._check_tenant(tenant_id)
        validate_lease_owner(owner)
        if not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        key = (tenant_id, graph_id, task_id)
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
            return TaskLease(graph_id, task_id, tenant_id, owner, fence, expiry)

    async def renew(self, lease: TaskLease, *, tenant_id: str, lease_seconds: int) -> TaskLease:
        self._ensure_open()
        self._check_tenant(tenant_id)
        validate_lease_owner(lease.owner)
        if lease.tenant_id != tenant_id or not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        now = datetime.now(timezone.utc)
        current = self._nodes.get((tenant_id, lease.graph_id, lease.task_id))
        if current is None or current.status is not TaskStatus.RUNNING or current.owner != lease.owner or current.fence != lease.fence or current.lease_expires_at is None or current.lease_expires_at <= now:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        expiry = now + timedelta(seconds=lease_seconds)
        self._nodes[(tenant_id, lease.graph_id, lease.task_id)] = replace(current, lease_expires_at=expiry)
        self._mark_changed()
        return replace(lease, lease_expires_at=expiry)

    async def complete(self, lease: TaskLease, *, tenant_id: str, execution_id: "str | None", result_digest: str) -> TaskTerminalRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if lease.tenant_id != tenant_id:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        current = self._nodes.get((tenant_id, lease.graph_id, lease.task_id))
        now = datetime.now(timezone.utc)
        if current is None or current.status is not TaskStatus.RUNNING or current.owner != lease.owner or current.fence != lease.fence or current.lease_expires_at is None or current.lease_expires_at <= now:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        self._nodes[(tenant_id, lease.graph_id, lease.task_id)] = replace(current, status=TaskStatus.SUCCEEDED, owner=None, result_digest=result_digest, execution_id=execution_id, lease_expires_at=None)
        self._mark_changed()
        return TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, TaskStatus.SUCCEEDED, result_digest, None, None, execution_id=execution_id)

    async def fail(self, lease: TaskLease, *, tenant_id: str, error_code: str, error_digest: str) -> TaskTerminalRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if lease.tenant_id != tenant_id:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        current = self._nodes.get((tenant_id, lease.graph_id, lease.task_id))
        now = datetime.now(timezone.utc)
        if current is None or current.status is not TaskStatus.RUNNING or current.owner != lease.owner or current.fence != lease.fence or current.lease_expires_at is None or current.lease_expires_at <= now:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        self._nodes[(tenant_id, lease.graph_id, lease.task_id)] = replace(current, status=TaskStatus.FAILED, owner=None, error_code=error_code, error_digest=error_digest, lease_expires_at=None)
        self._mark_changed()
        return TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, TaskStatus.FAILED, None, error_code, error_digest)

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[TaskNodeView, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(sorted((item for key, item in self._nodes.items() if key[:2] == (tenant_id, graph_id)), key=lambda item: item.task_id))


class _EvaluationRepository(_Base):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
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


class _ToolRepository(_Base, ToolStateStore):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._records: dict[tuple[str, str], ToolOperationRecord] = {}

    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        if re.fullmatch(r"[0-9a-f]{64}", record.idempotency_key_hash) is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        key = (record.tenant_id, record.operation_id)
        current = self._records.get(key)
        if current is not None:
            if (
                current.run_id != record.run_id
                or current.tool_call_id != record.tool_call_id
                or current.idempotency_key_hash != record.idempotency_key_hash
                or current.tool_name != record.tool_name
                or current.arguments_hash != record.arguments_hash
                or current.binding_fingerprint != record.binding_fingerprint
                or current.replay_safe != record.replay_safe
            ):
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            return current
        self._records[key] = record
        self._mark_changed()
        return record

    async def get_operation(self, operation_id: str, *, tenant_id: str) -> ToolOperationRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, operation_id))

    async def claim(self, operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        validate_lease_owner(owner)
        if not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = self._records.get((tenant_id, operation_id))
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
            raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
        if current.status is ToolOperationStatus.CLAIMED and current.lease_expires_at is not None and current.lease_expires_at <= datetime.now(timezone.utc) and not current.replay_safe:
            unknown = replace(current, status=ToolOperationStatus.EFFECT_UNKNOWN, lease_expires_at=None)
            self._records[(tenant_id, operation_id)] = unknown
            self._mark_changed()
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        if current.owner is not None and current.lease_expires_at is not None and current.lease_expires_at > datetime.now(timezone.utc):
            raise AIError(ErrorCode.TASK_OWNER_CONFLICT)
        updated = replace(current, status=ToolOperationStatus.CLAIMED, owner=owner, fence=current.fence + 1, lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds))
        self._records[(tenant_id, operation_id)] = updated
        self._mark_changed()
        return updated

    async def renew(self, operation_id: str, *, tenant_id: str, owner: str, fence: int, lease_seconds: int) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = await self._require_claim(operation_id, tenant_id, owner, fence)
        updated = replace(current, lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds))
        self._records[(tenant_id, operation_id)] = updated
        self._mark_changed()
        return updated

    async def complete(self, operation_id: str, *, tenant_id: str, owner: str, fence: int, result_ref: "str | None", result_digest: str) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = await self._require_claim(operation_id, tenant_id, owner, fence)
        if current.status is ToolOperationStatus.COMPLETED:
            if current.result_digest == result_digest:
                return current
            raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
        updated = replace(current, status=ToolOperationStatus.COMPLETED, result_ref=result_ref, result_digest=result_digest, lease_expires_at=None)
        self._records[(tenant_id, operation_id)] = updated
        self._mark_changed()
        return updated

    async def fail(self, operation_id: str, *, tenant_id: str, owner: str, fence: int, error_code: str) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = await self._require_claim(operation_id, tenant_id, owner, fence)
        updated = replace(current, status=ToolOperationStatus.FAILED, error_code=error_code, lease_expires_at=None)
        self._records[(tenant_id, operation_id)] = updated
        self._mark_changed()
        return updated

    async def _require_claim(self, operation_id: str, tenant_id: str, owner: str, fence: int) -> ToolOperationRecord:
        validate_lease_owner(owner)
        current = self._records.get((tenant_id, operation_id))
        if current is None or current.owner != owner or current.fence != fence or current.lease_expires_at is None or current.lease_expires_at <= datetime.now(timezone.utc):
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        return current


class InMemoryBlobStore(_Base, BlobStore):
    def __init__(self, namespace: str) -> None:
        super().__init__(namespace)
        self._blobs: dict[tuple[str, str], bytes] = {}

    async def put_bytes(self, *, tenant_id: str, data: bytes, expected_digest: str | None = None) -> BlobRef:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if len(data) > _MAX_INLINE_BLOB_BYTES:
            raise ValueError("put_stream is required for blobs larger than 4 MiB")
        digest = hashlib.sha256(data).hexdigest()
        if expected_digest is not None and expected_digest != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._blobs[(tenant_id, digest)] = data
        self._mark_changed()
        return BlobRef(tenant_id, digest, len(data), f"memory:{self.namespace}:{tenant_id}:{digest}")

    async def put_stream(self, *, tenant_id: str, chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> BlobRef:
        self._ensure_open()
        if expected_size < 0 or expected_size > _MAX_BLOB_BYTES:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        self._check_tenant(tenant_id)
        data = bytearray()
        async for chunk in chunks:
            if not 1 <= len(chunk) <= 1024 * 1024 or len(data) + len(chunk) > _MAX_BLOB_BYTES:
                raise ValueError("invalid blob chunk")
            data.extend(chunk)
        if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._blobs[(tenant_id, expected_digest)] = bytes(data)
        self._mark_changed()
        return BlobRef(tenant_id, expected_digest, len(data), f"memory:{self.namespace}:{tenant_id}:{expected_digest}")

    async def stat(self, ref: BlobRef, *, tenant_id: str) -> BlobRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        data = self._blobs.get((tenant_id, ref.digest))
        return None if data is None or ref.tenant_id != tenant_id else BlobRef(tenant_id, ref.digest, len(data), ref.locator)

    async def _open(self, ref: BlobRef, tenant_id: str) -> AsyncIterator[bytes]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        data = self._blobs.get((tenant_id, ref.digest))
        if data is None or ref.tenant_id != tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for offset in range(0, len(data), 64 * 1024):
            yield data[offset:offset + 64 * 1024]

    def open(self, ref: BlobRef, *, tenant_id: str) -> AsyncIterator[bytes]:
        return self._open(ref, tenant_id)


class DomainBlobRouter(DomainBlobStore):
    def __init__(self, stores: "dict[StorageDomain, BlobStore]") -> None:
        self._stores = dict(stores)

    def for_domain(self, domain: StorageDomain) -> BlobStore:
        try:
            return self._stores[domain]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def initialize(self) -> None:
        seen: set[int] = set()
        for store in self._stores.values():
            if id(store) not in seen:
                await store.initialize()
                seen.add(id(store))

    async def close(self) -> None:
        seen: set[int] = set()
        for store in reversed(tuple(self._stores.values())):
            if id(store) not in seen:
                await store.close()
                seen.add(id(store))


class FilesystemBlobStore(_Base, BlobStore):
    """Tenant-scoped streaming Blob store for filesystem runtimes."""

    def __init__(self, root: Path, namespace: str) -> None:
        super().__init__(namespace)
        self._root = root
        self._leases = FilesystemLeaseCoordinator(root / "leases")

    async def initialize(self) -> None:
        await super().initialize()
        self._transaction_directory.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._recover_uploads)

    async def put_bytes(self, *, tenant_id: str, data: bytes, expected_digest: str | None = None) -> BlobRef:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if len(data) > _MAX_INLINE_BLOB_BYTES:
            raise ValueError("put_stream is required for blobs larger than 4 MiB")
        digest = hashlib.sha256(data).hexdigest()
        if expected_digest is not None and expected_digest != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._publish(tenant_id, digest, data)

    async def put_stream(self, *, tenant_id: str, chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> BlobRef:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if expected_size < 0 or expected_size > _MAX_BLOB_BYTES:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _validate_blob_digest(expected_digest)
        target = self._body_path(tenant_id, expected_digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{expected_digest}.", dir=target.parent)
        upload_id = uuid.uuid4().hex
        journal = self._transaction_directory / f"{upload_id}.json"
        journal_payload: dict[str, JsonValue] = {
            "schema_version": 1,
            "record_type": "blob-upload",
            "namespace": self.namespace,
            "upload_id": upload_id,
            "tenant_id": tenant_id,
            "digest": expected_digest,
            "size": expected_size,
            "temporary": temporary,
            "body": str(target),
            "metadata": str(self._metadata_path(tenant_id, expected_digest)),
            "status": "PREPARED",
        }
        self._transaction_directory.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(write_json_atomic, journal, journal_payload, fsync=True)
        size = 0
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                async for chunk in chunks:
                    if not 1 <= len(chunk) <= 1024 * 1024 or size + len(chunk) > _MAX_BLOB_BYTES:
                        raise ValueError("invalid blob chunk")
                    await asyncio.to_thread(handle.write, chunk)
                    digest.update(chunk)
                    size += len(chunk)
                await asyncio.to_thread(handle.flush)
                await asyncio.to_thread(os.fsync, handle.fileno())
            if size != expected_size or digest.hexdigest() != expected_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            lease = await self._leases.acquire(f"blob:{tenant_id}:{expected_digest}")
            try:
                await asyncio.to_thread(self._commit_upload, journal, journal_payload, lease.fence)
            finally:
                await self._leases.release(lease)
            return BlobRef(tenant_id, expected_digest, size, f"file:{self.namespace}:{tenant_id}:{expected_digest}")
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            journal.unlink(missing_ok=True)
            raise

    async def stat(self, ref: BlobRef, *, tenant_id: str) -> BlobRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if ref.tenant_id != tenant_id:
            return None
        metadata = self._metadata(tenant_id, ref.digest)
        body = self._body_path(tenant_id, ref.digest)
        if metadata is None:
            if body.exists():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return None
        if metadata.get("tenant_id") != tenant_id or metadata.get("digest") != ref.digest or metadata.get("status") != "COMPLETED" or not body.is_file():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            if body.stat().st_size != int(metadata["size"]):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except (OSError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return BlobRef(tenant_id, ref.digest, int(metadata["size"]), ref.locator or f"file:{self.namespace}:{tenant_id}:{ref.digest}")

    def open(self, ref: BlobRef, *, tenant_id: str) -> AsyncIterator[bytes]:
        return self._open(ref, tenant_id)

    async def _open(self, ref: BlobRef, tenant_id: str) -> AsyncIterator[bytes]:
        blob = await self.stat(ref, tenant_id=tenant_id)
        if blob is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        path = self._body_path(tenant_id, blob.digest)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while True:
                chunk = await asyncio.to_thread(handle.read, 64 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                yield chunk
        if size != blob.size or digest.hexdigest() != blob.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _publish(self, tenant_id: str, digest: str, data: bytes) -> BlobRef:
        async def chunks() -> AsyncIterator[bytes]:
            yield data

        return await self.put_stream(tenant_id=tenant_id, chunks=chunks(), expected_size=len(data), expected_digest=digest)

    @property
    def _transaction_directory(self) -> Path:
        return self._root / "transactions" / "blob"

    def _commit_upload(self, journal: Path, payload: dict[str, JsonValue], fence: int) -> None:
        temporary = Path(str(payload["temporary"]))
        target = Path(str(payload["body"]))
        tenant_id = str(payload["tenant_id"])
        digest = str(payload["digest"])
        if not temporary.is_file():
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
        sync_directory(target.parent)
        self._write_metadata(tenant_id, digest, int(payload["size"]))
        committed = dict(payload)
        committed["status"] = "COMMITTED"
        committed["fence"] = fence
        write_json_atomic(journal, committed, fsync=True)
        journal.unlink(missing_ok=True)
        sync_directory(journal.parent)

    def _recover_uploads(self) -> None:
        for journal in sorted(self._transaction_directory.glob("*.json")):
            try:
                payload = read_json(journal)
                if payload.get("namespace") != self.namespace or payload.get("record_type") != "blob-upload":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                temporary = Path(str(payload["temporary"]))
                target = Path(str(payload["body"]))
                metadata = Path(str(payload["metadata"]))
                tenant_id = str(payload["tenant_id"])
                digest = str(payload["digest"])
                if target.is_file() and metadata.is_file():
                    journal.unlink(missing_ok=True)
                elif target.is_file() and not metadata.is_file() and not temporary.is_file():
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                elif metadata.is_file() and not target.is_file() and not temporary.is_file():
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif temporary.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, target)
                    self._write_metadata(tenant_id, digest, int(payload["size"]))
                    journal.unlink(missing_ok=True)
                else:
                    target.unlink(missing_ok=True)
                    metadata.unlink(missing_ok=True)
                    journal.unlink(missing_ok=True)
            except AIError:
                raise
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error

    def _body_path(self, tenant_id: str, digest: str) -> Path:
        _validate_blob_digest(digest)
        return self._root / "blobs" / _component(tenant_id) / digest[:2] / digest

    def _metadata_path(self, tenant_id: str, digest: str) -> Path:
        _validate_blob_digest(digest)
        return self._root / "blob-meta" / _component(tenant_id) / f"{digest}.json"

    def _metadata(self, tenant_id: str, digest: str) -> dict[str, JsonValue] | None:
        path = self._metadata_path(tenant_id, digest)
        if not path.is_file():
            return None
        try:
            value = read_json(path)
            if value.get("schema_version") != 1 or value.get("record_type") != "blob-meta" or value.get("namespace") != self.namespace:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            payload = value.get("payload")
            if not isinstance(payload, dict) or value.get("payload_sha256") != canonical_sha256(payload):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return payload
        except (OSError, TypeError, ValueError):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _write_metadata(self, tenant_id: str, digest: str, size: int) -> None:
        payload: dict[str, JsonValue] = {"tenant_id": tenant_id, "digest": digest, "size": size, "status": "COMPLETED"}
        write_json_atomic(
            self._metadata_path(tenant_id, digest),
            {
                "schema_version": 1,
                "record_type": "blob-meta",
                "namespace": self.namespace,
                "record_revision": 0,
                "payload": payload,
                "payload_sha256": canonical_sha256(payload),
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            fsync=True,
        )


class RoutedOperationRepository:
    """Route operation records to the owning domain target."""

    def __init__(self, durable: OperationLedgerRepository, memory: OperationLedgerRepository, persist: frozenset[StorageDomain], domain: StorageDomain) -> None:
        self._durable = durable
        self._memory = memory
        self._persist = persist
        self._domain = domain

    def _store(self, kind: ResourceKind) -> OperationLedgerRepository:
        if _operation_domain(kind) is not self._domain:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._durable if self._domain in self._persist else self._memory

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def append(self, record: OperationLedgerInput) -> OperationLedgerRecord:
        return await self._store(record.resource_kind).append(record)

    async def get(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None:
        record = await self._memory.get(operation_id, tenant_id=tenant_id)
        if record is not None:
            if _operation_domain(record.resource_kind) is not self._domain:
                return None
            return record
        if self._domain in self._persist:
            record = await self._durable.get(operation_id, tenant_id=tenant_id)
            if record is not None and _operation_domain(record.resource_kind) is not self._domain:
                return None
            return record
        return None

    async def compare_and_swap(self, operation_id: str, *, tenant_id: str, expected_status: OperationStatus, next_record: OperationLedgerRecord) -> OperationLedgerRecord:
        return await self._store(next_record.resource_kind).compare_and_swap(operation_id, tenant_id=tenant_id, expected_status=expected_status, next_record=next_record)

    async def list_pending(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, limit: int) -> tuple[OperationLedgerRecord, ...]:
        return await self._store(resource_kind).list_pending(resource_kind, resource_id, tenant_id=tenant_id, limit=limit)

    async def compact_terminal(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, through_sequence: int) -> str:
        return await self._store(resource_kind).compact_terminal(resource_kind, resource_id, tenant_id=tenant_id, through_sequence=through_sequence)


class RoutedIdempotencyRepository:
    def __init__(self, durable: "IdempotencyRepository", memory: "IdempotencyRepository", persist: frozenset[StorageDomain], domain: StorageDomain) -> None:
        self._durable = durable
        self._memory = memory
        self._persist = persist
        self._domain = domain

    def _store(self) -> "IdempotencyRepository":
        return self._durable if self._domain in self._persist else self._memory

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord:
        return await self._store().reserve(record)

    async def get(self, scope: str, key_hash: str, *, tenant_id: str) -> IdempotencyRecord | None:
        record = await self._memory.get(scope, key_hash, tenant_id=tenant_id)
        if record is not None or self._domain not in self._persist:
            return record
        return await self._durable.get(scope, key_hash, tenant_id=tenant_id)

    async def list_by_execution(self, execution_id: str, *, tenant_id: str) -> tuple[IdempotencyRecord, ...]:
        return await self._store().list_by_execution(execution_id, tenant_id=tenant_id)

    async def compare_and_swap(self, scope: str, key_hash: str, *, tenant_id: str, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord:
        return await self._store().compare_and_swap(scope, key_hash, tenant_id=tenant_id, expected_status=expected_status, next_record=next_record)


def _operation_domain(kind: ResourceKind) -> StorageDomain:
    return {
        ResourceKind.SESSION: StorageDomain.CONVERSATION,
        ResourceKind.EXECUTION: StorageDomain.EXECUTION,
        ResourceKind.MEMORY: StorageDomain.MEMORY,
        ResourceKind.ARTIFACT: StorageDomain.ARTIFACT,
        ResourceKind.TASK_GRAPH: StorageDomain.TASK,
        ResourceKind.EVALUATION: StorageDomain.EVALUATION,
        ResourceKind.DOWNLOAD_GRANT: StorageDomain.ARTIFACT,
        ResourceKind.APPROVAL: StorageDomain.RECOVERY,
        ResourceKind.EXTERNAL_CALL: StorageDomain.RECOVERY,
        ResourceKind.TOOL_OPERATION: StorageDomain.RECOVERY,
    }.get(kind, StorageDomain.RECOVERY)


def _route_runtime_stores(
    durable: RuntimeStores,
    memory: RuntimeStores,
    persist: frozenset[StorageDomain],
) -> RuntimeStores:
    def selected(domain: StorageDomain) -> bool:
        return domain in persist

    full_persistence = StorageDomain.durable() <= persist
    conversation_sessions = durable.conversation.sessions if selected(StorageDomain.CONVERSATION) else memory.conversation.sessions
    conversation_blobs = durable.conversation.blobs if full_persistence else memory.conversation.blobs
    execution_store = durable.execution if selected(StorageDomain.EXECUTION) else memory.execution
    memory_store = durable.memory if selected(StorageDomain.MEMORY) else memory.memory
    artifact_store = durable.artifact if selected(StorageDomain.ARTIFACT) else memory.artifact
    task_store = durable.task if selected(StorageDomain.TASK) else memory.task
    evaluation_store = durable.evaluation if selected(StorageDomain.EVALUATION) else memory.evaluation
    recovery_store = durable.recovery if selected(StorageDomain.RECOVERY) else memory.recovery

    def operation(domain: StorageDomain) -> RoutedOperationRepository:
        operation_stores = {
            StorageDomain.CONVERSATION: (durable.conversation, memory.conversation),
            StorageDomain.EXECUTION: (durable.execution, memory.execution),
            StorageDomain.MEMORY: (durable.memory, memory.memory),
            StorageDomain.ARTIFACT: (durable.artifact, memory.artifact),
            StorageDomain.TASK: (durable.task, memory.task),
            StorageDomain.EVALUATION: (durable.evaluation, memory.evaluation),
            StorageDomain.RECOVERY: (durable.recovery, memory.recovery),
        }
        durable_store, memory_store = operation_stores[domain]
        return RoutedOperationRepository(durable_store.operations, memory_store.operations, persist, domain)

    return RuntimeStores(
        namespace=durable.namespace,
        conversation=ConversationStore(conversation_sessions, operation(StorageDomain.CONVERSATION) if full_persistence else memory.conversation.operations, conversation_blobs),
        execution=ExecutionStore(execution_store.executions, execution_store.results, RoutedIdempotencyRepository(durable.execution.idempotency, memory.execution.idempotency, persist, StorageDomain.EXECUTION), execution_store.events, operation(StorageDomain.EXECUTION), execution_store.blobs),
        memory=MemoryStore(memory_store.records, operation(StorageDomain.MEMORY), memory_store.blobs),
        artifact=ArtifactStore(artifact_store.records, operation(StorageDomain.ARTIFACT), artifact_store.blobs),
        task=TaskStore(task_store.tasks, operation(StorageDomain.TASK)),
        evaluation=EvaluationStore(evaluation_store.records, RoutedIdempotencyRepository(durable.evaluation.idempotency, memory.evaluation.idempotency, persist, StorageDomain.EVALUATION), operation(StorageDomain.EVALUATION), evaluation_store.blobs),
        recovery=RecoveryStore(recovery_store.approvals, recovery_store.externals, recovery_store.checkpoints, operation(StorageDomain.RECOVERY), recovery_store.tools, recovery_store.blobs),
    )


class InMemoryRuntime:
    def __init__(
        self,
        persistence: RuntimeStores,
        components: tuple[RuntimeRepository, ...],
        *,
        operation_repositories: "Mapping[StorageDomain, OperationLedgerRepository] | None" = None,
    ) -> None:
        self.persistence = persistence
        self.components = components
        self.operation_repositories = {} if operation_repositories is None else dict(operation_repositories)
        self._evaluation_idempotency = components[14] if len(components) > 14 else None
        self._recovery_checkpoint = components[15] if len(components) > 15 else None
        self._initialized = False
        self._closed = False
        self._initialize_lock = asyncio.Lock()

    persistence: RuntimeStores
    components: tuple[RuntimeRepository, ...]
    operation_repositories: dict[StorageDomain, OperationLedgerRepository]

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            initialized: list[RuntimeRepository] = []
            try:
                for component in self.components:
                    if component not in initialized:
                        await component.initialize()
                        initialized.append(component)
                self._initialized = True
                self._closed = False
                _logger.debug("runtime stores initialized: namespace=%s", self.persistence.namespace)
            except BaseException:
                for component in reversed(initialized):
                    try:
                        await component.close()
                    except BaseException:
                        pass
                raise

    async def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        for component in reversed(self.components):
            try:
                await component.close()
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure
        self._closed = True


class _DurableRuntime(InMemoryRuntime):
    def __init__(self, persistence: RuntimeStores, components: tuple[RuntimeRepository, ...], state_path: str, persist: frozenset[StorageDomain], writer_lock: FilesystemWriterLock | None = None, durable_components: tuple[RuntimeRepository, ...] | None = None, operation_repositories: "Mapping[StorageDomain, OperationLedgerRepository] | None" = None) -> None:
        super().__init__(persistence, components, operation_repositories=operation_repositories)
        self._durable_components = components if durable_components is None else durable_components
        self._state_path = state_path
        self._persist = persist
        self._writer_lock = writer_lock or FilesystemWriterLock(Path(state_path).parent / "runtime.lock")
        self._transaction_lock = _SharedTransactionLock()
        self._commit_lock = threading.Lock()
        for index, component in enumerate(self._durable_components):
            domain = _COMPONENT_DOMAINS[index] if index < len(_COMPONENT_DOMAINS) else None
            _configure_transaction_component(component, self._transaction_lock, domain)
        self._transaction_lock.configure(
            commit=self._commit_domain,
            rollback=self._rollback_domains,
            reject_cross_domain=True,
        )
        self._durable_initialize_lock = asyncio.Lock()

    @property
    def writer_lock(self) -> FilesystemWriterLock:
        return self._writer_lock

    @property
    def runtime_root(self) -> Path:
        return Path(self._state_path).parent

    async def initialize(self) -> None:
        async with self._durable_initialize_lock:
            if self._initialized:
                return
            Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
            await self._writer_lock.acquire()
            try:
                await asyncio.to_thread(self._load)
                await super().initialize()
                if not Path(self._state_path).is_file():
                    self._write_manifest()
            except BaseException:
                await self._writer_lock.release()
                raise

    async def close(self) -> None:
        if self._closed:
            await self._writer_lock.release()
            return
        try:
            await super().close()
        finally:
            await self._writer_lock.release()
        _logger.debug("filesystem runtime stores closed: namespace=%s", self.persistence.namespace)

    def _rollback_domains(self, domains: frozenset[StorageDomain]) -> None:
        with self._commit_lock:
            for domain in domains:
                self._load_domain(domain)

    def _commit_domain(self, domain: StorageDomain) -> None:
        with self._commit_lock:
            if self._closed:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            try:
                self._flush_domain(domain)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED:
                    self._closed = True
                    raise
                self._load_domain(domain)
                raise
            except BaseException as error:
                self._load_domain(domain)
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    def _load_payload(self, value: dict[str, JsonValue], *, clear: bool = True) -> None:
        sessions = self.components[0]
        executions = self.components[1]
        results = self.components[2]
        idempotency = self.components[3]
        evaluation_idempotency = self._evaluation_idempotency
        recovery_checkpoint = self._recovery_checkpoint
        events = self.components[4]
        tasks = self.components[5]
        evaluations = self.components[6]
        memories = self.components[7]
        artifacts = self.components[8]
        approvals = self.components[9]
        externals = self.components[10]
        tools = self.components[12]
        if not isinstance(sessions, _SessionRepository) or not isinstance(executions, _ExecutionRepository) or not isinstance(idempotency, _IdempotencyRepository) or not isinstance(evaluation_idempotency, _IdempotencyRepository):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if clear:
            self._clear_payload()
        for raw in value.get("sessions", []):
            sessions._records[(str(raw["tenant_id"]), str(raw["session_id"]))] = _session_from_json(raw)
        for raw in value.get("executions", []):
            executions._records[(str(raw["tenant_id"]), str(raw["execution_id"]))] = _execution_from_json(raw)
        for raw in value.get("idempotency", []):
            record = _idempotency_from_json(raw)
            target = evaluation_idempotency if record.scope == "evaluation.run" else idempotency
            target._records[(record.tenant_id, record.scope, str(raw["key_hash"]))] = record
        if isinstance(recovery_checkpoint, _RecoveryCheckpointRepository):
            for raw in value.get("recovery_checkpoints", []):
                record = _recovery_checkpoint_from_json(raw)
                recovery_checkpoint._records[(record.tenant_id, record.checkpoint_id)] = record
        if isinstance(results, _ResultRepository):
            for raw in value.get("results", []):
                record = _result_from_json(raw)
                results._results[(record.tenant_id, record.execution_id)] = record
        if isinstance(events, _EventRepository):
            for raw in value.get("events", []):
                record = _event_from_json(raw)
                events._items.setdefault((record.tenant_id, record.execution_id), []).append(record)
        if isinstance(tasks, _TaskRepository):
            for raw in value.get("task_plans", []):
                tenant_id = str(raw["tenant_id"])
                view = _task_plan_from_json(raw["view"])
                tasks._plans[(tenant_id, view.graph_id)] = view
            for raw in value.get("task_nodes", []):
                node = _task_node_from_json(raw["node"])
                tasks._nodes[(str(raw["tenant_id"]), node.graph_id, node.task_id)] = node
        if isinstance(evaluations, _EvaluationRepository):
            for raw in value.get("evaluations", []):
                record = _evaluation_from_json(raw)
                evaluations._records[(record.tenant_id, record.evaluation_id)] = record
        if isinstance(memories, _MemoryRepository):
            for raw in value.get("memories", []):
                record = _memory_from_json(raw)
                memories._records[(record.tenant_id, record.memory_id)] = record
        if isinstance(artifacts, _ArtifactRepository):
            for raw in value.get("artifacts", []):
                record = _artifact_from_json(raw)
                artifacts._records[(record.tenant_id, record.artifact_id)] = record
        if isinstance(approvals, _ApprovalRepository):
            for raw in value.get("approvals", []):
                record = _approval_from_json(raw)
                approvals._records[(record.tenant_id, record.approval_id)] = record
        if isinstance(externals, _ExternalRepository):
            for raw in value.get("externals", []):
                record = _external_from_json(raw)
                externals._records[(record.tenant_id, record.call_id)] = record
        for raw in value.get("operations", []):
            record = _operation_from_json(raw)
            operation_repository = self.operation_repositories.get(_operation_storage_domain(record.resource_kind.value))
            if isinstance(operation_repository, _OperationRepository):
                operation_repository._records[(record.tenant_id, record.operation_id)] = record
        if isinstance(tools, _ToolRepository):
            for raw in value.get("tools", []):
                record = _tool_from_json(raw)
                tools._records[(record.tenant_id, record.operation_id)] = record

    def _clear_payload(self) -> None:
        for component in self._durable_components:
            if isinstance(component, _SessionRepository):
                component._records.clear()
            elif isinstance(component, _ExecutionRepository):
                component._records.clear()
            elif isinstance(component, _ResultRepository):
                component._results.clear()
            elif isinstance(component, _IdempotencyRepository):
                component._records.clear()
            elif isinstance(component, _RecoveryCheckpointRepository):
                component._records.clear()
            elif isinstance(component, _EventRepository):
                component._items.clear()
            elif isinstance(component, _TaskRepository):
                component._plans.clear()
                component._nodes.clear()
            elif isinstance(component, _EvaluationRepository):
                component._records.clear()
            elif isinstance(component, _MemoryRepository):
                component._records.clear()
            elif isinstance(component, _ArtifactRepository):
                component._records.clear()
            elif isinstance(component, _ApprovalRepository):
                component._records.clear()
            elif isinstance(component, _ExternalRepository):
                component._records.clear()
            elif isinstance(component, _OperationRepository):
                component._records.clear()
            elif isinstance(component, _ToolRepository):
                component._records.clear()
            elif isinstance(component, InMemoryBlobStore):
                component._blobs.clear()

    def _write_manifest(self) -> None:
        manifest = {
            "format": "linktools-ai-runtime",
            "generation": 1,
            "namespace": self.persistence.namespace,
        }
        write_json_atomic(Path(self._state_path), manifest, fsync=True)

    def _flush_domain(self, domain: StorageDomain) -> None:
        if domain is StorageDomain.ASSET:
            return
        root = Path(self._state_path).parent
        directory = root / domain.value
        directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(directory / "records.json", self._domain_records(domain), fsync=True)
        _logger.debug("runtime domain committed: namespace=%s domain=%s", self.persistence.namespace, domain.value)

    def _clear_domain(self, domain: StorageDomain) -> None:
        for component in self._durable_components:
            if isinstance(component, _Base) and component._owner_domain is domain:
                if isinstance(component, _SessionRepository):
                    component._records.clear()
                elif isinstance(component, _ExecutionRepository):
                    component._records.clear()
                elif isinstance(component, _ResultRepository):
                    component._results.clear()
                elif isinstance(component, _IdempotencyRepository):
                    component._records.clear()
                elif isinstance(component, _RecoveryCheckpointRepository):
                    component._records.clear()
                elif isinstance(component, _EventRepository):
                    component._items.clear()
                elif isinstance(component, _TaskRepository):
                    component._plans.clear()
                    component._nodes.clear()
                elif isinstance(component, _EvaluationRepository):
                    component._records.clear()
                elif isinstance(component, _MemoryRepository):
                    component._records.clear()
                elif isinstance(component, _ArtifactRepository):
                    component._records.clear()
                elif isinstance(component, _ApprovalRepository):
                    component._records.clear()
                elif isinstance(component, _ExternalRepository):
                    component._records.clear()
                elif isinstance(component, _OperationRepository):
                    component._records.clear()
                elif isinstance(component, _ToolRepository):
                    component._records.clear()

    def _load_domain(self, domain: StorageDomain) -> None:
        self._clear_domain(domain)
        if domain is StorageDomain.ASSET:
            return
        records_path = Path(self._state_path).parent / domain.value / "records.json"
        if not records_path.is_file():
            return
        records = read_json(records_path)
        if not isinstance(records, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _validate_state_record_uniqueness(_domain_validation_records(records))
        domain_payload = _empty_payload()
        _merge_domain_payload(domain_payload, records)
        self._load_payload(domain_payload, clear=False)

    def _domain_records(self, domain: StorageDomain) -> dict[str, JsonValue]:
        sessions, executions, results, idempotency, events, tasks, evaluations, memories, artifacts, approvals, externals, _operations, tools = self.components[:13]
        evaluation_idempotency = self._evaluation_idempotency
        checkpoints = self._recovery_checkpoint
        if not isinstance(idempotency, _IdempotencyRepository) or not isinstance(evaluation_idempotency, _IdempotencyRepository):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        operation_repository = self.operation_repositories.get(domain)
        operation_values = [] if not isinstance(operation_repository, _OperationRepository) else [
            _json_record(item)
            for item in operation_repository._records.values()
        ]
        values: dict[str, JsonValue] = {}
        if domain is not StorageDomain.CONVERSATION or StorageDomain.durable() <= self._persist:
            values["operations"] = operation_values
        if domain is StorageDomain.CONVERSATION:
            values["sessions"] = [_record_json(item) for item in sessions._records.values()]
        elif domain is StorageDomain.EXECUTION:
            values["executions"] = [_record_json(item) for item in executions._records.values()]
            values["results"] = [] if not isinstance(results, _ResultRepository) else [_json_record(item) for item in results._results.values()]
            values["events"] = [] if not isinstance(events, _EventRepository) else [_json_record(item) for items in events._items.values() for item in items]
            values["idempotency"] = [_idempotency_json(record, key_hash) for (tenant_id, scope, key_hash), record in idempotency._records.items() if scope != "evaluation.run"]
        elif domain is StorageDomain.MEMORY:
            values["memories"] = [] if not isinstance(memories, _MemoryRepository) else [_json_record(item) for item in memories._records.values()]
        elif domain is StorageDomain.ARTIFACT:
            values["artifacts"] = [] if not isinstance(artifacts, _ArtifactRepository) else [_json_record(item) for item in artifacts._records.values()]
        elif domain is StorageDomain.TASK:
            values["tasks"] = [] if not isinstance(tasks, _TaskRepository) else [
                *({"record_type": "plan", "tenant_id": tenant_id, "view": _json_record(view)} for (tenant_id, _), view in tasks._plans.items()),
                *({"record_type": "node", "tenant_id": tenant_id, "node": _json_record(node)} for (tenant_id, _, _), node in tasks._nodes.items()),
            ]
        elif domain is StorageDomain.EVALUATION:
            values["evaluations"] = [] if not isinstance(evaluations, _EvaluationRepository) else [_json_record(item) for item in evaluations._records.values()]
            values["idempotency"] = [_idempotency_json(record, key_hash) for (tenant_id, scope, key_hash), record in evaluation_idempotency._records.items() if scope == "evaluation.run"]
        elif domain is StorageDomain.RECOVERY:
            values["approvals"] = [] if not isinstance(approvals, _ApprovalRepository) else [_json_record(item) for item in approvals._records.values()]
            values["externals"] = [] if not isinstance(externals, _ExternalRepository) else [_json_record(item) for item in externals._records.values()]
            values["tools"] = [] if not isinstance(tools, _ToolRepository) else [_json_record(item) for item in tools._records.values()]
            values["recovery_checkpoints"] = [] if not isinstance(checkpoints, _RecoveryCheckpointRepository) else [_json_record(item) for item in checkpoints._records.values()]
        return values


class FilesystemRuntime(_DurableRuntime):
    """Durable filesystem runtime with one independent file per domain."""

    def _load(self) -> None:
        state_path = Path(self._state_path)
        if not state_path.is_file():
            if any(item.name != "runtime.lock" for item in state_path.parent.iterdir()):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._load_payload(_empty_payload())
            return
        try:
            state = read_json(state_path)
            if state.get("format") != "linktools-ai-runtime" or state.get("generation") != 1 or state.get("namespace") != self.persistence.namespace:
                raise ValueError("runtime state identity mismatch")
            self._load_payload(_empty_payload())
            for domain in self._persist:
                self._load_domain(domain)
            self._validate_payload()
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            await super().initialize()
        except BaseException:
            self._initialized = False
            await InMemoryRuntime.close(self)
            raise

    def _validate_payload(self) -> None:
        executions = self.components[1]
        results = self.components[2]
        if isinstance(executions, _ExecutionRepository) and isinstance(results, _ResultRepository):
            for key, result in results._results.items():
                execution = executions._records.get(key)
                if execution is None or execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} or execution.status is not result.status:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for key, execution in executions._records.items():
                if execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} and key not in results._results:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for component in self.components:
            if isinstance(component, _EventRepository):
                for key, records in component._items.items():
                    if [item.sequence for item in records] != list(range(1, len(records) + 1)):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    execution = executions._records.get(key) if isinstance(executions, _ExecutionRepository) else None
                    if execution is None or execution.event_sequence != len(records):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_state_record_uniqueness(records: "dict[str, JsonValue]") -> None:
    keys = {
        "sessions": ("tenant_id", "session_id"),
        "executions": ("tenant_id", "execution_id"),
        "results": ("tenant_id", "execution_id"),
        "idempotency": ("tenant_id", "scope", "key_hash"),
        "events": ("tenant_id", "execution_id", "sequence"),
        "evaluations": ("tenant_id", "evaluation_id"),
        "memories": ("tenant_id", "memory_id"),
        "artifacts": ("tenant_id", "artifact_id"),
        "approvals": ("tenant_id", "approval_id"),
        "externals": ("tenant_id", "call_id"),
        "operations": ("tenant_id", "operation_id"),
        "tools": ("tenant_id", "operation_id"),
    }
    for name, fields in keys.items():
        seen: set[tuple[str, ...]] = set()
        for item in records[name]:
            if not isinstance(item, dict):
                raise ValueError(f"runtime {name} record must be an object")
            try:
                identity = tuple(str(item[field]) for field in fields)
            except KeyError as error:
                raise ValueError(f"runtime {name} identity is incomplete") from error
            if identity in seen:
                raise ValueError(f"runtime {name} identity is duplicated")
            seen.add(identity)
    for item in records["idempotency"]:
        if re.fullmatch(r"[0-9a-f]{64}", str(item["key_hash"])) is None:
            raise ValueError("runtime idempotency key hash is invalid")
    task_keys: set[tuple[str, str, str]] = set()
    for item in records["tasks"]:
        if not isinstance(item, dict) or item.get("record_type") not in {"plan", "node"} or not isinstance(item.get("tenant_id"), str):
            raise ValueError("runtime task identity is invalid")
        payload = item.get("view") if item["record_type"] == "plan" else item.get("node")
        if not isinstance(payload, dict):
            raise ValueError("runtime task payload is invalid")
        identity = (str(item["tenant_id"]), str(payload.get("graph_id")), str(payload.get("task_id", "")))
        if identity in task_keys:
            raise ValueError("runtime task identity is duplicated")
        task_keys.add(identity)


def _domain_validation_records(records: dict[str, JsonValue]) -> dict[str, JsonValue]:
    empty = {"sessions": [], "executions": [], "results": [], "idempotency": [], "events": [], "tasks": [], "evaluations": [], "memories": [], "artifacts": [], "approvals": [], "externals": [], "operations": [], "tools": []}
    for key in empty:
        if key in records:
            empty[key] = records[key]
    return empty


def _empty_payload() -> dict[str, JsonValue]:
    return {
        "sessions": [], "executions": [], "results": [], "idempotency": [], "events": [],
        "task_plans": [], "task_nodes": [], "evaluations": [], "memories": [], "artifacts": [],
        "approvals": [], "externals": [], "recovery_checkpoints": [], "operations": [], "tools": [],
    }


def _domain_payload(payload: dict[str, JsonValue], domain: StorageDomain) -> dict[str, JsonValue]:
    keys: dict[StorageDomain, tuple[str, ...]] = {
        StorageDomain.CONVERSATION: ("sessions", "operations"),
        StorageDomain.EXECUTION: ("executions", "results", "idempotency", "events", "operations"),
        StorageDomain.MEMORY: ("memories", "operations"),
        StorageDomain.ARTIFACT: ("artifacts",),
        StorageDomain.TASK: ("task_plans", "task_nodes", "operations"),
        StorageDomain.EVALUATION: ("evaluations", "idempotency", "operations"),
        StorageDomain.RECOVERY: ("approvals", "externals", "recovery_checkpoints", "tools", "operations"),
        StorageDomain.ASSET: (),
    }
    values = {key: payload[key] for key in keys[domain] if key != "operations"}
    if domain is StorageDomain.EXECUTION:
        values["idempotency"] = [item for item in values["idempotency"] if isinstance(item, dict) and item.get("scope") != "evaluation.run"]
    elif domain is StorageDomain.EVALUATION:
        values["idempotency"] = [item for item in values["idempotency"] if isinstance(item, dict) and item.get("scope") == "evaluation.run"]
    if "operations" in keys[domain]:
        values["operations"] = [
            item for item in payload["operations"]
            if isinstance(item, dict) and _operation_storage_domain(item.get("resource_kind")) is domain
        ]
    values["tasks"] = [
        {"record_type": "plan", **item}
        for item in values.pop("task_plans", [])
        if isinstance(item, dict)
    ] + [
        {"record_type": "node", **item}
        for item in values.pop("task_nodes", [])
        if isinstance(item, dict)
    ]
    return values


def _operation_storage_domain(value: object) -> StorageDomain:
    return {
        "SESSION": StorageDomain.CONVERSATION,
        "EXECUTION": StorageDomain.EXECUTION,
        "MEMORY": StorageDomain.MEMORY,
        "ARTIFACT": StorageDomain.ARTIFACT,
        "TASK_GRAPH": StorageDomain.TASK,
        "EVALUATION": StorageDomain.EVALUATION,
        "DOWNLOAD_GRANT": StorageDomain.ARTIFACT,
        "APPROVAL": StorageDomain.RECOVERY,
        "EXTERNAL_CALL": StorageDomain.RECOVERY,
        "TOOL_OPERATION": StorageDomain.RECOVERY,
    }.get(str(value), StorageDomain.RECOVERY)


def _merge_domain_payload(payload: dict[str, JsonValue], records: dict[str, JsonValue]) -> None:
    for key, value in records.items():
        if key == "tasks":
            if not isinstance(value, list):
                raise ValueError("runtime task records must be a list")
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("runtime task record must be an object")
                target = "task_plans" if item.get("record_type") == "plan" else "task_nodes" if item.get("record_type") == "node" else None
                if target is None:
                    raise ValueError("runtime task record type is invalid")
                payload[target].append({key: value for key, value in item.items() if key != "record_type"})
        elif key in payload:
            if not isinstance(value, list):
                raise ValueError("runtime domain records must be lists")
            payload[key].extend(value)
def _record_json(record: SessionRecord | ExecutionRecord | IdempotencyRecord) -> dict[str, JsonValue]:
    value = asdict(record)
    return _json_value(value)


def _json_value(value: "JsonValue | datetime | Enum") -> JsonValue:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum) and isinstance(value.value, str):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported persisted value: {type(value).__name__}")


def _time(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def _session_from_json(value: dict[str, JsonValue]) -> SessionRecord:
    raw_cursor = value.get("continuation")
    continuation = None if raw_cursor is None else ConversationCursor(str(raw_cursor["run_id"]), int(raw_cursor["revision"]))
    return SessionRecord(session_id=str(value["session_id"]), tenant_id=str(value["tenant_id"]), owner_principal_id=str(value["owner_principal_id"]), binding_digest=str(value["binding_digest"]), status=SessionStatus(str(value["status"])), revision=int(value["revision"]), resource_generation=int(value["resource_generation"]), cwd=None if value.get("cwd") is None else str(value["cwd"]), metadata=value.get("metadata", {}), created_at=_time(value["created_at"]), updated_at=_time(value["updated_at"]), closed_at=None if value.get("closed_at") is None else _time(value["closed_at"]), continuation=continuation)


def _execution_from_json(value: dict[str, JsonValue]) -> ExecutionRecord:
    return ExecutionRecord(execution_id=str(value["execution_id"]), tenant_id=str(value["tenant_id"]), session_id=None if value.get("session_id") is None else str(value["session_id"]), binding_digest=str(value["binding_digest"]), parent_execution_id=None if value.get("parent_execution_id") is None else str(value["parent_execution_id"]), root_execution_id=str(value["root_execution_id"]), source_execution_id=None if value.get("source_execution_id") is None else str(value["source_execution_id"]), base_execution_id=None if value.get("base_execution_id") is None else str(value["base_execution_id"]), lineage_kind=ExecutionLineageKind(str(value.get("lineage_kind", "RUN"))), status=ExecutionStatus(str(value["status"])), revision=int(value["revision"]), event_sequence=int(value["event_sequence"]), agent_run_sequence=int(value.get("agent_run_sequence", 0)), result_ref=None if value.get("result_ref") is None else str(value["result_ref"]), result_digest=None if value.get("result_digest") is None else str(value["result_digest"]), error_code=None if value.get("error_code") is None else str(value["error_code"]), safe_error_details=value.get("safe_error_details", {}), created_at=_time(value["created_at"]), updated_at=_time(value["updated_at"]), memory_namespace=None if value.get("memory_namespace") is None else str(value["memory_namespace"]), conversation_run_id=None if value.get("conversation_run_id") is None else str(value["conversation_run_id"]))


def _idempotency_from_json(value: dict[str, JsonValue]) -> IdempotencyRecord:
    key_hash = str(value["key_hash"])
    if re.fullmatch(r"[0-9a-f]{64}", key_hash) is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return IdempotencyRecord(str(value["tenant_id"]), str(value["scope"]), key_hash, str(value["request_digest"]), str(value["execution_id"]), IdempotencyStatus(str(value["status"])), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), _time(value["created_at"]), _time(value["updated_at"]))


def _recovery_checkpoint_from_json(value: dict[str, JsonValue]) -> RecoveryCheckpoint:
    return RecoveryCheckpoint(
        str(value["checkpoint_id"]),
        str(value["execution_id"]),
        str(value["tenant_id"]),
        str(value["binding_digest"]),
        str(value["run_id"]),
        str(value["phase"]),
        None if value.get("pending_operation_id") is None else str(value["pending_operation_id"]),
        value.get("payload", {}),
        int(value["revision"]),
        _time(value["created_at"]),
        _time(value["updated_at"]),
        value.get("approval_state", {}),
        value.get("external_state", {}),
        value.get("tool_effect_state", {}),
        value.get("idempotency_state", {}),
        value.get("terminal_handoff"),
    )


def _json_record(record: object) -> dict[str, JsonValue]:
    return _json_value(asdict(record))


def _idempotency_json(record: IdempotencyRecord, key_hash: str) -> dict[str, JsonValue]:
    value = _record_json(record)
    value["key_hash"] = key_hash
    return value


def _result_from_json(value: dict[str, JsonValue]) -> ResultRecord:
    return ResultRecord(str(value["execution_id"]), str(value["tenant_id"]), ExecutionStatus(str(value["status"])), str(value["output_schema_id"]), int(value["output_schema_revision"]), str(value["output_schema_fingerprint"]), None if value.get("payload_ref") is None else str(value["payload_ref"]), None if value.get("payload_digest") is None else str(value["payload_digest"]), StopReason(str(value["stop_reason"])), int(value["input_tokens"]), int(value["output_tokens"]), int(value["total_cost_micros"]), _time(str(value["created_at"])))


def _event_from_json(value: dict[str, JsonValue]) -> ExecutionEventRecord:
    return ExecutionEventRecord(str(value["execution_id"]), str(value["tenant_id"]), int(value["sequence"]), ExecutionEventType(str(value["event_type"])), value.get("payload", {}))


def _task_plan_from_json(value: dict[str, JsonValue]) -> TaskGraphView:
    nodes = tuple(TaskNode(str(item["task_id"]), tuple(item.get("dependencies", [])), None if item.get("binding_digest") is None else str(item["binding_digest"]), int(item.get("budget_cost", 1))) for item in value.get("nodes", []))
    return TaskGraphView(str(value["graph_id"]), TaskStatus(str(value["status"])), nodes)


def _task_node_from_json(value: dict[str, JsonValue]) -> TaskNodeView:
    return TaskNodeView(str(value["graph_id"]), str(value["task_id"]), tuple(value.get("dependencies", [])), TaskStatus(str(value["status"])), None if value.get("owner") is None else str(value["owner"]), int(value["fence"]), None if value.get("lease_expires_at") is None else _time(str(value["lease_expires_at"])), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), None if value.get("error_digest") is None else str(value["error_digest"]), None if value.get("execution_id") is None else str(value["execution_id"]))


def _evaluation_from_json(value: dict[str, JsonValue]) -> EvaluationRecord:
    return EvaluationRecord(str(value["evaluation_id"]), str(value["tenant_id"]), str(value["execution_id"]), str(value["dataset_id"]), int(value["dataset_revision"]), str(value["evaluator_id"]), int(value["evaluator_revision"]), str(value["binding_digest"]), str(value["output_schema_fingerprint"]), None if value.get("artifact_digest") is None else str(value["artifact_digest"]), EvaluationStatus(str(value["status"])), int(value["revision"]), value.get("metrics", {}), _time(str(value["created_at"])), _time(str(value["updated_at"])))


def _memory_from_json(value: dict[str, JsonValue]) -> MemoryRecord:
    return MemoryRecord(str(value["memory_id"]), str(value["tenant_id"]), str(value["memory_namespace_key"]), str(value["content_ref"]), str(value["content_digest"]), value.get("metadata", {}), int(value["revision"]), _time(str(value["created_at"])), _time(str(value["updated_at"])))


def _artifact_from_json(value: dict[str, JsonValue]) -> ArtifactRecord:
    return ArtifactRecord(str(value["artifact_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["producer"]), str(value["media_type"]), int(value["size"]), str(value["digest"]), str(value["blob_ref"]), _time(str(value["created_at"])))


def _approval_from_json(value: dict[str, JsonValue]) -> ApprovalRecord:
    return ApprovalRecord(str(value["approval_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["operation_id"]), ApprovalStatus(str(value["status"])), None if value.get("decision_id") is None else str(value["decision_id"]), None if value.get("decision") is None else ApprovalDecision(str(value["decision"])), None if value.get("decided_by") is None else str(value["decided_by"]), None if value.get("decision_digest") is None else str(value["decision_digest"]), _time(str(value["created_at"])), None if value.get("decided_at") is None else _time(str(value["decided_at"])))


def _external_from_json(value: dict[str, JsonValue]) -> ExternalResultRecord:
    return ExternalResultRecord(str(value["call_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["operation_id"]), ExternalCallStatus(str(value["status"])), None if value.get("result_id") is None else str(value["result_id"]), None if value.get("payload_ref") is None else str(value["payload_ref"]), None if value.get("payload_digest") is None else str(value["payload_digest"]), _time(str(value["created_at"])), None if value.get("supplied_at") is None else _time(str(value["supplied_at"])))


def _operation_from_json(value: dict[str, JsonValue]) -> OperationLedgerRecord:
    return OperationLedgerRecord(str(value["operation_id"]), str(value["tenant_id"]), ResourceKind(str(value["resource_kind"])), str(value["resource_id"]), None if value.get("execution_id") is None else str(value["execution_id"]), OperationKind(str(value["kind"])), OperationStatus(str(value["status"])), str(value["request_digest"]), None if value.get("result_ref") is None else str(value["result_ref"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), bool(value["compactable"]), int(value["sequence"]), _time(str(value["created_at"])), _time(str(value["updated_at"])))


def _tool_from_json(value: dict[str, JsonValue]) -> ToolOperationRecord:
    return ToolOperationRecord(str(value["operation_id"]), str(value["tenant_id"]), str(value["run_id"]), str(value["tool_call_id"]), str(value["idempotency_key_hash"]), str(value["tool_name"]), str(value["arguments_hash"]), str(value["binding_fingerprint"]), bool(value["replay_safe"]), ToolOperationStatus(str(value["status"])), None if value.get("owner") is None else str(value["owner"]), int(value["fence"]), None if value.get("lease_expires_at") is None else _time(str(value["lease_expires_at"])), None if value.get("result_ref") is None else str(value["result_ref"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), _time(str(value["created_at"])), _time(str(value["updated_at"])))


def _component(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).rstrip(b"=").decode("ascii")


def _validate_blob_digest(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _operation_immutable(record: OperationLedgerRecord) -> tuple[object, ...]:
    return (
        record.tenant_id, record.resource_kind, record.resource_id, record.execution_id,
        record.kind, record.request_digest, record.result_ref, record.result_digest,
        record.error_code, record.compactable,
    )


def _operation_input_immutable(record: OperationLedgerInput) -> tuple[object, ...]:
    return (
        record.tenant_id, record.resource_kind, record.resource_id, record.execution_id,
        record.kind, record.request_digest, record.result_ref, record.result_digest,
        record.error_code, record.compactable,
    )


def _build_runtime(*, filesystem: bool, namespace: str, state_path: "str | None" = None, persist: "frozenset[StorageDomain] | StorageDomain | None" = None, writer_lock: "FilesystemWriterLock | None" = None) -> InMemoryRuntime:
    validate_persistence_namespace(namespace)
    selected_domains = _normalize_runtime_persist(persist, filesystem=filesystem)
    sessions = _SessionRepository(namespace)
    executions = _ExecutionRepository(namespace)
    execution_idempotency = _IdempotencyRepository(namespace)
    evaluation_idempotency = _IdempotencyRepository(namespace)
    recovery_checkpoint = _RecoveryCheckpointRepository(namespace)
    blob_domains = (StorageDomain.CONVERSATION, StorageDomain.EXECUTION, StorageDomain.MEMORY, StorageDomain.ARTIFACT, StorageDomain.EVALUATION, StorageDomain.RECOVERY)
    full_persistence = StorageDomain.durable() <= selected_domains
    if filesystem and state_path is not None:
        blob_stores = {
            domain: FilesystemBlobStore(Path(state_path).parent / domain.value / "blob", namespace)
            if domain in selected_domains and (domain is not StorageDomain.CONVERSATION or full_persistence)
            else InMemoryBlobStore(namespace)
            for domain in blob_domains
        }
    else:
        blob_stores = {domain: InMemoryBlobStore(namespace) for domain in blob_domains}
    blob_router = DomainBlobRouter(blob_stores)
    operation_repositories = {
        domain: _OperationRepository(namespace)
        for domain in (
            StorageDomain.CONVERSATION,
            StorageDomain.EXECUTION,
            StorageDomain.MEMORY,
            StorageDomain.ARTIFACT,
            StorageDomain.TASK,
            StorageDomain.EVALUATION,
            StorageDomain.RECOVERY,
        )
    }
    execution_operations = operation_repositories[StorageDomain.EXECUTION]
    components: tuple[RuntimeRepository, ...] = (
        sessions,
        executions,
        _ResultRepository(executions, namespace),
        execution_idempotency,
        _EventRepository(executions, namespace),
        _TaskRepository(namespace),
        _EvaluationRepository(namespace),
        _MemoryRepository(namespace),
        _ArtifactRepository(namespace),
        _ApprovalRepository(namespace),
        _ExternalRepository(namespace),
        execution_operations,
        _ToolRepository(namespace),
        blob_router,
        evaluation_idempotency,
        recovery_checkpoint,
        operation_repositories[StorageDomain.CONVERSATION],
        operation_repositories[StorageDomain.MEMORY],
        operation_repositories[StorageDomain.ARTIFACT],
        operation_repositories[StorageDomain.TASK],
        operation_repositories[StorageDomain.EVALUATION],
        operation_repositories[StorageDomain.RECOVERY],
    )
    executions.bind_start_repositories(components[3], components[4], components[11])
    components[2].bind_terminal_repositories(components[3], components[4], components[11])
    if isinstance(components[7], _MemoryRepository):
        components[7].bind_operation_repository(operation_repositories[StorageDomain.MEMORY])
    transaction_lock = _SharedTransactionLock()
    for index, component in enumerate(components):
        _configure_transaction_component(component, transaction_lock, _COMPONENT_DOMAINS[index])
    persistence = RuntimeStores(
        namespace=namespace,
        conversation=ConversationStore(sessions, operation_repositories[StorageDomain.CONVERSATION], blob_router.for_domain(StorageDomain.CONVERSATION)),
        execution=ExecutionStore(executions, components[2], components[3], components[4], execution_operations, blob_router.for_domain(StorageDomain.EXECUTION)),
        memory=MemoryStore(components[7], operation_repositories[StorageDomain.MEMORY], blob_router.for_domain(StorageDomain.MEMORY)),
        artifact=ArtifactStore(components[8], operation_repositories[StorageDomain.ARTIFACT], blob_router.for_domain(StorageDomain.ARTIFACT)),
        task=TaskStore(components[5], operation_repositories[StorageDomain.TASK]),
        evaluation=EvaluationStore(components[6], evaluation_idempotency, operation_repositories[StorageDomain.EVALUATION], blob_router.for_domain(StorageDomain.EVALUATION)),
        recovery=RecoveryStore(components[9], components[10], recovery_checkpoint, operation_repositories[StorageDomain.RECOVERY], components[12], blob_router.for_domain(StorageDomain.RECOVERY)),
    )
    if filesystem:
        if state_path is None:
            raise ValueError("filesystem runtime requires a state path")
        memory_runtime = _build_runtime(filesystem=False, namespace=namespace)
        routed = _route_runtime_stores(persistence, memory_runtime.persistence, selected_domains)
        return FilesystemRuntime(
            routed,
            (*components, *memory_runtime.components),
            state_path,
            selected_domains,
            writer_lock,
            durable_components=components,
            operation_repositories=operation_repositories,
        )
    return InMemoryRuntime(persistence, components, operation_repositories=operation_repositories)


def build_in_memory_runtime(*, namespace: "str | None" = None) -> InMemoryRuntime:
    selected_namespace = f"memory-{uuid.uuid4().hex}" if namespace is None else namespace
    return _build_runtime(filesystem=False, namespace=selected_namespace)


def build_filesystem_runtime(root: str, *, namespace: str, persist: "frozenset[StorageDomain] | StorageDomain | None" = None, writer_lock: "FilesystemWriterLock | None" = None) -> FilesystemRuntime:
    if not isinstance(root, str) or not root.strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    validate_persistence_namespace(namespace)
    root_path = Path(root).expanduser().resolve()
    namespace_digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    state_path = root_path / namespace_digest / "manifest.json"
    return _build_runtime(filesystem=True, namespace=namespace, state_path=str(state_path), persist=persist, writer_lock=writer_lock)


def _normalize_runtime_persist(
    persist: "frozenset[StorageDomain] | StorageDomain | None",
    *,
    filesystem: bool,
) -> frozenset[StorageDomain]:
    if persist is None:
        return frozenset({StorageDomain.CONVERSATION}) if filesystem else frozenset()
    values = frozenset({persist}) if isinstance(persist, StorageDomain) else frozenset(persist)
    if not all(isinstance(item, StorageDomain) for item in values):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if StorageDomain.ALL in values:
        if len(values) != 1:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return StorageDomain.durable()
    return values


__all__ = ["FilesystemBlobStore", "FilesystemRuntime", "InMemoryBlobStore", "InMemoryRuntime", "build_filesystem_runtime", "build_in_memory_runtime"]
