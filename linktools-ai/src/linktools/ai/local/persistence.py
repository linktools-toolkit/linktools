#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference runtime persistence for local MEMORY and FILE profiles."""

import asyncio
import base64
import hashlib
import uuid
import json
import os
import re
import tempfile
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from ..capability.tool import ToolOperationRecord, ToolStateStore
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256
from ..core.json import JsonValue
from ..core.paging import Page
from ..core.principal import ResourceRef
from ..core.validation import validate_observation_payload
from ..core.value import (
    ApprovalDecision, ApprovalStatus,
    ExecutionEventType, ExecutionLineageKind, ExecutionProfile, ExecutionStatus, ExternalCallStatus, IdempotencyStatus,
    OperationKind, OperationStatus, ResourceKind, SessionStatus, StopReason,
    EvaluationStatus, TaskStatus, ToolOperationStatus,
)
from ..runtime.persistence import (
    ApprovalRecord, ArtifactRecord, BlobRef, BlobStore, EvaluationRecord, ExecutionEventRecord,
    ExecutionRecord, ExecutionStartClaim, ExecutionStartReservation, ExecutionStartReservationResult, ExecutionStartUnknownCommit, ExecutionTerminalCommit, ExecutionTerminalCommitResult, ExternalResultRecord, IdempotencyTerminalUpdate, OperationTerminalUpdate,
    IdempotencyRecord, MemoryRecord, OperationLedgerInput, OperationLedgerRecord, ResultRecord, RuntimePersistence,
    RuntimeBackend, RuntimePersistenceMode, RuntimeRepository, SessionRecord, TaskLease, TaskNodeView,
)
from ..task.model import TaskGraph, TaskGraphView, TaskNode, TaskTerminalRecord
from ..storage.files import read_json, write_json_atomic
from ..storage.lock import FileLeaseCoordinator, FileWriterLock
from linktools.core import environ

_MAX_BLOB_BYTES = 2 * 1024 * 1024 * 1024
_MAX_INLINE_BLOB_BYTES = 4 * 1024 * 1024
_logger = environ.get_logger("ai.local.persistence")


class _SharedTransactionLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self._depth = 0

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

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if asyncio.current_task() is not self._owner:
            raise RuntimeError("runtime transaction lock owner mismatch")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


class _Base:
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str, backend: RuntimeBackend | None = None) -> None:
        self._mode = mode
        self._backend = backend or (RuntimeBackend.FILE if mode is RuntimePersistenceMode.FILE else RuntimeBackend.MEMORY)
        self._namespace = namespace
        self._atomic_domain_id = atomic_domain_id
        self._local_tenant_id: str | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._on_change: Callable[[], None] | None = None
        self._refresh_source: Callable[[], None] | None = None

    @property
    def mode(self) -> RuntimePersistenceMode:
        return self._mode

    @property
    def backend(self) -> RuntimeBackend:
        return self._backend

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def atomic_domain_id(self) -> str:
        return self._atomic_domain_id

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    def _mark_changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _ensure_open(self) -> None:
        if self._closed:
            raise LinktoolsAIError(ErrorCode.STORAGE_CLOSED)
        if self._refresh_source is not None:
            self._refresh_source()

    def _check_tenant(self, tenant_id: str) -> None:
        if self._local_tenant_id is not None and tenant_id != self._local_tenant_id:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED)


class _SessionRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._records: dict[tuple[str, str], SessionRecord] = {}

    async def create(self, record: SessionRecord) -> SessionRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            key = (record.tenant_id, record.session_id)
            if key in self._records:
                raise LinktoolsAIError(ErrorCode.SESSION_CONFLICT)
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
                raise LinktoolsAIError(ErrorCode.SESSION_REVISION_CONFLICT)
            if next_record.revision != expected_revision + 1 or next_record.tenant_id != tenant_id:
                raise LinktoolsAIError(ErrorCode.SESSION_REVISION_CONFLICT)
            self._records[key] = next_record
            self._mark_changed()
            return next_record

class _ExecutionRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._records: dict[tuple[str, str], ExecutionRecord] = {}
        self._idempotency: _IdempotencyRepository | None = None
        self._events: _EventRepository | None = None

    def bind_start_repositories(self, idempotency: "_IdempotencyRepository", events: "_EventRepository") -> None:
        self._idempotency = idempotency
        self._events = events

    async def create(self, record: ExecutionRecord) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            key = (record.tenant_id, record.execution_id)
            if key in self._records:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
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

    async def compare_and_swap(self, execution_id: str, *, tenant_id: str, expected_snapshot_revision: int, next_record: ExecutionRecord) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None or current.snapshot_revision != expected_snapshot_revision:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.snapshot_revision != expected_snapshot_revision + 1:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
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
            raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async with self._lock:
            key = (claim.tenant_id, claim.execution_id)
            current = self._records.get(key)
            identity = await self._idempotency.get(claim.scope, claim.key_hash, tenant_id=claim.tenant_id)
            if current is None or identity is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
            if current.status is not ExecutionStatus.PENDING_START or current.snapshot_revision != claim.expected_execution_revision or current.event_sequence != claim.expected_event_sequence or current.agent_run_sequence != 0:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            if identity.status is not IdempotencyStatus.RESERVED or identity.execution_id != claim.execution_id or identity.request_digest != claim.request_digest:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            now = claim.started_at
            started = replace(current, status=ExecutionStatus.STARTED, snapshot_revision=current.snapshot_revision + 1, event_sequence=current.event_sequence + 1, updated_at=now, agent_run_sequence=1)
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
            raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if reservation.execution.tenant_id != reservation.idempotency.tenant_id or reservation.execution.execution_id != reservation.idempotency.execution_id:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with self._lock:
            idempotency_key = (reservation.idempotency.tenant_id, reservation.idempotency.scope, reservation.idempotency.key_hash)
            existing_idempotency = self._idempotency._records.get(idempotency_key)
            if existing_idempotency is not None:
                if existing_idempotency.request_digest != reservation.idempotency.request_digest:
                    raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                existing_execution = self._records.get((existing_idempotency.tenant_id, existing_idempotency.execution_id))
                if existing_execution is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return ExecutionStartReservationResult(existing_execution, existing_idempotency, False)
            if (reservation.execution.tenant_id, reservation.execution.execution_id) in self._records:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
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
            if current is None or current.status is not ExecutionStatus.STARTED or current.snapshot_revision != expected_revision or current.agent_run_sequence != expected_agent_run_sequence:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(current, snapshot_revision=current.snapshot_revision + 1, agent_run_sequence=current.agent_run_sequence + 1, updated_at=datetime.now(timezone.utc))
            self._records[key] = updated
            self._mark_changed()
            return updated

    async def mark_start_unknown(self, commit: ExecutionStartUnknownCommit) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(commit.tenant_id)
        if self._idempotency is None or self._events is None:
            raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async with self._lock:
            key = (commit.tenant_id, commit.execution_id)
            current = self._records.get(key)
            identity = await self._idempotency.get(commit.scope, commit.key_hash, tenant_id=commit.tenant_id)
            if current is None or identity is None or current.status is not ExecutionStatus.STARTED or current.snapshot_revision != commit.expected_execution_revision or identity.status is not IdempotencyStatus.STARTED:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            unknown = replace(current, status=ExecutionStatus.START_UNKNOWN, snapshot_revision=current.snapshot_revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.started_at)
            self._records[key] = unknown
            self._idempotency._records[(commit.tenant_id, commit.scope, commit.key_hash)] = replace(identity, status=IdempotencyStatus.START_UNKNOWN, updated_at=commit.started_at)
            self._events._items.setdefault((commit.tenant_id, commit.execution_id), []).append(ExecutionEventRecord(commit.execution_id, commit.tenant_id, current.event_sequence + 1, ExecutionEventType.EXECUTION_START_UNKNOWN, {}))
            self._mark_changed()
            return unknown

    async def advance_sequence(self, execution_id: str, *, tenant_id: str, kind: str, expected_sequence: int) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
            if kind != "event":
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            sequence = current.event_sequence
            if sequence != expected_sequence:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(
                current,
                event_sequence=sequence + 1,
                snapshot_revision=current.snapshot_revision + 1,
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
            if current is None or current.snapshot_revision != expected_revision:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.tenant_id != tenant_id or next_record.execution_id != execution_id or next_record.snapshot_revision != expected_revision + 1:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            self._records[key] = next_record
            self._mark_changed()
            return next_record


class _IdempotencyRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._records: dict[tuple[str, str, str], IdempotencyRecord] = {}

    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        if re.fullmatch(r"[0-9a-f]{64}", record.key_hash) is None:
            raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        async with self._lock:
            key = (record.tenant_id, record.scope, record.key_hash)
            current = self._records.get(key)
            if current is None:
                self._records[key] = record
                self._mark_changed()
                return record
            if current.request_digest != record.request_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
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
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            if next_record.tenant_id != tenant_id or next_record.key_hash != key_hash:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            self._records[store_key] = next_record
            self._mark_changed()
            return next_record


class _EventRepository(_Base):
    def __init__(self, executions: _ExecutionRepository, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
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
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            item = ExecutionEventRecord(execution_id, tenant_id, sequence + 1, event_type, payload)
            await self._executions.advance_sequence(execution_id, tenant_id=tenant_id, kind="event", expected_sequence=expected_sequence)
            items.append(item)
            self._mark_changed()
            return item

    async def list(self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int) -> Page[ExecutionEventRecord]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if not 1 <= limit <= 200:
            raise LinktoolsAIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(item for item in self._items.get((tenant_id, execution_id), ()) if item.sequence > after_sequence)
        return Page(values[:limit], str(values[limit - 1].sequence) if len(values) > limit else None)


class _ResultRepository(_Base):
    def __init__(self, executions: _ExecutionRepository, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._executions = executions
        self._results: dict[tuple[str, str], ResultRecord] = {}
        self._sessions: _SessionRepository | None = None
        self._idempotency: _IdempotencyRepository | None = None
        self._events: _EventRepository | None = None
        self._operations: "_OperationRepository | None" = None

    def bind_terminal_repositories(self, sessions: _SessionRepository, idempotency: _IdempotencyRepository, events: _EventRepository, operations: "_OperationRepository") -> None:
        self._sessions = sessions
        self._idempotency = idempotency
        self._events = events
        self._operations = operations

    async def commit_terminal(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult:
        self._ensure_open()
        execution = commit.terminal_execution
        self._check_tenant(execution.tenant_id)
        result = commit.result
        if (
            execution.execution_id != result.execution_id
            or execution.tenant_id != result.tenant_id
            or execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
            or result.status is not execution.status
        ):
            raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        if commit.expected_event_sequence < 0 or execution.event_sequence != commit.expected_event_sequence + 1:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        if commit.terminal_event_type not in {ExecutionEventType.EXECUTION_SUCCEEDED, ExecutionEventType.EXECUTION_FAILED, ExecutionEventType.EXECUTION_CANCELLED}:
            raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        if not isinstance(commit.terminal_event_payload, dict):
            raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        validate_observation_payload(commit.terminal_event_payload)
        if commit.session_head is not None and (execution.status is not ExecutionStatus.SUCCEEDED or execution.lineage_kind not in {ExecutionLineageKind.SESSION_RESUME, ExecutionLineageKind.RETRY}):
            raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        if self._sessions is None or self._idempotency is None or self._events is None or self._operations is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = (execution.tenant_id, execution.execution_id)
        async with self._lock:
            current_result = self._results.get(key)
            if current_result is not None:
                current = self._executions._records.get(key)
                event = self._events._items.get(key, [])
                identity = self._find_idempotency(commit, execution)
                if (current is not None and current.status is execution.status and current.result_digest == execution.result_digest and current_result == result and len(event) > commit.expected_event_sequence and event[commit.expected_event_sequence].event_type is commit.terminal_event_type and event[commit.expected_event_sequence].payload == commit.terminal_event_payload and self._idempotency_matches(identity, commit.idempotency)):
                    return ExecutionTerminalCommitResult(current, current_result)
                raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            current = self._executions._records.get(key)
            if current is None or current.snapshot_revision != commit.expected_execution_revision or current.event_sequence != commit.expected_event_sequence or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            identity = self._find_idempotency(commit, execution)
            self._validate_idempotency(identity, commit.idempotency, execution)
            self._validate_operation(commit.operation, execution)
            self._executions._records[key] = execution
            self._results[key] = result
            self._events._items.setdefault(key, []).append(ExecutionEventRecord(execution.execution_id, execution.tenant_id, commit.expected_event_sequence + 1, commit.terminal_event_type, commit.terminal_event_payload))
            if commit.idempotency is not None:
                identity_key = (execution.tenant_id, commit.idempotency.scope, commit.idempotency.key_hash)
                self._idempotency._records[identity_key] = replace(identity, status=commit.idempotency.next_status, result_digest=commit.idempotency.result_digest, error_code=commit.idempotency.error_code, updated_at=execution.updated_at)
            if commit.operation is not None:
                current_operation = self._operations._records[(execution.tenant_id, commit.operation.operation_id)]
                self._operations._records[(execution.tenant_id, commit.operation.operation_id)] = replace(current_operation, status=commit.operation.next_status, result_ref=commit.operation.result_ref, result_digest=commit.operation.result_digest, error_code=commit.operation.error_code, updated_at=execution.updated_at)
            if commit.session_head is not None:
                session = self._sessions._records.get((execution.tenant_id, commit.session_head.session_id))
                if session is not None and session.status is SessionStatus.OPEN and session.head_execution_id == commit.session_head.expected_head_execution_id:
                    self._sessions._records[(execution.tenant_id, session.session_id)] = replace(session, revision=session.revision + 1, head_execution_id=commit.session_head.next_head_execution_id, updated_at=execution.updated_at)
            self._mark_changed()
            return ExecutionTerminalCommitResult(execution, result)

    def _find_idempotency(self, commit: ExecutionTerminalCommit, execution: ExecutionRecord) -> IdempotencyRecord | None:
        records = tuple(record for record in self._idempotency._records.values() if record.tenant_id == execution.tenant_id and record.execution_id == execution.execution_id)
        if len(records) > 1:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return records[0] if records else None

    def _validate_idempotency(self, current: IdempotencyRecord | None, update: IdempotencyTerminalUpdate | None, execution: ExecutionRecord) -> None:
        if update is None:
            if current is not None and current.status in {IdempotencyStatus.RESERVED, IdempotencyStatus.STARTED}:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if current is None or current.tenant_id != execution.tenant_id or current.execution_id != execution.execution_id or current.scope != update.scope or current.key_hash != update.key_hash or current.request_digest != update.request_digest or current.status is not update.expected_status:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)

    def _idempotency_matches(self, current: IdempotencyRecord | None, update: IdempotencyTerminalUpdate | None) -> bool:
        return update is None or (current is not None and current.status is update.next_status and current.result_digest == update.result_digest and current.error_code == update.error_code)

    def _validate_operation(self, update: OperationTerminalUpdate | None, execution: ExecutionRecord) -> None:
        if update is None:
            return
        current = self._operations._records.get((execution.tenant_id, update.operation_id))
        if current is None or current.execution_id != execution.execution_id or current.status is not update.expected_status:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)

    async def get(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._results.get((tenant_id, execution_id))


class _MemoryRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._records: dict[tuple[str, str], MemoryRecord] = {}

    async def get_header(self, memory_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        record = self._records.get((tenant_id, memory_id))
        return None if record is None else ResourceRef(ResourceKind.MEMORY, memory_id, tenant_id, record.owner_id)

    async def put(self, record: MemoryRecord, *, expected_revision: int | None) -> MemoryRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            key = (record.tenant_id, record.memory_id)
            current = self._records.get(key)
            if current is None and expected_revision not in (None, 0):
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            if current is not None and current.revision != expected_revision:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            next_record = replace(record, revision=0 if current is None else current.revision + 1)
            self._records[key] = next_record
            self._mark_changed()
            return next_record

    async def get(self, memory_id: str, *, tenant_id: str) -> MemoryRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, memory_id))

    async def list(self, *, tenant_id: str, owner_id: str, cursor: str | None, limit: int) -> Page[MemoryRecord]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if not 1 <= limit <= 200:
            raise LinktoolsAIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(sorted((item for item in self._records.values() if item.tenant_id == tenant_id and item.owner_id == owner_id), key=lambda item: item.memory_id))
        start = 0 if cursor is None else next((index + 1 for index, item in enumerate(values) if item.memory_id == cursor), len(values))
        page = values[start:start + limit]
        return Page(page, page[-1].memory_id if len(values) > start + limit else None)


class _ArtifactRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._records: dict[tuple[str, str], ArtifactRecord] = {}

    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        current = self._records.get((record.tenant_id, record.artifact_id))
        if current is not None and current != record:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
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
            raise LinktoolsAIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(sorted((item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id), key=lambda item: item.artifact_id))
        start = 0 if cursor is None else next((index + 1 for index, item in enumerate(values) if item.artifact_id == cursor), len(values))
        page = values[start:start + limit]
        return Page(page, page[-1].artifact_id if len(values) > start + limit else None)


class _ApprovalRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
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
            raise LinktoolsAIError(ErrorCode.APPROVAL_CONFLICT)
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
                raise LinktoolsAIError(ErrorCode.APPROVAL_CONFLICT)
            updated = replace(current, status=ApprovalStatus.APPROVED if decision is ApprovalDecision.APPROVE else ApprovalStatus.DENIED, decision_id=decision_id, decision=decision, decided_by=principal_id, decision_digest=decision_digest, decided_at=decided_at)
            self._records[(tenant_id, approval_id)] = updated
            self._mark_changed()
            return updated

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id and item.status is ApprovalStatus.PENDING)


class _ExternalRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
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
            raise LinktoolsAIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
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
            raise LinktoolsAIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        updated = replace(current, status=ExternalCallStatus.SUPPLIED, result_id=result_id, payload_ref=payload_ref, payload_digest=payload_digest, supplied_at=supplied_at)
        self._records[(tenant_id, call_id)] = updated
        self._mark_changed()
        return updated

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ExternalResultRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id and item.status is ExternalCallStatus.PENDING)


class _OperationRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._records: dict[tuple[str, str], OperationLedgerRecord] = {}

    async def append(self, record: OperationLedgerInput) -> OperationLedgerRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            key = (record.tenant_id, record.operation_id)
            current = self._records.get(key)
            if current is not None:
                if _operation_immutable(current) != _operation_input_immutable(record):
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
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
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            self._records[(tenant_id, operation_id)] = next_record
            self._mark_changed()
            return next_record

    async def list_pending(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, limit: int) -> tuple[OperationLedgerRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if limit < 1 or limit > 1000:
            raise LinktoolsAIError(ErrorCode.PAGE_LIMIT_INVALID)
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
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
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
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
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

    async def cancel_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = self._plans.get((tenant_id, graph_id))
        if current is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
            return current
        updated = replace(current, status=TaskStatus.CANCELLED)
        self._plans[(tenant_id, graph_id)] = updated
        for key, node in tuple(self._nodes.items()):
            if key[:2] != (tenant_id, graph_id):
                continue
            if node.status in {TaskStatus.PENDING, TaskStatus.READY}:
                self._nodes[key] = replace(node, status=TaskStatus.CANCELLED, lease_expires_at=None)
            elif node.status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED, TaskStatus.SUCCEEDED}:
                continue
            else:
                self._nodes[key] = replace(node, status=TaskStatus.CANCELLED, lease_expires_at=None)
        self._mark_changed()
        return updated

    async def claim(self, graph_id: str, task_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if not owner.strip() or not 1 <= lease_seconds <= 3600:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        key = (tenant_id, graph_id, task_id)
        async with self._lock:
            current = self._nodes.get(key)
            if current is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
            now = datetime.now(timezone.utc)
            reclaimable = current.status is TaskStatus.RUNNING and current.lease_expires_at is not None and current.lease_expires_at <= now
            if current.status not in {TaskStatus.PENDING, TaskStatus.READY} and not reclaimable:
                raise LinktoolsAIError(ErrorCode.TASK_NOT_READY)
            dependencies = tuple(self._nodes.get((tenant_id, graph_id, dependency)) for dependency in current.dependencies)
            if any(dependency is None for dependency in dependencies):
                raise LinktoolsAIError(ErrorCode.TASK_DEPENDENCY_UNKNOWN)
            if any(dependency.status is not TaskStatus.SUCCEEDED for dependency in dependencies if dependency is not None):
                raise LinktoolsAIError(ErrorCode.TASK_NOT_READY)
            fence = current.fence + 1
            expiry = now + timedelta(seconds=lease_seconds)
            self._nodes[key] = replace(current, status=TaskStatus.RUNNING, owner=owner, fence=fence, lease_expires_at=expiry)
            self._mark_changed()
            return TaskLease(graph_id, task_id, tenant_id, owner, fence, expiry)

    async def renew(self, lease: TaskLease, *, tenant_id: str) -> TaskLease:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = self._nodes.get((tenant_id, lease.graph_id, lease.task_id))
        if current is None or current.owner != lease.owner or current.fence != lease.fence or current.lease_expires_at is None or current.lease_expires_at <= datetime.now(timezone.utc):
            raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        expiry = datetime.now(timezone.utc) + timedelta(seconds=60)
        self._nodes[(tenant_id, lease.graph_id, lease.task_id)] = replace(current, lease_expires_at=expiry)
        self._mark_changed()
        return replace(lease, lease_expires_at=expiry)

    async def complete(self, lease: TaskLease, *, tenant_id: str, result_digest: str) -> TaskTerminalRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = self._nodes.get((tenant_id, lease.graph_id, lease.task_id))
        if current is None or current.owner != lease.owner or current.fence != lease.fence:
            raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        if current.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
            if current.status is TaskStatus.SUCCEEDED and current.result_digest == result_digest:
                return TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, current.status, result_digest, None, None)
            raise LinktoolsAIError(ErrorCode.TASK_TERMINAL_CONFLICT)
        self._nodes[(tenant_id, lease.graph_id, lease.task_id)] = replace(current, status=TaskStatus.SUCCEEDED, result_digest=result_digest, lease_expires_at=None)
        self._mark_changed()
        return TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, TaskStatus.SUCCEEDED, result_digest, None, None)

    async def fail(self, lease: TaskLease, *, tenant_id: str, error_code: str, error_digest: str) -> TaskTerminalRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = self._nodes.get((tenant_id, lease.graph_id, lease.task_id))
        if current is None or current.owner != lease.owner or current.fence != lease.fence:
            raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        if current.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
            if current.status is TaskStatus.FAILED and current.error_code == error_code and current.error_digest == error_digest:
                return TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, current.status, None, error_code, error_digest)
            raise LinktoolsAIError(ErrorCode.TASK_TERMINAL_CONFLICT)
        self._nodes[(tenant_id, lease.graph_id, lease.task_id)] = replace(current, status=TaskStatus.FAILED, error_code=error_code, error_digest=error_digest, lease_expires_at=None)
        self._mark_changed()
        return TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, TaskStatus.FAILED, None, error_code, error_digest)

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[TaskNodeView, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(sorted((item for key, item in self._nodes.items() if key[:2] == (tenant_id, graph_id)), key=lambda item: item.task_id))


class _EvaluationRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
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
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
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
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            self._records[(tenant_id, evaluation_id)] = next_record
            self._mark_changed()
            return next_record

    async def list_by_execution(self, execution_id: str, *, tenant_id: str) -> tuple[EvaluationRecord, ...]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return tuple(sorted((item for item in self._records.values() if item.tenant_id == tenant_id and item.execution_id == execution_id), key=lambda item: item.evaluation_id))


class _ToolRepository(_Base, ToolStateStore):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._records: dict[tuple[str, str], ToolOperationRecord] = {}

    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        if re.fullmatch(r"[0-9a-f]{64}", record.idempotency_key_hash) is None:
            raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
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
                raise LinktoolsAIError(ErrorCode.TOOL_OPERATION_CONFLICT)
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
        current = self._records.get((tenant_id, operation_id))
        if current is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
            raise LinktoolsAIError(ErrorCode.TASK_TERMINAL_CONFLICT)
        if current.status is ToolOperationStatus.CLAIMED and current.lease_expires_at is not None and current.lease_expires_at <= datetime.now(timezone.utc) and not current.replay_safe:
            unknown = replace(current, status=ToolOperationStatus.EFFECT_UNKNOWN, lease_expires_at=None)
            self._records[(tenant_id, operation_id)] = unknown
            self._mark_changed()
            raise LinktoolsAIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        if current.owner is not None and current.lease_expires_at is not None and current.lease_expires_at > datetime.now(timezone.utc):
            raise LinktoolsAIError(ErrorCode.TASK_OWNER_CONFLICT)
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

    async def complete(self, operation_id: str, *, tenant_id: str, owner: str, fence: int, result_ref: str | None, result_digest: str) -> ToolOperationRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        current = await self._require_claim(operation_id, tenant_id, owner, fence)
        if current.status is ToolOperationStatus.COMPLETED:
            if current.result_digest == result_digest:
                return current
            raise LinktoolsAIError(ErrorCode.TOOL_RESULT_CONFLICT)
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
        current = self._records.get((tenant_id, operation_id))
        if current is None or current.owner != owner or current.fence != fence or current.lease_expires_at is None or current.lease_expires_at <= datetime.now(timezone.utc):
            raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        return current


class MemoryBlobStore(_Base, BlobStore):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._blobs: dict[tuple[str, str], bytes] = {}

    async def put_bytes(self, *, tenant_id: str, data: bytes, expected_digest: str | None = None) -> BlobRef:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if len(data) > _MAX_INLINE_BLOB_BYTES:
            raise ValueError("put_stream is required for blobs larger than 4 MiB")
        digest = hashlib.sha256(data).hexdigest()
        if expected_digest is not None and expected_digest != digest:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._blobs[(tenant_id, digest)] = data
        self._mark_changed()
        return BlobRef(tenant_id, digest, len(data), f"memory:{self.namespace}:{tenant_id}:{digest}")

    async def put_stream(self, *, tenant_id: str, chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> BlobRef:
        self._ensure_open()
        if expected_size < 0 or expected_size > _MAX_BLOB_BYTES:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        self._check_tenant(tenant_id)
        data = bytearray()
        async for chunk in chunks:
            if not 1 <= len(chunk) <= 1024 * 1024 or len(data) + len(chunk) > _MAX_BLOB_BYTES:
                raise ValueError("invalid blob chunk")
            data.extend(chunk)
        if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_digest:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for offset in range(0, len(data), 64 * 1024):
            yield data[offset:offset + 64 * 1024]

    def open(self, ref: BlobRef, *, tenant_id: str) -> AsyncIterator[bytes]:
        return self._open(ref, tenant_id)


class FileBlobStore(_Base, BlobStore):
    """Tenant-scoped streaming Blob store for FILE runtimes."""

    def __init__(self, root: Path, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._root = root
        self._leases = FileLeaseCoordinator(root / "leases")

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
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._publish(tenant_id, digest, data)

    async def put_stream(self, *, tenant_id: str, chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> BlobRef:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if expected_size < 0 or expected_size > _MAX_BLOB_BYTES:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
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
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return None
        if metadata.get("tenant_id") != tenant_id or metadata.get("digest") != ref.digest or metadata.get("status") != "COMPLETED" or not body.is_file():
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            if body.stat().st_size != int(metadata["size"]):
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except (OSError, TypeError, ValueError) as error:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return BlobRef(tenant_id, ref.digest, int(metadata["size"]), ref.locator or f"file:{self.namespace}:{tenant_id}:{ref.digest}")

    def open(self, ref: BlobRef, *, tenant_id: str) -> AsyncIterator[bytes]:
        return self._open(ref, tenant_id)

    async def _open(self, ref: BlobRef, tenant_id: str) -> AsyncIterator[bytes]:
        blob = await self.stat(ref, tenant_id=tenant_id)
        if blob is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

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
            raise LinktoolsAIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        self._write_metadata(tenant_id, digest, int(payload["size"]))
        committed = dict(payload)
        committed["status"] = "COMMITTED"
        committed["fence"] = fence
        write_json_atomic(journal, committed, fsync=True)
        journal.unlink(missing_ok=True)
        _fsync_directory(journal.parent)

    def _recover_uploads(self) -> None:
        for journal in sorted(self._transaction_directory.glob("*.json")):
            try:
                payload = read_json(journal)
                if payload.get("namespace") != self.namespace or payload.get("record_type") != "blob-upload":
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                temporary = Path(str(payload["temporary"]))
                target = Path(str(payload["body"]))
                metadata = Path(str(payload["metadata"]))
                tenant_id = str(payload["tenant_id"])
                digest = str(payload["digest"])
                if target.is_file() and metadata.is_file():
                    journal.unlink(missing_ok=True)
                elif target.is_file() and not metadata.is_file() and not temporary.is_file():
                    raise LinktoolsAIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                elif metadata.is_file() and not target.is_file() and not temporary.is_file():
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif temporary.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, target)
                    self._write_metadata(tenant_id, digest, int(payload["size"]))
                    journal.unlink(missing_ok=True)
                else:
                    target.unlink(missing_ok=True)
                    metadata.unlink(missing_ok=True)
                    journal.unlink(missing_ok=True)
            except LinktoolsAIError:
                raise
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise LinktoolsAIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error

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
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            payload = value.get("payload")
            if not isinstance(payload, dict) or value.get("payload_sha256") != canonical_sha256(payload):
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return payload
        except (OSError, TypeError, ValueError):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

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


class MemoryRuntime:
    def __init__(self, persistence: RuntimePersistence, components: tuple[RuntimeRepository, ...]) -> None:
        self.persistence = persistence
        self.components = components
        self._initialized = False
        self._closed = False
        self._initialize_lock = asyncio.Lock()

    persistence: RuntimePersistence
    components: tuple[RuntimeRepository, ...]

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
                _logger.info("runtime persistence initialized mode=%s backend=%s namespace=%s", self.persistence.mode, self.persistence.backend, self.persistence.namespace)
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


class _DurableRuntime(MemoryRuntime):
    def __init__(self, persistence: RuntimePersistence, components: tuple[RuntimeRepository, ...], state_path: str, writer_lock: FileWriterLock | None = None) -> None:
        super().__init__(persistence, components)
        self._state_path = state_path
        self._writer_lock = writer_lock or FileWriterLock(Path(state_path).with_name("runtime.lock"))
        self._transaction_lock = _SharedTransactionLock()
        self._commit_lock = threading.Lock()
        for component in components:
            if isinstance(component, _Base):
                component._lock = self._transaction_lock
        self._durable_initialize_lock = asyncio.Lock()

    @property
    def writer_lock(self) -> FileWriterLock:
        return self._writer_lock

    async def initialize(self) -> None:
        async with self._durable_initialize_lock:
            if self._initialized:
                return
            Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
            await self._writer_lock.acquire()
            try:
                await asyncio.to_thread(self._load)
                for component in self.components:
                    if isinstance(component, _Base):
                        component._on_change = self._commit
                await super().initialize()
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
        _logger.info("runtime persistence closed mode=%s namespace=%s", self.persistence.mode, self.persistence.namespace)

    def _commit(self) -> None:
        with self._commit_lock:
            if self._closed:
                raise LinktoolsAIError(ErrorCode.STORAGE_CLOSED)
            try:
                self._flush()
            except LinktoolsAIError as error:
                if error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED:
                    self._closed = True
                    raise
                self._load()
                raise
            except BaseException as error:
                self._load()
                raise LinktoolsAIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    def _load_payload(self, value: dict[str, JsonValue]) -> None:
        sessions = self.components[0]
        executions = self.components[1]
        results = self.components[2]
        idempotency = self.components[3]
        events = self.components[4]
        tasks = self.components[5]
        evaluations = self.components[6]
        memories = self.components[7]
        artifacts = self.components[8]
        approvals = self.components[9]
        externals = self.components[10]
        operations = self.components[11]
        tools = self.components[12]
        if not isinstance(sessions, _SessionRepository) or not isinstance(executions, _ExecutionRepository) or not isinstance(idempotency, _IdempotencyRepository):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._clear_payload()
        for raw in value.get("sessions", []):
            sessions._records[(str(raw["tenant_id"]), str(raw["session_id"]))] = _session_from_json(raw)
        for raw in value.get("executions", []):
            executions._records[(str(raw["tenant_id"]), str(raw["execution_id"]))] = _execution_from_json(raw)
        for raw in value.get("idempotency", []):
            record = _idempotency_from_json(raw)
            idempotency._records[(record.tenant_id, record.scope, str(raw["key_hash"]))] = record
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
        if isinstance(operations, _OperationRepository):
            for raw in value.get("operations", []):
                record = _operation_from_json(raw)
                operations._records[(record.tenant_id, record.operation_id)] = record
        if isinstance(tools, _ToolRepository):
            for raw in value.get("tools", []):
                record = _tool_from_json(raw)
                tools._records[(record.tenant_id, record.operation_id)] = record

    def _clear_payload(self) -> None:
        for component in self.components:
            if isinstance(component, _SessionRepository):
                component._records.clear()
            elif isinstance(component, _ExecutionRepository):
                component._records.clear()
            elif isinstance(component, _ResultRepository):
                component._results.clear()
            elif isinstance(component, _IdempotencyRepository):
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
            elif isinstance(component, MemoryBlobStore):
                component._blobs.clear()

    def _flush(self) -> None:
        records = self._repository_payload()
        tasks = [
            {"record_type": "plan", **item}
            for item in records.pop("task_plans", [])
            if isinstance(item, dict)
        ] + [
            {"record_type": "node", **item}
            for item in records.pop("task_nodes", [])
            if isinstance(item, dict)
        ]
        records["tasks"] = tasks
        state: dict[str, JsonValue] = {
            "schema_version": 2,
            "namespace": self.persistence.namespace,
            "atomic_domain_id": self.persistence.atomic_domain_id,
            "records": records,
        }
        write_json_atomic(Path(self._state_path), state, fsync=True)
        _logger.debug("runtime metadata committed: backend=%s namespace=%s", self.persistence.backend, self.persistence.namespace)
    def _repository_payload(self) -> dict[str, JsonValue]:
        sessions = self.components[0]
        executions = self.components[1]
        results = self.components[2]
        idempotency = self.components[3]
        events = self.components[4]
        tasks = self.components[5]
        evaluations = self.components[6]
        memories = self.components[7]
        artifacts = self.components[8]
        approvals = self.components[9]
        externals = self.components[10]
        operations = self.components[11]
        tools = self.components[12]
        if not isinstance(sessions, _SessionRepository) or not isinstance(executions, _ExecutionRepository) or not isinstance(idempotency, _IdempotencyRepository):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return {
            "sessions": [_record_json(item) for item in sorted(sessions._records.values(), key=lambda value: (value.tenant_id, value.session_id))],
            "executions": [_record_json(item) for item in sorted(executions._records.values(), key=lambda value: (value.tenant_id, value.execution_id))],
            "idempotency": [_idempotency_json(record, key_hash) for (tenant_id, scope, key_hash), record in sorted(idempotency._records.items())],
            "results": [_json_record(item) for item in sorted(results._results.values(), key=lambda value: (value.tenant_id, value.execution_id))] if isinstance(results, _ResultRepository) else [],
            "events": [_json_record(item) for items in events._items.values() for item in items] if isinstance(events, _EventRepository) else [],
            "task_plans": [{"tenant_id": tenant_id, "view": _json_record(view)} for (tenant_id, _), view in tasks._plans.items()] if isinstance(tasks, _TaskRepository) else [],
            "task_nodes": [{"tenant_id": tenant_id, "node": _json_record(node)} for (tenant_id, _, _), node in tasks._nodes.items()] if isinstance(tasks, _TaskRepository) else [],
            "evaluations": [_json_record(item) for item in evaluations._records.values()] if isinstance(evaluations, _EvaluationRepository) else [],
            "memories": [_json_record(item) for item in memories._records.values()] if isinstance(memories, _MemoryRepository) else [],
            "artifacts": [_json_record(item) for item in artifacts._records.values()] if isinstance(artifacts, _ArtifactRepository) else [],
            "approvals": [_json_record(item) for item in approvals._records.values()] if isinstance(approvals, _ApprovalRepository) else [],
            "externals": [_json_record(item) for item in externals._records.values()] if isinstance(externals, _ExternalRepository) else [],
            "operations": [_json_record(item) for item in operations._records.values()] if isinstance(operations, _OperationRepository) else [],
            "tools": [_json_record(item) for item in tools._records.values()] if isinstance(tools, _ToolRepository) else [],
        }


class FileRuntime(_DurableRuntime):
    """Durable FILE runtime backed by one versioned metadata state file."""

    def _load(self) -> None:
        state_path = Path(self._state_path)
        if not state_path.is_file():
            legacy = state_path.parent / "runtime-file-manifest.json"
            if legacy.exists() or (state_path.parent / "session-state").exists():
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            payload: dict[str, JsonValue] = {"sessions": [], "executions": [], "results": [], "idempotency": [], "events": [], "tasks": [], "evaluations": [], "memories": [], "artifacts": [], "approvals": [], "externals": [], "operations": [], "tools": []}
            self._load_payload(payload)
            return
        try:
            state = read_json(state_path)
            if state.get("schema_version") != 2 or state.get("namespace") != self.persistence.namespace or state.get("atomic_domain_id") != self.persistence.atomic_domain_id:
                raise ValueError("runtime state identity mismatch")
            records = state.get("records")
            if not isinstance(records, dict):
                raise ValueError("runtime records must be an object")
            required = ("sessions", "executions", "results", "idempotency", "events", "tasks", "evaluations", "memories", "artifacts", "approvals", "externals", "operations", "tools")
            if set(records) != set(required) or any(not isinstance(records[name], list) for name in required):
                raise ValueError("runtime records shape mismatch")
            _validate_state_record_uniqueness(records)
            payload = dict(records)
            task_records = payload.pop("tasks")
            if any(not isinstance(item, dict) or item.get("record_type") not in {"plan", "node"} for item in task_records):
                raise ValueError("runtime task record shape mismatch")
            payload["task_plans"] = [item for item in task_records if isinstance(item, dict) and item.get("record_type") == "plan"]
            payload["task_nodes"] = [item for item in task_records if isinstance(item, dict) and item.get("record_type") == "node"]
            self._load_payload(payload)
            self._validate_payload()
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            await super().initialize()
        except BaseException:
            self._initialized = False
            await MemoryRuntime.close(self)
            raise

    def _validate_payload(self) -> None:
        executions = self.components[1]
        results = self.components[2]
        if isinstance(executions, _ExecutionRepository) and isinstance(results, _ResultRepository):
            for key, result in results._results.items():
                execution = executions._records.get(key)
                if execution is None or execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} or execution.status is not result.status:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for key, execution in executions._records.items():
                if execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} and key not in results._results:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for component in self.components:
            if isinstance(component, _EventRepository):
                for key, records in component._items.items():
                    if [item.sequence for item in records] != list(range(1, len(records) + 1)):
                        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    execution = executions._records.get(key) if isinstance(executions, _ExecutionRepository) else None
                    if execution is None or execution.event_sequence != len(records):
                        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


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
    return SessionRecord(str(value["session_id"]), str(value["tenant_id"]), str(value["owner_principal_id"]), str(value["binding_digest"]), SessionStatus(str(value["status"])), int(value["revision"]), int(value["resource_generation"]), None if value.get("cwd") is None else str(value["cwd"]), value.get("metadata", {}), _time(value["created_at"]), _time(value["updated_at"]), None if value.get("closed_at") is None else _time(value["closed_at"]), ExecutionProfile(str(value.get("profile", ExecutionProfile.LOCAL_CODING.value))), None if value.get("head_execution_id") is None else str(value["head_execution_id"]))


def _execution_from_json(value: dict[str, JsonValue]) -> ExecutionRecord:
    return ExecutionRecord(str(value["execution_id"]), str(value["tenant_id"]), None if value.get("session_id") is None else str(value["session_id"]), ExecutionProfile(str(value["profile"])), str(value["binding_digest"]), None if value.get("parent_execution_id") is None else str(value["parent_execution_id"]), str(value["root_execution_id"]), ExecutionStatus(str(value["status"])), int(value["snapshot_revision"]), int(value["event_sequence"]), None if value.get("result_ref") is None else str(value["result_ref"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), value.get("safe_error_details", {}), _time(value["created_at"]), _time(value["updated_at"]), None if value.get("source_execution_id") is None else str(value["source_execution_id"]), None if value.get("base_execution_id") is None else str(value["base_execution_id"]), ExecutionLineageKind(str(value.get("lineage_kind", "RUN"))), int(value.get("agent_run_sequence", 0)))


def _idempotency_from_json(value: dict[str, JsonValue]) -> IdempotencyRecord:
    key_hash = str(value["key_hash"])
    if re.fullmatch(r"[0-9a-f]{64}", key_hash) is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return IdempotencyRecord(str(value["tenant_id"]), str(value["scope"]), key_hash, str(value["request_digest"]), str(value["execution_id"]), IdempotencyStatus(str(value["status"])), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), _time(value["created_at"]), _time(value["updated_at"]))


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
    return TaskNodeView(str(value["graph_id"]), str(value["task_id"]), tuple(value.get("dependencies", [])), TaskStatus(str(value["status"])), None if value.get("owner") is None else str(value["owner"]), int(value["fence"]), None if value.get("lease_expires_at") is None else _time(str(value["lease_expires_at"])), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), None if value.get("error_digest") is None else str(value["error_digest"]))


def _evaluation_from_json(value: dict[str, JsonValue]) -> EvaluationRecord:
    return EvaluationRecord(str(value["evaluation_id"]), str(value["tenant_id"]), str(value["execution_id"]), str(value["dataset_id"]), int(value["dataset_revision"]), str(value["evaluator_id"]), int(value["evaluator_revision"]), str(value["binding_digest"]), str(value["output_schema_fingerprint"]), None if value.get("artifact_digest") is None else str(value["artifact_digest"]), EvaluationStatus(str(value["status"])), int(value["revision"]), value.get("metrics", {}), _time(str(value["created_at"])), _time(str(value["updated_at"])))


def _memory_from_json(value: dict[str, JsonValue]) -> MemoryRecord:
    return MemoryRecord(str(value["memory_id"]), str(value["tenant_id"]), str(value["owner_id"]), str(value["kind"]), str(value["content_ref"]), str(value["content_digest"]), value.get("metadata", {}), int(value["revision"]), _time(str(value["created_at"])), _time(str(value["updated_at"])))


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_blob_digest(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


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


def _build_runtime(backend: RuntimeBackend, *, namespace: str, state_path: str | None = None, atomic_domain_id: str | None = None, local_tenant_id: str | None = None, writer_lock: FileWriterLock | None = None) -> MemoryRuntime:
    mode = RuntimePersistenceMode.MEMORY if backend is RuntimeBackend.MEMORY else RuntimePersistenceMode.FILE
    domain = atomic_domain_id or f"{backend.value}-domain-{uuid.uuid4().hex}"
    args = (mode, namespace, domain)
    sessions = _SessionRepository(*args)
    executions = _ExecutionRepository(*args)
    components: tuple[RuntimeRepository, ...] = (
        sessions,
        executions,
        _ResultRepository(executions, *args),
        _IdempotencyRepository(*args),
        _EventRepository(executions, *args),
        _TaskRepository(*args),
        _EvaluationRepository(*args),
        _MemoryRepository(*args),
        _ArtifactRepository(*args),
        _ApprovalRepository(*args),
        _ExternalRepository(*args),
        _OperationRepository(*args),
        _ToolRepository(*args),
        FileBlobStore(Path(state_path).parent, *args) if backend is RuntimeBackend.FILE and state_path is not None else MemoryBlobStore(*args),
    )
    executions.bind_start_repositories(components[3], components[4])
    components[2].bind_terminal_repositories(components[0], components[3], components[4], components[11])
    transaction_lock = _SharedTransactionLock()
    for component in components:
        if isinstance(component, _Base):
            component._lock = transaction_lock
    if local_tenant_id is not None:
        for component in components:
            if isinstance(component, _Base):
                component._local_tenant_id = local_tenant_id
    persistence = RuntimePersistence(
        mode=mode,
        backend=backend,
        namespace=namespace,
        sessions=sessions,
        executions=executions,
        results=components[2],
        idempotency=components[3],
        events=components[4],
        tasks=components[5],
        evaluations=components[6],
        memories=components[7],
        artifacts=components[8],
        approvals=components[9],
        externals=components[10],
        operations=components[11],
        tools=components[12],
        blobs=components[13],
        local_tenant_id=local_tenant_id,
    )
    if backend is RuntimeBackend.FILE:
        if state_path is None:
            raise ValueError("FILE runtime requires a state path")
        return FileRuntime(persistence, components, state_path, writer_lock)
    return MemoryRuntime(persistence, components)


def build_memory_runtime(*, namespace: str | None = None) -> MemoryRuntime:
    return _build_runtime(RuntimeBackend.MEMORY, namespace=namespace or f"memory-{uuid.uuid4().hex}")


def build_file_runtime(root: str, *, project_id: str, local_tenant_id: str, writer_lock: FileWriterLock | None = None) -> FileRuntime:
    if not local_tenant_id.strip() or not project_id.strip():
        raise ValueError("FILE runtime identity is incomplete")
    root_path = Path(root).expanduser().resolve()
    namespace_digest = hashlib.sha256(project_id.encode("utf-8")).hexdigest()
    state_path = root_path / ".linktools" / "runtime" / namespace_digest / "state.json"
    domain = hashlib.sha256(f"file{root_path}{project_id}2".encode("utf-8")).hexdigest()
    return _build_runtime(RuntimeBackend.FILE, namespace=project_id, state_path=str(state_path), atomic_domain_id=domain, local_tenant_id=local_tenant_id, writer_lock=writer_lock)


__all__ = ["FileBlobStore", "FileRuntime", "MemoryBlobStore", "MemoryRuntime", "build_file_runtime", "build_memory_runtime"]
