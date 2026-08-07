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
import sqlite3
import tempfile
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
    CaptureState,
    ExecutionEventType, ExecutionProfile, ExecutionStatus, ExternalCallStatus, IdempotencyStatus,
    OperationKind, OperationStatus, ResourceKind, SessionStatus, StopReason,
    EvaluationStatus, TaskStatus, ToolOperationStatus, TraceKind,
)
from ..runtime.persistence import (
    ApprovalRecord, ArtifactRecord, BlobRef, BlobStore, EvaluationRecord, ExecutionEventRecord,
    ExecutionRecord, ExecutionTerminalCommit, ExecutionTerminalCommitResult, ExternalResultRecord,
    IdempotencyRecord, MemoryRecord, OperationLedgerRecord, ResultRecord, RuntimePersistence,
    RuntimePersistenceMode, RuntimeRepository, SessionRecord, SessionTurnRecord, TaskLease,
    TaskNodeView, TraceRecord,
)
from ..task.model import TaskGraph, TaskGraphView, TaskNode, TaskTerminalRecord
from ..storage.files import read_json, write_bytes_atomic, write_json_atomic
from ..storage.lock import FileLeaseCoordinator
from ..storage.names import storage_name
from linktools.core import environ

_MAX_BLOB_BYTES = 2 * 1024 * 1024 * 1024
_MAX_INLINE_BLOB_BYTES = 4 * 1024 * 1024
_logger = environ.get_logger("ai.local.persistence")


class _Base:
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        self._mode = mode
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
        self._turns: dict[tuple[str, str], list[SessionTurnRecord]] = {}

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

    async def append_turn(self, turn: SessionTurnRecord, *, expected_sequence: int) -> SessionTurnRecord:
        self._ensure_open()
        self._check_tenant(turn.tenant_id)
        session = self._records.get((turn.tenant_id, turn.session_id))
        if session is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._lock:
            key = (turn.tenant_id, turn.session_id)
            turns = self._turns.setdefault(key, [])
            sequence = turns[-1].sequence if turns else 0
            if sequence != expected_sequence:
                raise LinktoolsAIError(ErrorCode.SESSION_REVISION_CONFLICT)
            if turn.sequence != expected_sequence + 1:
                raise LinktoolsAIError(ErrorCode.SESSION_REVISION_CONFLICT)
            turns.append(turn)
            self._mark_changed()
            return turn

    async def list_turns(self, session_id: str, *, tenant_id: str, after_sequence: int, limit: int) -> Page[SessionTurnRecord]:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if limit < 1 or limit > 200:
            raise LinktoolsAIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = tuple(item for item in self._turns.get((tenant_id, session_id), ()) if item.sequence > after_sequence)
        return Page(values[:limit], str(values[limit - 1].sequence) if len(values) > limit else None)


class _ExecutionRepository(_Base):
    def __init__(self, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._records: dict[tuple[str, str], ExecutionRecord] = {}

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

    async def advance_sequence(self, execution_id: str, *, tenant_id: str, kind: str, expected_sequence: int) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
            sequence = current.event_sequence if kind == "event" else current.trace_sequence
            if sequence != expected_sequence:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(
                current,
                event_sequence=sequence + 1 if kind == "event" else current.event_sequence,
                trace_sequence=sequence + 1 if kind == "trace" else current.trace_sequence,
                snapshot_revision=current.snapshot_revision + 1,
                updated_at=datetime.now(timezone.utc),
            )
            self._records[key] = updated
            self._mark_changed()
            return updated

    async def mark_trace_persistence_failed(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        async with self._lock:
            key = (tenant_id, execution_id)
            current = self._records.get(key)
            if current is None:
                raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
            updated = replace(current, error_code=ErrorCode.TRACE_PERSISTENCE_FAILED.value, snapshot_revision=current.snapshot_revision + 1, updated_at=datetime.now(timezone.utc))
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
        if not record.key.strip():
            raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        async with self._lock:
            key = (record.tenant_id, record.scope, _key_hash(record.key))
            current = self._records.get(key)
            if current is None:
                self._records[key] = record
                self._mark_changed()
                return record
            if current.request_digest != record.request_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return current

    async def get(self, scope: str, key: str, *, tenant_id: str) -> IdempotencyRecord | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return self._records.get((tenant_id, scope, _key_hash(key)))

    async def compare_and_swap(self, scope: str, key: str, *, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord:
        self._ensure_open()
        self._check_tenant(next_record.tenant_id)
        async with self._lock:
            store_key = (next_record.tenant_id, scope, _key_hash(key))
            current = self._records.get(store_key)
            if current is None or current.status is not expected_status:
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


class _TraceRepository(_Base):
    def __init__(self, executions: _ExecutionRepository, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._executions = executions
        self._items: dict[tuple[str, str], list[TraceRecord]] = {}

    async def append(self, execution_id: str, *, tenant_id: str, expected_sequence: int, kind: TraceKind, payload: JsonValue) -> TraceRecord:
        self._ensure_open()
        self._check_tenant(tenant_id)
        validate_observation_payload(payload)
        async with self._lock:
            items = self._items.setdefault((tenant_id, execution_id), [])
            sequence = items[-1].sequence if items else 0
            if sequence != expected_sequence:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            item = TraceRecord(execution_id, tenant_id, sequence + 1, kind, payload)
            try:
                await self._executions.advance_sequence(execution_id, tenant_id=tenant_id, kind="trace", expected_sequence=expected_sequence)
            except BaseException:
                await self._executions.mark_trace_persistence_failed(execution_id, tenant_id=tenant_id)
                raise
            items.append(item)
            self._mark_changed()
            return item

    async def list(self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int) -> Page[TraceRecord]:
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
        key = (execution.tenant_id, execution.execution_id)
        async with self._lock:
            current_result = self._results.get(key)
            if current_result is not None:
                if current_result.payload_digest == result.payload_digest:
                    current = await self._executions.get(execution.execution_id, tenant_id=execution.tenant_id)
                    if current is None:
                        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    return ExecutionTerminalCommitResult(current, current_result)
                raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            current = await self._executions.get(execution.execution_id, tenant_id=execution.tenant_id)
            if current is None or current.snapshot_revision != commit.expected_execution_revision:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            await self._executions.commit_terminal(execution.execution_id, tenant_id=execution.tenant_id, expected_revision=commit.expected_execution_revision, next_record=execution)
            self._results[key] = result
            self._mark_changed()
            return ExecutionTerminalCommitResult(execution, result)

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

    async def next_sequence(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str) -> int:
        self._ensure_open()
        self._check_tenant(tenant_id)
        return max(
            (item.sequence for item in self._records.values() if item.tenant_id == tenant_id and item.resource_kind is resource_kind and item.resource_id == resource_id),
            default=0,
        ) + 1

    async def create(self, record: OperationLedgerRecord) -> OperationLedgerRecord:
        self._ensure_open()
        self._check_tenant(record.tenant_id)
        async with self._lock:
            key = (record.tenant_id, record.operation_id)
            if key in self._records and self._records[key].request_digest != record.request_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if key not in self._records:
                previous = tuple(item.sequence for item in self._records.values() if item.tenant_id == record.tenant_id and item.resource_kind is record.resource_kind and item.resource_id == record.resource_id)
                expected_sequence = max(previous, default=0) + 1
                if record.sequence != expected_sequence:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                self._records[key] = record
                self._mark_changed()
            return self._records[key]

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


class SqlBlobStore(MemoryBlobStore):
    """SQLite chunk store with an UPLOADING-to-COMPLETED publication fence."""

    def __init__(self, database_path: str, mode: RuntimePersistenceMode, namespace: str, atomic_domain_id: str) -> None:
        super().__init__(mode, namespace, atomic_domain_id)
        self._database_path = database_path

    async def initialize(self) -> None:
        await super().initialize()
        await asyncio.to_thread(self._ensure_blob_schema)

    async def put_bytes(self, *, tenant_id: str, data: bytes, expected_digest: str | None = None) -> BlobRef:
        if len(data) > _MAX_INLINE_BLOB_BYTES:
            raise ValueError("put_stream is required for blobs larger than 4 MiB")

        async def chunks() -> AsyncIterator[bytes]:
            yield data

        digest = hashlib.sha256(data).hexdigest()
        if expected_digest is not None and expected_digest != digest:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self.put_stream(tenant_id=tenant_id, chunks=chunks(), expected_size=len(data), expected_digest=digest)

    async def put_stream(self, *, tenant_id: str, chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> BlobRef:
        self._ensure_open()
        self._check_tenant(tenant_id)
        _validate_blob_digest(expected_digest)
        if expected_size < 0 or expected_size > _MAX_BLOB_BYTES:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        existing = await asyncio.to_thread(self._completed_blob, tenant_id, expected_digest)
        if existing is not None:
            if existing.size != expected_size:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        await asyncio.to_thread(self._begin_blob, tenant_id, expected_digest)
        size = 0
        digest = hashlib.sha256()
        index = 0
        try:
            async for chunk in chunks:
                if not 1 <= len(chunk) <= 1024 * 1024 or size + len(chunk) > _MAX_BLOB_BYTES:
                    raise ValueError("invalid blob chunk")
                chunk_digest = hashlib.sha256(chunk).hexdigest()
                await asyncio.to_thread(self._write_chunk, tenant_id, expected_digest, index, chunk, chunk_digest)
                digest.update(chunk)
                size += len(chunk)
                index += 1
            if size != expected_size or digest.hexdigest() != expected_digest:
                await asyncio.to_thread(self._mark_blob_failed, tenant_id, expected_digest)
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await asyncio.to_thread(self._complete_blob, tenant_id, expected_digest, size, index)
            return BlobRef(tenant_id, expected_digest, size, f"sql:{self.namespace}:{tenant_id}:{expected_digest}")
        except BaseException:
            await asyncio.to_thread(self._mark_blob_failed, tenant_id, expected_digest)
            raise

    async def stat(self, ref: BlobRef, *, tenant_id: str) -> BlobRef | None:
        self._ensure_open()
        self._check_tenant(tenant_id)
        if ref.tenant_id != tenant_id:
            return None
        record = await asyncio.to_thread(self._completed_blob, tenant_id, ref.digest)
        return None if record is None else BlobRef(tenant_id, ref.digest, record.size, ref.locator or f"sql:{self.namespace}:{tenant_id}:{ref.digest}")

    def open(self, ref: BlobRef, *, tenant_id: str) -> AsyncIterator[bytes]:
        return self._open_sql(ref, tenant_id)

    async def _open_sql(self, ref: BlobRef, tenant_id: str) -> AsyncIterator[bytes]:
        blob = await self.stat(ref, tenant_id=tenant_id)
        if blob is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        chunk_count = await asyncio.to_thread(self._chunk_count, tenant_id, blob.digest)
        index = 0
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = await asyncio.to_thread(self._read_chunk, tenant_id, blob.digest, index)
            if chunk is None:
                if index != chunk_count or size != blob.size or digest.hexdigest() != blob.digest:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return
            digest.update(chunk)
            size += len(chunk)
            yield chunk
            index += 1

    def _ensure_blob_schema(self) -> None:
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS ai_runtime_blobs (tenant_id TEXT NOT NULL, digest TEXT NOT NULL, size INTEGER NOT NULL, status TEXT NOT NULL, chunk_count INTEGER NOT NULL, created_at TEXT NOT NULL, completed_at TEXT, PRIMARY KEY(tenant_id, digest))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS ai_runtime_blob_chunks (tenant_id TEXT NOT NULL, digest TEXT NOT NULL, chunk_index INTEGER NOT NULL, content BLOB NOT NULL, content_size INTEGER NOT NULL, content_digest TEXT NOT NULL, PRIMARY KEY(tenant_id, digest, chunk_index))"
            )

    def _completed_blob(self, tenant_id: str, digest: str) -> BlobRef | None:
        with sqlite3.connect(self._database_path) as connection:
            row = connection.execute("SELECT size FROM ai_runtime_blobs WHERE tenant_id = ? AND digest = ? AND status = 'COMPLETED'", (tenant_id, digest)).fetchone()
        return None if row is None else BlobRef(tenant_id, digest, int(row[0]), f"sql:{self.namespace}:{tenant_id}:{digest}")

    def _begin_blob(self, tenant_id: str, digest: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                "INSERT INTO ai_runtime_blobs(tenant_id, digest, size, status, chunk_count, created_at, completed_at) VALUES (?, ?, 0, 'UPLOADING', 0, ?, NULL) ON CONFLICT(tenant_id, digest) DO UPDATE SET status = CASE WHEN ai_runtime_blobs.status = 'FAILED' THEN 'UPLOADING' ELSE ai_runtime_blobs.status END",
                (tenant_id, digest, now),
            )

    def _write_chunk(self, tenant_id: str, digest: str, index: int, content: bytes, content_digest: str) -> None:
        with sqlite3.connect(self._database_path) as connection:
            row = connection.execute("SELECT content_digest FROM ai_runtime_blob_chunks WHERE tenant_id = ? AND digest = ? AND chunk_index = ?", (tenant_id, digest, index)).fetchone()
            if row is not None:
                if str(row[0]) != content_digest:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return
            connection.execute("INSERT INTO ai_runtime_blob_chunks(tenant_id, digest, chunk_index, content, content_size, content_digest) VALUES (?, ?, ?, ?, ?, ?)", (tenant_id, digest, index, content, len(content), content_digest))

    def _complete_blob(self, tenant_id: str, digest: str, size: int, chunk_count: int) -> None:
        with sqlite3.connect(self._database_path) as connection:
            result = connection.execute("UPDATE ai_runtime_blobs SET size = ?, status = 'COMPLETED', chunk_count = ?, completed_at = ? WHERE tenant_id = ? AND digest = ? AND status = 'UPLOADING'", (size, chunk_count, datetime.now(timezone.utc).isoformat(), tenant_id, digest))
            if result.rowcount != 1:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)

    def _mark_blob_failed(self, tenant_id: str, digest: str) -> None:
        with sqlite3.connect(self._database_path) as connection:
            connection.execute("UPDATE ai_runtime_blobs SET status = 'FAILED' WHERE tenant_id = ? AND digest = ?", (tenant_id, digest))

    def _read_chunk(self, tenant_id: str, digest: str, index: int) -> bytes | None:
        with sqlite3.connect(self._database_path) as connection:
            row = connection.execute("SELECT content, content_size, content_digest FROM ai_runtime_blob_chunks WHERE tenant_id = ? AND digest = ? AND chunk_index = ?", (tenant_id, digest, index)).fetchone()
        if row is None:
            return None
        content = bytes(row[0])
        if len(content) != int(row[1]) or hashlib.sha256(content).hexdigest() != str(row[2]):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return content

    def _chunk_count(self, tenant_id: str, digest: str) -> int:
        with sqlite3.connect(self._database_path) as connection:
            row = connection.execute(
                "SELECT chunk_count FROM ai_runtime_blobs WHERE tenant_id = ? AND digest = ? AND status = 'COMPLETED'",
                (tenant_id, digest),
            ).fetchone()
        if row is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return int(row[0])


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

    persistence: RuntimePersistence
    components: tuple[RuntimeRepository, ...]

    async def initialize(self) -> None:
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
            _logger.info("runtime persistence initialized mode=%s namespace=%s", self.persistence.mode, self.persistence.namespace)
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
    def __init__(self, persistence: RuntimePersistence, components: tuple[RuntimeRepository, ...], state_path: str) -> None:
        super().__init__(persistence, components)
        self._state_path = state_path
        self._flush_task: asyncio.Task[None] | None = None
        self._flush_dirty = False
        self._flush_active = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._load)
        for component in self.components:
            if isinstance(component, _Base):
                component._on_change = self._request_flush
                component._refresh_source = self._refresh_source
        await super().initialize()

    async def close(self) -> None:
        if self._closed:
            return
        await self._drain_flush()
        await asyncio.to_thread(self._flush)
        await super().close()
        _logger.info("runtime persistence closed mode=%s namespace=%s", self.persistence.mode, self.persistence.namespace)

    def _request_flush(self) -> None:
        self._flush_dirty = True
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while self._flush_dirty:
            self._flush_dirty = False
            self._flush_active = True
            try:
                await asyncio.to_thread(self._flush)
            finally:
                self._flush_active = False

    async def _drain_flush(self) -> None:
        task = self._flush_task
        if task is not None:
            await task
        if self._flush_dirty:
            await self._flush_loop()

    def _load_payload(self, value: dict[str, JsonValue]) -> None:
        sessions = self.components[0]
        executions = self.components[1]
        results = self.components[2]
        idempotency = self.components[3]
        events = self.components[4]
        traces = self.components[5]
        tasks = self.components[6]
        evaluations = self.components[7]
        memories = self.components[8]
        artifacts = self.components[9]
        approvals = self.components[10]
        externals = self.components[11]
        operations = self.components[12]
        tools = self.components[13]
        blobs = self.components[14]
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
        if isinstance(sessions, _SessionRepository):
            for raw in value.get("turns", []):
                tenant_id = str(raw["tenant_id"])
                turn = _turn_from_json(raw["turn"])
                sessions._turns.setdefault((tenant_id, turn.session_id), []).append(turn)
        if isinstance(results, _ResultRepository):
            for raw in value.get("results", []):
                record = _result_from_json(raw)
                results._results[(record.tenant_id, record.execution_id)] = record
        if isinstance(events, _EventRepository):
            for raw in value.get("events", []):
                record = _event_from_json(raw)
                events._items.setdefault((record.tenant_id, record.execution_id), []).append(record)
        if isinstance(traces, _TraceRepository):
            for raw in value.get("traces", []):
                record = _trace_from_json(raw)
                traces._items.setdefault((record.tenant_id, record.execution_id), []).append(record)
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
        if isinstance(blobs, MemoryBlobStore):
            for raw in value.get("blobs", []):
                tenant_id = str(raw["tenant_id"])
                digest = str(raw["digest"])
                blobs._blobs[(tenant_id, digest)] = base64.b64decode(str(raw["data"]))

    def _clear_payload(self) -> None:
        for component in self.components:
            if isinstance(component, _SessionRepository):
                component._records.clear()
                component._turns.clear()
            elif isinstance(component, _ExecutionRepository):
                component._records.clear()
            elif isinstance(component, _ResultRepository):
                component._results.clear()
            elif isinstance(component, _IdempotencyRepository):
                component._records.clear()
            elif isinstance(component, _EventRepository):
                component._items.clear()
            elif isinstance(component, _TraceRepository):
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

    def _refresh_source(self) -> None:
        if self._flush_dirty or self._flush_active or self._closed:
            return
        self._load()

    def _flush(self) -> None:
        sessions = self.components[0]
        executions = self.components[1]
        results = self.components[2]
        idempotency = self.components[3]
        events = self.components[4]
        traces = self.components[5]
        tasks = self.components[6]
        evaluations = self.components[7]
        memories = self.components[8]
        artifacts = self.components[9]
        approvals = self.components[10]
        externals = self.components[11]
        operations = self.components[12]
        tools = self.components[13]
        blobs = self.components[14]
        if not isinstance(sessions, _SessionRepository) or not isinstance(executions, _ExecutionRepository) or not isinstance(idempotency, _IdempotencyRepository):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        root = Path(self._state_path).parent
        for record in sessions._records.values():
            self._write_record(root / "session-state" / _component(record.tenant_id) / f"{_component(record.session_id)}.json", "session", self.persistence.namespace, record.revision, _record_json(record))
        for record in executions._records.values():
            self._write_record(root / "executions" / _component(record.tenant_id) / f"{_component(record.execution_id)}.json", "execution", self.persistence.namespace, record.snapshot_revision, _record_json(record))
        for (tenant_id, scope, key_hash), record in idempotency._records.items():
            scope_hash = hashlib.sha256(scope.encode("utf-8")).hexdigest()
            self._write_record(root / "idempotency" / _component(tenant_id) / scope_hash / f"{key_hash}.json", "idempotency", self.persistence.namespace, 0, _idempotency_json(record, key_hash))
        for (tenant_id, _), turns in sessions._turns.items():
            for turn in turns:
                self._write_record(root / "session-turns" / _component(tenant_id) / _component(turn.session_id) / f"{turn.sequence}.json", "session-turn", self.persistence.namespace, turn.sequence, _json_record(turn))
        if isinstance(results, _ResultRepository):
            for record in results._results.values():
                self._write_record(root / "results" / _component(record.tenant_id) / f"{_component(record.execution_id)}.json", "result", self.persistence.namespace, 0, _json_record(record))
        if isinstance(events, _EventRepository):
            for (tenant_id, execution_id), records in events._items.items():
                self._write_sequence(root / "events" / _component(tenant_id) / f"{_component(execution_id)}.jsonl", self.persistence.namespace, records)
        if isinstance(traces, _TraceRepository):
            for (tenant_id, execution_id), records in traces._items.items():
                self._write_sequence(root / "traces" / _component(tenant_id) / f"{_component(execution_id)}.jsonl", self.persistence.namespace, records)
        if isinstance(tasks, _TaskRepository):
            for (tenant_id, graph_id), view in tasks._plans.items():
                self._write_record(root / "tasks" / "plans" / _component(tenant_id) / f"{_component(graph_id)}.json", "task-plan", self.persistence.namespace, 0, _json_record(view))
            for (tenant_id, graph_id, task_id), node in tasks._nodes.items():
                self._write_record(root / "tasks" / "nodes" / _component(tenant_id) / _component(graph_id) / f"{_component(task_id)}.json", "task-node", self.persistence.namespace, node.fence, _json_record(node))
        if isinstance(evaluations, _EvaluationRepository):
            for record in evaluations._records.values():
                self._write_record(root / "evaluations" / _component(record.tenant_id) / f"{_component(record.evaluation_id)}.json", "evaluation", self.persistence.namespace, record.revision, _json_record(record))
        if isinstance(memories, _MemoryRepository):
            for record in memories._records.values():
                self._write_record(root / "memories" / _component(record.tenant_id) / f"{_component(record.memory_id)}.json", "memory", self.persistence.namespace, record.revision, _json_record(record))
        if isinstance(artifacts, _ArtifactRepository):
            for record in artifacts._records.values():
                self._write_record(root / "artifacts" / _component(record.tenant_id) / f"{_component(record.artifact_id)}.json", "artifact", self.persistence.namespace, 0, _json_record(record))
        if isinstance(approvals, _ApprovalRepository):
            for record in approvals._records.values():
                self._write_record(root / "approvals" / _component(record.tenant_id) / f"{_component(record.approval_id)}.json", "approval", self.persistence.namespace, 0, _json_record(record))
        if isinstance(externals, _ExternalRepository):
            for record in externals._records.values():
                self._write_record(root / "externals" / _component(record.tenant_id) / f"{_component(record.call_id)}.json", "external", self.persistence.namespace, 0, _json_record(record))
        if isinstance(operations, _OperationRepository):
            operation_root = root / "operations"
            expected_operation_paths = {
                operation_root / _component(record.tenant_id) / record.resource_kind.value / _component(record.resource_id) / f"{record.sequence}.json"
                for record in operations._records.values()
            }
            if operation_root.exists():
                for path in operation_root.rglob("*.json"):
                    if path not in expected_operation_paths:
                        path.unlink(missing_ok=True)
            for record in operations._records.values():
                self._write_record(root / "operations" / _component(record.tenant_id) / record.resource_kind.value / _component(record.resource_id) / f"{record.sequence}.json", "operation", self.persistence.namespace, record.sequence, _json_record(record))
        if isinstance(tools, _ToolRepository):
            for record in tools._records.values():
                self._write_record(root / "tools" / _component(record.tenant_id) / f"{_component(record.operation_id)}.json", "tool-operation", self.persistence.namespace, record.fence, _json_record(record))
        write_json_atomic(
            root / "runtime-file-manifest.json",
            {
                "schema_version": 1,
                "root_digest": self.persistence.atomic_domain_id,
                "namespace": self.persistence.namespace,
                "object_counts": {
                    "sessions": len(sessions._records),
                    "executions": len(executions._records),
                    "idempotency": len(idempotency._records),
                    "results": len(results._results) if isinstance(results, _ResultRepository) else 0,
                    "events": sum(len(items) for items in events._items.values()) if isinstance(events, _EventRepository) else 0,
                    "traces": sum(len(items) for items in traces._items.values()) if isinstance(traces, _TraceRepository) else 0,
                    "task_plans": len(tasks._plans) if isinstance(tasks, _TaskRepository) else 0,
                    "task_nodes": len(tasks._nodes) if isinstance(tasks, _TaskRepository) else 0,
                    "evaluations": len(evaluations._records) if isinstance(evaluations, _EvaluationRepository) else 0,
                    "memories": len(memories._records) if isinstance(memories, _MemoryRepository) else 0,
                    "artifacts": len(artifacts._records) if isinstance(artifacts, _ArtifactRepository) else 0,
                    "approvals": len(approvals._records) if isinstance(approvals, _ApprovalRepository) else 0,
                    "externals": len(externals._records) if isinstance(externals, _ExternalRepository) else 0,
                    "operations": len(operations._records) if isinstance(operations, _OperationRepository) else 0,
                    "tools": len(tools._records) if isinstance(tools, _ToolRepository) else 0,
                    "blobs": len(blobs._blobs) if isinstance(blobs, MemoryBlobStore) else 0,
                },
                "corruptions": [],
            },
            fsync=True,
        )

    @staticmethod
    def _write_record(path: Path, record_type: str, namespace: str, revision: int, payload: dict[str, JsonValue]) -> None:
        write_json_atomic(
            path,
            {
                "schema_version": 1,
                "record_type": record_type,
                "namespace": namespace,
                "record_revision": revision,
                "payload": payload,
                "payload_sha256": canonical_sha256(payload),
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            fsync=True,
        )

    @staticmethod
    def _write_sequence(path: Path, namespace: str, records: list[ExecutionEventRecord | TraceRecord]) -> None:
        previous: str | None = None
        lines: list[str] = []
        for record in records:
            payload = _json_record(record)
            line = {
                "schema_version": 1,
                "namespace": namespace,
                "sequence": record.sequence,
                "payload": payload,
                "payload_sha256": canonical_sha256(payload),
                "previous_line_sha256": previous,
            }
            line_digest = canonical_sha256(line)
            line["line_sha256"] = line_digest
            previous = line_digest
            lines.append(json.dumps(line, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        write_bytes_atomic(path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"), fsync=True)

    def _repository_payload(self) -> dict[str, JsonValue]:
        sessions = self.components[0]
        executions = self.components[1]
        results = self.components[2]
        idempotency = self.components[3]
        events = self.components[4]
        traces = self.components[5]
        tasks = self.components[6]
        evaluations = self.components[7]
        memories = self.components[8]
        artifacts = self.components[9]
        approvals = self.components[10]
        externals = self.components[11]
        operations = self.components[12]
        tools = self.components[13]
        blobs = self.components[14]
        if not isinstance(sessions, _SessionRepository) or not isinstance(executions, _ExecutionRepository) or not isinstance(idempotency, _IdempotencyRepository):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return {
            "sessions": [_record_json(item) for item in sorted(sessions._records.values(), key=lambda value: (value.tenant_id, value.session_id))],
            "executions": [_record_json(item) for item in sorted(executions._records.values(), key=lambda value: (value.tenant_id, value.execution_id))],
            "idempotency": [_idempotency_json(record, key_hash) for (tenant_id, scope, key_hash), record in sorted(idempotency._records.items())],
            "turns": [{"tenant_id": tenant_id, "turn": _json_record(turn)} for (tenant_id, _), turns in sessions._turns.items() for turn in turns] if isinstance(sessions, _SessionRepository) else [],
            "results": [_json_record(item) for item in sorted(results._results.values(), key=lambda value: (value.tenant_id, value.execution_id))] if isinstance(results, _ResultRepository) else [],
            "events": [_json_record(item) for items in events._items.values() for item in items] if isinstance(events, _EventRepository) else [],
            "traces": [_json_record(item) for items in traces._items.values() for item in items] if isinstance(traces, _TraceRepository) else [],
            "task_plans": [{"tenant_id": tenant_id, "view": _json_record(view)} for (tenant_id, _), view in tasks._plans.items()] if isinstance(tasks, _TaskRepository) else [],
            "task_nodes": [{"tenant_id": tenant_id, "node": _json_record(node)} for (tenant_id, _, _), node in tasks._nodes.items()] if isinstance(tasks, _TaskRepository) else [],
            "evaluations": [_json_record(item) for item in evaluations._records.values()] if isinstance(evaluations, _EvaluationRepository) else [],
            "memories": [_json_record(item) for item in memories._records.values()] if isinstance(memories, _MemoryRepository) else [],
            "artifacts": [_json_record(item) for item in artifacts._records.values()] if isinstance(artifacts, _ArtifactRepository) else [],
            "approvals": [_json_record(item) for item in approvals._records.values()] if isinstance(approvals, _ApprovalRepository) else [],
            "externals": [_json_record(item) for item in externals._records.values()] if isinstance(externals, _ExternalRepository) else [],
            "operations": [_json_record(item) for item in operations._records.values()] if isinstance(operations, _OperationRepository) else [],
            "tools": [_json_record(item) for item in tools._records.values()] if isinstance(tools, _ToolRepository) else [],
            "blobs": [
                {"tenant_id": tenant_id, "digest": digest, "data": base64.b64encode(data).decode("ascii")}
                for (tenant_id, digest), data in sorted(blobs._blobs.items())
            ] if isinstance(blobs, MemoryBlobStore) else [],
        }


class FileRuntime(_DurableRuntime):
    """Durable FILE runtime with atomic envelopes and sequence journals."""

    def _load(self) -> None:
        root = Path(self._state_path).parent
        manifest_path = root / "runtime-file-manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if (
                manifest.get("schema_version") != 1
                or manifest.get("namespace") != self.persistence.namespace
                or manifest.get("root_digest") != self.persistence.atomic_domain_id
            ):
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload: dict[str, JsonValue] = {
            "sessions": [], "turns": [], "executions": [], "idempotency": [], "results": [],
            "events": [], "traces": [], "task_plans": [], "task_nodes": [], "evaluations": [],
            "memories": [], "artifacts": [], "approvals": [], "externals": [], "operations": [], "tools": [],
        }
        for path in sorted((root / "session-state").glob("*/*.json")):
            payload["sessions"].append(_read_file_envelope(path, self.persistence.namespace))
        for path in sorted((root / "session-turns").glob("*/*/*.json")):
            payload["turns"].append({"tenant_id": _decode_component(path.parents[1].name), "turn": _read_file_envelope(path, self.persistence.namespace)})
        for path in sorted((root / "executions").glob("*/*.json")):
            payload["executions"].append(_read_file_envelope(path, self.persistence.namespace))
        for path in sorted((root / "idempotency").glob("*/*/*.json")):
            payload["idempotency"].append(_read_file_envelope(path, self.persistence.namespace))
        for name, directory in (("results", "results"), ("evaluations", "evaluations"), ("memories", "memories"), ("artifacts", "artifacts"), ("approvals", "approvals"), ("externals", "externals")):
            for path in sorted((root / directory).glob("*/*.json")):
                payload[name].append(_read_file_envelope(path, self.persistence.namespace))
        for path in sorted((root / "events").glob("*/*.jsonl")):
            payload["events"].extend(_read_sequence_file(path, self.persistence.namespace))
        for path in sorted((root / "traces").glob("*/*.jsonl")):
            payload["traces"].extend(_read_sequence_file(path, self.persistence.namespace))
        for path in sorted((root / "tasks" / "plans").glob("*/*.json")):
            payload["task_plans"].append({"tenant_id": _decode_component(path.parent.name), "view": _read_file_envelope(path, self.persistence.namespace)})
        for path in sorted((root / "tasks" / "nodes").glob("*/*/*.json")):
            payload["task_nodes"].append({"tenant_id": _decode_component(path.parents[1].name), "node": _read_file_envelope(path, self.persistence.namespace)})
        for path in sorted((root / "operations").glob("*/*/*/*.json")):
            payload["operations"].append(_read_file_envelope(path, self.persistence.namespace))
        for path in sorted((root / "tools").glob("*/*.json")):
            payload["tools"].append(_read_file_envelope(path, self.persistence.namespace))
        self._load_payload(payload)

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            await asyncio.to_thread(self._validate_sequences)
            await super().initialize()
        except BaseException:
            self._initialized = False
            await MemoryRuntime.close(self)
            raise

    def _validate_sequences(self) -> None:
        root = Path(self._state_path).parent
        for directory in (root / "events", root / "traces"):
            if not directory.exists():
                continue
            for path in directory.rglob("*.jsonl"):
                raw = path.read_bytes()
                if raw and not raw.endswith(b"\n"):
                    raw = raw[: raw.rfind(b"\n") + 1]
                    write_bytes_atomic(path, raw, fsync=True)
                _read_sequence_file(path, self.persistence.namespace)


class SqlRuntime(_DurableRuntime):
    """SQLite runtime storing one durable row per repository record."""

    async def initialize(self) -> None:
        if self._initialized:
            return
        await asyncio.to_thread(self._load)
        for component in self.components:
            if isinstance(component, _Base):
                component._on_change = self._request_flush
                component._refresh_source = self._refresh_source
        await MemoryRuntime.initialize(self)

    async def close(self) -> None:
        if self._closed:
            return
        await self._drain_flush()
        await asyncio.to_thread(self._flush)
        await MemoryRuntime.close(self)

    def _load(self) -> None:
        path = Path(self._state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            _ensure_runtime_schema(connection)
            rows = connection.execute(
                f"SELECT component, payload FROM {storage_name('runtime_repository_records')} "
                "WHERE namespace = ? ORDER BY component, record_key",
                (self.persistence.namespace,),
            ).fetchall()
        payload: dict[str, JsonValue] = {}
        for component, raw in rows:
            value = json.loads(str(raw))
            if not isinstance(value, dict):
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            payload.setdefault(str(component), []).append(value)
        self._load_payload(payload)

    def _flush(self) -> None:
        path = Path(self._state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._repository_payload()
        with sqlite3.connect(path) as connection:
            _ensure_runtime_schema(connection)
            connection.execute(
                f"DELETE FROM {storage_name('runtime_repository_records')} WHERE namespace = ?",
                (self.persistence.namespace,),
            )
            for component, records in payload.items():
                if not isinstance(records, list):
                    continue
                for value in records:
                    if not isinstance(value, dict):
                        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    key = _sql_record_key(str(component), value)
                    tenant_id = str(value.get("tenant_id", ""))
                    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    connection.execute(
                        f"INSERT INTO {storage_name('runtime_repository_records')}(namespace, component, record_key, tenant_id, payload) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(namespace, component, record_key) DO UPDATE SET "
                        "tenant_id=excluded.tenant_id, payload=excluded.payload",
                        (self.persistence.namespace, component, key, tenant_id, encoded),
                    )


def _ensure_runtime_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {storage_name('runtime_repository_records')} ("
        "namespace TEXT NOT NULL, component TEXT NOT NULL, record_key TEXT NOT NULL, "
        "tenant_id TEXT NOT NULL, payload TEXT NOT NULL, "
        "PRIMARY KEY(namespace, component, record_key))"
    )


def _sql_record_key(component: str, value: dict[str, JsonValue]) -> str:
    nested = value.get("turn") or value.get("view") or value.get("node")
    record = nested if isinstance(nested, dict) else value
    keys = {
        "sessions": ("tenant_id", "session_id"),
        "turns": ("tenant_id", "session_id", "sequence"),
        "executions": ("tenant_id", "execution_id"),
        "results": ("tenant_id", "execution_id"),
        "idempotency": ("tenant_id", "scope", "key_hash"),
        "events": ("tenant_id", "execution_id", "sequence"),
        "traces": ("tenant_id", "execution_id", "sequence"),
        "task_plans": ("tenant_id", "graph_id"),
        "task_nodes": ("tenant_id", "graph_id", "task_id"),
        "evaluations": ("tenant_id", "evaluation_id"),
        "memories": ("tenant_id", "memory_id"),
        "artifacts": ("tenant_id", "artifact_id"),
        "approvals": ("tenant_id", "approval_id"),
        "externals": ("tenant_id", "call_id"),
        "operations": ("tenant_id", "operation_id"),
        "tools": ("tenant_id", "operation_id"),
        "blobs": ("tenant_id", "digest"),
    }.get(component)
    if keys is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return "\x1f".join(
        str(value.get(key, "") if key == "tenant_id" else record.get(key, ""))
        for key in keys
    )


def _build_runtime(mode: RuntimePersistenceMode, *, namespace: str, state_path: str | None = None, atomic_domain_id: str | None = None, local_tenant_id: str | None = None) -> MemoryRuntime:
    domain = atomic_domain_id or f"{mode.value.lower()}-domain-{uuid.uuid4().hex}"
    args = (mode, namespace, domain)
    sessions = _SessionRepository(*args)
    executions = _ExecutionRepository(*args)
    components: tuple[RuntimeRepository, ...] = (
        sessions, executions,
        _ResultRepository(executions, *args), _IdempotencyRepository(*args), _EventRepository(executions, *args), _TraceRepository(executions, *args),
        _TaskRepository(*args), _EvaluationRepository(*args), _MemoryRepository(*args), _ArtifactRepository(*args),
        _ApprovalRepository(*args), _ExternalRepository(*args), _OperationRepository(*args), _ToolRepository(*args),
        FileBlobStore(Path(state_path).parent, *args) if mode is RuntimePersistenceMode.FILE and state_path is not None else SqlBlobStore(state_path, *args) if mode is RuntimePersistenceMode.SQL and state_path is not None else MemoryBlobStore(*args),
    )
    if local_tenant_id is not None:
        for component in components:
            if isinstance(component, _Base):
                component._local_tenant_id = local_tenant_id
    persistence = RuntimePersistence(
        mode, namespace, sessions, executions, components[2], components[3], components[4], components[5],
        components[6], components[7], components[8], components[9], components[10], components[11], components[12], components[13], components[14],
        local_tenant_id,
    )
    if mode is RuntimePersistenceMode.FILE:
        if state_path is None:
            raise ValueError("FILE runtime requires a state path")
        return FileRuntime(persistence, components, state_path)
    if mode is RuntimePersistenceMode.SQL:
        if state_path is None:
            raise ValueError("SQL runtime requires a database path")
        return SqlRuntime(persistence, components, state_path)
    return MemoryRuntime(persistence, components)


def build_memory_runtime(*, namespace: str | None = None) -> MemoryRuntime:
    return _build_runtime(RuntimePersistenceMode.MEMORY, namespace=namespace or f"memory-{uuid.uuid4().hex}")


def build_file_runtime(root: str, *, project_id: str, local_tenant_id: str) -> FileRuntime:
    if not local_tenant_id.strip() or not project_id.strip():
        raise ValueError("FILE runtime identity is incomplete")
    root_path = Path(root).expanduser().resolve()
    digest = hashlib.sha256(str(root_path).encode("utf-8")).hexdigest()
    namespace = f"{project_id}:{digest}"
    return _build_runtime(RuntimePersistenceMode.FILE, namespace=namespace, state_path=str(root_path / ".linktools" / "runtime-file-manifest.json"), atomic_domain_id=digest, local_tenant_id=local_tenant_id)


def build_sql_runtime(database_path: str, *, namespace: str, deployment_id: str, schema_digest: str = "runtime") -> SqlRuntime:
    if not namespace.strip() or not deployment_id.strip():
        raise ValueError("SQL runtime identity is incomplete")
    path = Path(database_path).expanduser().resolve()
    return _build_runtime(RuntimePersistenceMode.SQL, namespace=namespace, state_path=str(path), atomic_domain_id=canonical_sha256({"deployment_id": deployment_id, "schema_digest": schema_digest}))


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
    return SessionRecord(str(value["session_id"]), str(value["tenant_id"]), str(value["owner_principal_id"]), str(value["binding_digest"]), SessionStatus(str(value["status"])), int(value["revision"]), int(value["resource_generation"]), None if value.get("cwd") is None else str(value["cwd"]), value.get("metadata", {}), _time(value["created_at"]), _time(value["updated_at"]), None if value.get("closed_at") is None else _time(value["closed_at"]))


def _execution_from_json(value: dict[str, JsonValue]) -> ExecutionRecord:
    return ExecutionRecord(str(value["execution_id"]), str(value["tenant_id"]), None if value.get("session_id") is None else str(value["session_id"]), ExecutionProfile(str(value["profile"])), str(value["binding_digest"]), None if value.get("parent_execution_id") is None else str(value["parent_execution_id"]), str(value["root_execution_id"]), ExecutionStatus(str(value["status"])), int(value["snapshot_revision"]), int(value["event_sequence"]), int(value["trace_sequence"]), None if value.get("result_ref") is None else str(value["result_ref"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), value.get("safe_error_details", {}), _time(value["created_at"]), _time(value["updated_at"]))


def _idempotency_from_json(value: dict[str, JsonValue]) -> IdempotencyRecord:
    return IdempotencyRecord(str(value["tenant_id"]), str(value["scope"]), "", str(value["request_digest"]), str(value["execution_id"]), IdempotencyStatus(str(value["status"])), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), _time(value["created_at"]), _time(value["updated_at"]))


def _json_record(record: object) -> dict[str, JsonValue]:
    return _json_value(asdict(record))


def _idempotency_json(record: IdempotencyRecord, key_hash: str) -> dict[str, JsonValue]:
    value = _record_json(record)
    value.pop("key", None)
    value["key_hash"] = key_hash
    return value


def _turn_from_json(value: dict[str, JsonValue]) -> SessionTurnRecord:
    return SessionTurnRecord(str(value["session_id"]), str(value["tenant_id"]), int(value["sequence"]), str(value["execution_id"]), str(value["input_digest"]), tuple(value.get("delta_messages", [])), CaptureState(str(value["capture_state"])), None if value.get("completed_at") is None else _time(str(value["completed_at"])))


def _result_from_json(value: dict[str, JsonValue]) -> ResultRecord:
    return ResultRecord(str(value["execution_id"]), str(value["tenant_id"]), ExecutionStatus(str(value["status"])), str(value["output_schema_id"]), int(value["output_schema_revision"]), str(value["output_schema_fingerprint"]), None if value.get("payload_ref") is None else str(value["payload_ref"]), None if value.get("payload_digest") is None else str(value["payload_digest"]), StopReason(str(value["stop_reason"])), int(value["input_tokens"]), int(value["output_tokens"]), int(value["total_cost_micros"]), _time(str(value["created_at"])))


def _event_from_json(value: dict[str, JsonValue]) -> ExecutionEventRecord:
    return ExecutionEventRecord(str(value["execution_id"]), str(value["tenant_id"]), int(value["sequence"]), ExecutionEventType(str(value["event_type"])), value.get("payload", {}))


def _trace_from_json(value: dict[str, JsonValue]) -> TraceRecord:
    return TraceRecord(str(value["execution_id"]), str(value["tenant_id"]), int(value["sequence"]), TraceKind(str(value["kind"])), value.get("payload", {}))


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


def _read_file_envelope(path: Path, namespace: str) -> dict[str, JsonValue]:
    try:
        value = read_json(path)
        if value.get("schema_version") != 1 or value.get("namespace") != namespace:
            raise ValueError("file envelope identity mismatch")
        payload = value.get("payload")
        if not isinstance(payload, dict) or value.get("payload_sha256") != canonical_sha256(payload):
            raise ValueError("file envelope digest mismatch")
        return payload
    except LinktoolsAIError:
        raise
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _read_sequence_file(path: Path, namespace: str) -> list[dict[str, JsonValue]]:
    try:
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("sequence has an incomplete tail")
        previous: str | None = None
        expected = 1
        payloads: list[dict[str, JsonValue]] = []
        for line in raw.splitlines():
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("namespace") != namespace:
                raise ValueError("sequence envelope")
            if int(value["sequence"]) != expected or value.get("previous_line_sha256") != previous:
                raise ValueError("sequence link")
            payload = value["payload"]
            if not isinstance(payload, dict) or value.get("payload_sha256") != canonical_sha256(payload):
                raise ValueError("payload digest")
            line_digest = value.get("line_sha256")
            unsigned = dict(value)
            unsigned.pop("line_sha256", None)
            if line_digest != canonical_sha256(unsigned):
                raise ValueError("line digest")
            payloads.append(payload)
            previous = str(line_digest)
            expected += 1
        return payloads
    except LinktoolsAIError:
        raise
    except (OSError, TypeError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _component(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).rstrip(b"=").decode("ascii")


def _decode_component(value: str) -> str:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode((value + padding).encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _key_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_blob_digest(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["FileBlobStore", "FileRuntime", "MemoryBlobStore", "MemoryRuntime", "SqlBlobStore", "SqlRuntime", "build_file_runtime", "build_memory_runtime", "build_sql_runtime"]
