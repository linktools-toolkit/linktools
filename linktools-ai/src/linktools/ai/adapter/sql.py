#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL RuntimePersistence owner shared by SQLite, MySQL and PostgreSQL."""

import base64
import asyncio
import json
import re
import hashlib
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import TYPE_CHECKING

from ..capability.tool import ToolOperationRecord, ToolStateStore
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.json import JsonValue, canonical_json_bytes
from ..core.paging import Page
from ..core.principal import ResourceRef
from ..core.value import (
    ApprovalDecision, ApprovalStatus, EvaluationStatus, ExecutionEventType, ExecutionProfile,
    ExecutionStatus, ExternalCallStatus, IdempotencyStatus, OperationKind, OperationStatus, ResourceKind,
    SessionStatus, StopReason, TaskStatus, ToolOperationStatus, ExecutionLineageKind,
)
from ..runtime.persistence import (
    ApprovalRecord, ArtifactRecord, BlobRef, BlobStore, EvaluationRecord, ExecutionEventRecord,
    ExecutionRecord, ExecutionStartClaim, ExecutionStartReservation, ExecutionStartReservationResult, ExecutionStartUnknownCommit, ExecutionTerminalCommit, ExecutionTerminalCommitResult, ExternalResultRecord,
    IdempotencyRecord, MemoryRecord, OperationLedgerInput, OperationLedgerRecord, RuntimeBackend,
    ResultRecord, RuntimePersistence, RuntimePersistenceMode, RuntimeRepository, SessionRecord,
    TaskLease, TaskNodeView,
)
from ..task.model import TaskGraph, TaskGraphView, TaskNode, TaskTerminalRecord
from ..storage.database import StorageDatabase
from linktools.core import environ
from .schema import SqlRuntimeTables

if TYPE_CHECKING:
    from sqlalchemy import Table


_MAX_SQL_BLOB_BYTES = 2 * 1024 * 1024 * 1024
_MAX_SQL_BLOB_CHUNK = 1024 * 1024
_MAX_SQL_INLINE_BLOB_BYTES = 4 * 1024 * 1024
_logger = environ.get_logger("ai.adapter.sql")


class _SqlRuntimeRepository(ToolStateStore, BlobStore, RuntimeRepository):
    def __init__(self, owner: "_SqlRuntimeOwner", table_name: str) -> None:
        self._owner = owner
        self._table = owner.tables[table_name]
        self._table_name = table_name

    @property
    def mode(self) -> RuntimePersistenceMode:
        return RuntimePersistenceMode.SQL

    @property
    def backend(self) -> RuntimeBackend:
        return self._owner.backend

    @property
    def namespace(self) -> str:
        return self._owner.namespace

    @property
    def namespace_key(self) -> str:
        return self._owner.namespace_key

    @property
    def atomic_domain_id(self) -> str:
        return self._owner.atomic_domain_id

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def _get(self, record_id: str, *, tenant_id: str, table: "Table | None" = None) -> object | None:
        from sqlalchemy import select
        target = self._table if table is None else table
        async with self._owner.database.session_factory() as session:
            row = (await session.execute(select(target).where(target.c.namespace_key == self.namespace_key, target.c.tenant_id == tenant_id, target.c.record_id == record_id))).mappings().first()
        return None if row is None else _decode_payload(self._owner.table_name(target), row["payload"])

    async def _insert(self, record: object, *, record_id: str, tenant_id: str, sequence: int = 0, revision: int = 0, status: str = "", table: "Table | None" = None) -> object:
        target = self._table if table is None else table
        now = _record_time(record)
        values = {"namespace_key": self.namespace_key, "tenant_id": tenant_id, "record_id": record_id, "sequence": sequence, "revision": revision, "status": status, "payload": _encode_payload(record), "created_at": now, "updated_at": now}
        if isinstance(record, SessionRecord):
            values.update(session_id=record.session_id, profile=record.profile.value, head_execution_id=record.head_execution_id)
        if isinstance(record, ExecutionRecord):
            values.update(session_id=record.session_id, parent_execution_id=record.parent_execution_id, source_execution_id=record.source_execution_id, base_execution_id=record.base_execution_id, lineage_kind=record.lineage_kind, agent_run_sequence=record.agent_run_sequence)
        if isinstance(record, ToolOperationRecord):
            values.update(run_id=record.run_id, tool_call_id=record.tool_call_id, owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at)
        if isinstance(record, TaskNodeView):
            values.update(owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at)
        if isinstance(record, IdempotencyRecord):
            values.update(scope=record.scope, key_hash=record.key_hash)
        if isinstance(record, OperationLedgerRecord):
            values.update(resource_kind=record.resource_kind.value, resource_id=record.resource_id)
        from sqlalchemy import insert
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                await session.execute(insert(target).values(values))
        return record

    async def _replace(self, record: object, *, record_id: str, tenant_id: str, expected_revision: int, revision: int, status: str = "", table: "Table | None" = None) -> object:
        target = self._table if table is None else table
        from sqlalchemy import update
        now = _record_time(record)
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                values = {"payload": _encode_payload(record), "revision": revision, "status": status, "updated_at": now}
                if isinstance(record, SessionRecord):
                    values.update(session_id=record.session_id, profile=record.profile.value, head_execution_id=record.head_execution_id)
                if isinstance(record, ExecutionRecord):
                    values.update(session_id=record.session_id, parent_execution_id=record.parent_execution_id, source_execution_id=record.source_execution_id, base_execution_id=record.base_execution_id, lineage_kind=record.lineage_kind, agent_run_sequence=record.agent_run_sequence)
                if isinstance(record, ToolOperationRecord):
                    values.update(run_id=record.run_id, tool_call_id=record.tool_call_id, owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at)
                if isinstance(record, TaskNodeView):
                    values.update(owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at)
                if isinstance(record, IdempotencyRecord):
                    values.update(scope=record.scope, key_hash=record.key_hash)
                if isinstance(record, OperationLedgerRecord):
                    values.update(resource_kind=record.resource_kind.value, resource_id=record.resource_id)
                result = await session.execute(update(target).where(target.c.namespace_key == self.namespace_key, target.c.tenant_id == tenant_id, target.c.record_id == record_id, target.c.revision == expected_revision).values(values))
                if result.rowcount != 1:
                    current = await self._get(record_id, tenant_id=tenant_id, table=target)
                    if current is None:
                        raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        return record

    async def create(self, record: object) -> object:
        tenant_id = str(record.tenant_id)
        record_id = _record_id(record)
        try:
            return await self._insert(record, record_id=record_id, tenant_id=tenant_id, revision=_revision(record), status=_status(record))
        except Exception as error:
            if _is_integrity(error):
                existing = await self._get(record_id, tenant_id=tenant_id)
                if existing == record:
                    return existing
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT) from error
            raise

    async def get_header(self, record_id: str, *, tenant_id: str) -> ResourceRef | None:
        record = await self._get(record_id, tenant_id=tenant_id)
        return None if record is None else ResourceRef(_resource_kind(self._table_name), record_id, tenant_id)

    async def get(self, record_id: str, key_hash: "str | None" = None, *, tenant_id: str) -> "object | None":
        if self._table_name == "idempotency":
            if key_hash is None:
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self.get_idempotency(record_id, key_hash, tenant_id=tenant_id)
        return await self._get(record_id, tenant_id=tenant_id)

    async def list(self, record_id: "str | None" = None, *, tenant_id: str, owner_principal_id: "str | None" = None, owner_id: "str | None" = None, cursor: "str | None" = None, after_sequence: int = 0, limit: int = 100) -> "tuple[SessionRecord, ...] | Page[object]":
        if self._table_name == "execution_events":
            if record_id is None:
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._list_event(record_id, tenant_id=tenant_id, after_sequence=after_sequence, limit=limit)
        values = await self._list_records(tenant_id=tenant_id)
        if self._table_name == "memories":
            return Page(tuple(item for item in values if isinstance(item, MemoryRecord) and item.owner_id == owner_id)[:limit], None)
        return tuple(item for item in values if owner_principal_id is None or item.owner_principal_id == owner_principal_id)

    async def _list_records(self, *, tenant_id: str, table: "Table | None" = None) -> tuple[object, ...]:
        from sqlalchemy import select
        target = self._table if table is None else table
        async with self._owner.database.session_factory() as session:
            rows = (await session.execute(select(target).where(target.c.namespace_key == self.namespace_key, target.c.tenant_id == tenant_id).order_by(target.c.record_id))).mappings().all()
        logical_name = self._owner.table_name(target)
        return tuple(_decode_payload(logical_name, row["payload"]) for row in rows)

    async def compare_and_swap(self, record_id: str, *, tenant_id: str, expected_revision: int = 0, expected_snapshot_revision: "int | None" = None, expected_status: "IdempotencyStatus | None" = None, next_record: object) -> object:
        if self._table_name == "idempotency":
            if not isinstance(next_record, IdempotencyRecord) or expected_status is None:
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self.compare_idempotency(record_id, next_record.key_hash, tenant_id=tenant_id, expected_status=expected_status, next_record=next_record)
        if self._table_name == "operation_ledger":
            if not isinstance(next_record, OperationLedgerRecord) or not isinstance(expected_status, OperationStatus):
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            current = await self._get(record_id, tenant_id=tenant_id)
            if not isinstance(current, OperationLedgerRecord) or current.status is not expected_status or next_record.sequence != current.sequence or next_record.tenant_id != current.tenant_id or next_record.resource_kind is not current.resource_kind or next_record.resource_id != current.resource_id or next_record.operation_id != current.operation_id:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            return await self._replace_status(next_record, table_name="operation_ledger", tenant_id=tenant_id, record_id=record_id, expected_status=expected_status)
        if isinstance(next_record, ExecutionRecord):
            if expected_snapshot_revision is None:
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            expected_revision = expected_snapshot_revision
            if next_record.snapshot_revision != expected_revision + 1:
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        if isinstance(next_record, SessionRecord) and next_record.revision != expected_revision + 1:
            raise LinktoolsAIError(ErrorCode.SESSION_REVISION_CONFLICT)
        return await self._replace(next_record, record_id=record_id, tenant_id=tenant_id, expected_revision=expected_revision, revision=expected_revision + 1, status=_status(next_record))

    async def list_by_session(self, session_id: str, *, tenant_id: str, statuses: frozenset[ExecutionStatus] | None = None) -> tuple[ExecutionRecord, ...]:
        values = tuple(item for item in await self._list_records(tenant_id=tenant_id) if item.session_id == session_id)
        return tuple(item for item in values if statuses is None or item.status in statuses)

    async def list_children(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionRecord, ...]:
        values = tuple(item for item in await self._list_records(tenant_id=tenant_id) if item.parent_execution_id == execution_id)
        return tuple(sorted(values, key=lambda item: item.execution_id))

    async def create_plan(self, graph: TaskGraph, *, tenant_id: str) -> TaskGraphView:
        from sqlalchemy import insert
        view = TaskGraphView(graph.graph_id, TaskStatus.PENDING, graph.nodes)
        graph_table = self._owner.tables["task_graphs"]
        node_table = self._owner.tables["task_nodes"]
        now = datetime.now(timezone.utc)
        try:
            async with self._owner.database.session_factory() as session:
                async with session.begin():
                    await session.execute(insert(graph_table).values(namespace_key=self.namespace_key, tenant_id=tenant_id, record_id=graph.graph_id, sequence=0, revision=0, status=view.status.value, payload=_encode_payload(view), created_at=now, updated_at=now))
                    for node in graph.nodes:
                        record = TaskNodeView(graph.graph_id, node.task_id, node.dependencies, TaskStatus.PENDING, None, 0, None, None, None, None)
                        await session.execute(insert(node_table).values(namespace_key=self.namespace_key, tenant_id=tenant_id, record_id=f"{graph.graph_id}:{node.task_id}", sequence=0, revision=0, status=TaskStatus.PENDING.value, owner=None, fence=0, lease_expires_at=None, payload=_encode_payload(record), created_at=now, updated_at=now))
        except Exception as error:
            if _is_integrity(error):
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT) from error
            raise
        return view

    async def get_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None:
        value = await self._get(graph_id, tenant_id=tenant_id, table=self._owner.tables["task_graphs"])
        return value if isinstance(value, TaskGraphView) else None

    async def cancel_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        view = await self.get_plan(graph_id, tenant_id=tenant_id)
        if view is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        updated = TaskGraphView(view.graph_id, TaskStatus.CANCELLED, view.nodes)
        await self._replace(updated, record_id=graph_id, tenant_id=tenant_id, expected_revision=0, revision=0, status=updated.status.value, table=self._owner.tables["task_graphs"])
        return updated

    async def _claim_task(self, graph_id: str, task_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease:
        node = await self._get(f"{graph_id}:{task_id}", tenant_id=tenant_id, table=self._owner.tables["task_nodes"])
        if not isinstance(node, TaskNodeView):
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        dependencies_ready = True
        for dependency in node.dependencies:
            dependency_node = await self._get(f"{graph_id}:{dependency}", tenant_id=tenant_id, table=self._owner.tables["task_nodes"])
            if not isinstance(dependency_node, TaskNodeView) or dependency_node.status is not TaskStatus.SUCCEEDED:
                dependencies_ready = False
                break
        now = datetime.now(timezone.utc)
        expired = node.status is TaskStatus.RUNNING and node.lease_expires_at is not None and node.lease_expires_at <= now
        if (node.status not in {TaskStatus.PENDING, TaskStatus.READY} and not expired) or not dependencies_ready:
            raise LinktoolsAIError(ErrorCode.TASK_OWNER_CONFLICT)
        lease = TaskLease(graph_id, task_id, tenant_id, owner, node.fence + 1, now + timedelta(seconds=lease_seconds))
        updated = replace(node, status=TaskStatus.RUNNING, owner=owner, fence=lease.fence, lease_expires_at=lease.lease_expires_at)
        await self._task_update(updated, tenant_id=tenant_id, expected_status=node.status, expected_owner=node.owner if expired else None, expected_fence=node.fence, expected_lease_before=now if expired else None)
        return lease

    async def _renew_task(self, lease: TaskLease, *, tenant_id: str) -> TaskLease:
        node = await self._task_node(lease, tenant_id)
        renewed = replace(lease, lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=max(1, int((lease.lease_expires_at - datetime.now(timezone.utc)).total_seconds()))) )
        await self._task_update(replace(node, lease_expires_at=renewed.lease_expires_at), tenant_id=tenant_id, expected_status=TaskStatus.RUNNING, expected_owner=lease.owner, expected_fence=lease.fence, expected_lease_after=datetime.now(timezone.utc))
        return renewed

    async def _complete_task(self, lease: TaskLease, *, tenant_id: str, result_digest: str) -> TaskTerminalRecord:
        node = await self._task_node(lease, tenant_id)
        terminal = TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, TaskStatus.SUCCEEDED, result_digest, None, None)
        await self._task_update(replace(node, status=TaskStatus.SUCCEEDED, owner=None, lease_expires_at=None, result_digest=result_digest), tenant_id=tenant_id, expected_status=TaskStatus.RUNNING, expected_owner=lease.owner, expected_fence=lease.fence, expected_lease_after=datetime.now(timezone.utc))
        return terminal

    async def _fail_task(self, lease: TaskLease, *, tenant_id: str, error_code: str, error_digest: str) -> TaskTerminalRecord:
        node = await self._task_node(lease, tenant_id)
        terminal = TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, TaskStatus.FAILED, None, error_code, error_digest)
        await self._task_update(replace(node, status=TaskStatus.FAILED, owner=None, lease_expires_at=None, error_code=error_code, error_digest=error_digest), tenant_id=tenant_id, expected_status=TaskStatus.RUNNING, expected_owner=lease.owner, expected_fence=lease.fence, expected_lease_after=datetime.now(timezone.utc))
        return terminal

    async def _task_node(self, lease: TaskLease, tenant_id: str) -> TaskNodeView:
        node = await self._get(f"{lease.graph_id}:{lease.task_id}", tenant_id=tenant_id, table=self._owner.tables["task_nodes"])
        if not isinstance(node, TaskNodeView) or node.owner != lease.owner or node.fence != lease.fence or node.lease_expires_at is None or node.lease_expires_at <= datetime.now(timezone.utc):
            raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        return node

    async def _task_update(self, record: TaskNodeView, *, tenant_id: str, expected_status: TaskStatus, expected_owner: "str | None", expected_fence: int, expected_lease_after: "datetime | None" = None, expected_lease_before: "datetime | None" = None) -> TaskNodeView:
        from sqlalchemy import select, update
        table = self._owner.tables["task_nodes"]
        record_id = f"{record.graph_id}:{record.task_id}"
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(table).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record_id))).mappings().first()
                if row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                predicate = [table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record_id, table.c.revision == row["revision"], table.c.status == expected_status.value, table.c.fence == expected_fence]
                predicate.append(table.c.owner.is_(None) if expected_owner is None else table.c.owner == expected_owner)
                if expected_lease_after is not None:
                    predicate.append(table.c.lease_expires_at > expected_lease_after)
                if expected_lease_before is not None:
                    predicate.append(table.c.lease_expires_at <= expected_lease_before)
                outcome = await session.execute(update(table).where(*predicate).values(payload=_encode_payload(record), revision=int(row["revision"]) + 1, status=record.status.value, owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at, updated_at=_record_time(record)))
                if outcome.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        return record

    async def _replace_status(self, record: object, *, table_name: str, tenant_id: str, record_id: str, expected_status: object) -> object:
        from sqlalchemy import select, update
        table = self._owner.tables[table_name]
        status = expected_status.value if isinstance(expected_status, StrEnum) else str(expected_status)
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(table).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record_id))).mappings().first()
                if row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                outcome = await session.execute(update(table).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record_id, table.c.revision == row["revision"], table.c.status == status).values(payload=_encode_payload(record), revision=int(row["revision"]) + 1, status=_status(record), updated_at=_record_time(record)))
                if outcome.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        return record

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[TaskNodeView, ...]:
        values = await self._list_records(tenant_id=tenant_id, table=self._owner.tables["task_nodes"])
        return tuple(item for item in values if item.graph_id == graph_id)

    async def _list_evaluations(self, execution_id: str, *, tenant_id: str) -> tuple[EvaluationRecord, ...]:
        values = await self._list_records(tenant_id=tenant_id)
        return tuple(item for item in values if isinstance(item, EvaluationRecord) and item.execution_id == execution_id)

    async def put(self, record: MemoryRecord, *, expected_revision: "int | None") -> MemoryRecord:
        current = await self._get(record.memory_id, tenant_id=record.tenant_id)
        if current is None:
            return await self._insert(record, record_id=record.memory_id, tenant_id=record.tenant_id, revision=record.revision, table=self._owner.tables["memories"])
        if not isinstance(current, MemoryRecord) or expected_revision is None or current.revision != expected_revision:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        return await self._replace(record, record_id=record.memory_id, tenant_id=record.tenant_id, expected_revision=expected_revision, revision=record.revision, table=self._owner.tables["memories"])

    async def list_memories(self, *, tenant_id: str, owner_id: str, cursor: "str | None", limit: int) -> Page[MemoryRecord]:
        values = tuple(item for item in await self._list_records(tenant_id=tenant_id) if isinstance(item, MemoryRecord) and item.owner_id == owner_id)
        return Page(values[:limit], None)

    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord:
        return await self.create(record)

    async def get_metadata(self, artifact_id: str, *, tenant_id: str) -> ArtifactRecord | None:
        value = await self._get(artifact_id, tenant_id=tenant_id, table=self._owner.tables["artifacts"])
        return value if isinstance(value, ArtifactRecord) else None

    async def list_by_execution(self, execution_id: str, *, tenant_id: str, cursor: "str | None" = None, limit: int = 100) -> "Page[ArtifactRecord] | tuple[EvaluationRecord, ...]":
        if self._table_name == "idempotency":
            values = tuple(item for item in await self._list_records(tenant_id=tenant_id) if isinstance(item, IdempotencyRecord) and item.execution_id == execution_id)
            return tuple(sorted(values, key=lambda item: (item.scope, item.key_hash)))
        if self._table_name == "evaluations":
            return await self._list_evaluations(execution_id, tenant_id=tenant_id)
        values = tuple(item for item in await self._list_records(tenant_id=tenant_id) if isinstance(item, ArtifactRecord) and item.execution_id == execution_id)
        return Page(values[:limit], None)

    async def decide(self, approval_id: str, *, tenant_id: str, expected_status: ApprovalStatus, decision_id: str, decision: ApprovalDecision, principal_id: str, decision_digest: str, decided_at: datetime) -> ApprovalRecord:
        current = await self._get(approval_id, tenant_id=tenant_id)
        if not isinstance(current, ApprovalRecord) or current.status is not expected_status:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        updated = replace(current, status=ApprovalStatus.APPROVED if decision is ApprovalDecision.APPROVE else ApprovalStatus.DENIED, decision_id=decision_id, decision=decision, decided_by=principal_id, decision_digest=decision_digest, decided_at=decided_at)
        return await self._replace_status(updated, table_name="approvals", tenant_id=tenant_id, record_id=approval_id, expected_status=expected_status)

    async def _list_external(self, execution_id: str, *, tenant_id: str) -> tuple[object, ...]:
        values = await self._list_records(tenant_id=tenant_id)
        return tuple(item for item in values if (isinstance(item, ApprovalRecord) and item.execution_id == execution_id and item.status is ApprovalStatus.PENDING) or (isinstance(item, ExternalResultRecord) and item.execution_id == execution_id and item.status is ExternalCallStatus.PENDING))

    async def create_call(self, record: ExternalResultRecord) -> ExternalResultRecord:
        return await self.create(record)

    async def supply(self, call_id: str, *, tenant_id: str, expected_status: ExternalCallStatus, result_id: str, payload_ref: str, payload_digest: str, supplied_at: datetime) -> ExternalResultRecord:
        current = await self._get(call_id, tenant_id=tenant_id, table=self._owner.tables["external_results"])
        if not isinstance(current, ExternalResultRecord) or current.status is not expected_status:
            raise LinktoolsAIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        updated = replace(current, status=ExternalCallStatus.SUPPLIED, result_id=result_id, payload_ref=payload_ref, payload_digest=payload_digest, supplied_at=supplied_at)
        return await self._replace_status(updated, table_name="external_results", tenant_id=tenant_id, record_id=call_id, expected_status=expected_status)

    async def advance_sequence(self, execution_id: str, *, tenant_id: str, kind: str, expected_sequence: int) -> ExecutionRecord:
        current = await self._get(execution_id, tenant_id=tenant_id)
        if current is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        if kind != "event":
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        field = "event_sequence"
        current_sequence = current.event_sequence
        if current_sequence != expected_sequence:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        updated = replace(current, **{field: expected_sequence + 1, "snapshot_revision": current.snapshot_revision + 1, "updated_at": datetime.now(timezone.utc)})
        return await self._replace(updated, record_id=execution_id, tenant_id=tenant_id, expected_revision=current.snapshot_revision, revision=updated.snapshot_revision)

    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord:
        from sqlalchemy import insert, select, update
        if self._table_name != "executions":
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        execution_table = self._table
        idempotency_table = self._owner.tables["idempotency"]
        event_table = self._owner.tables["execution_events"]
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                execution_row = (await session.execute(select(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == claim.tenant_id, execution_table.c.record_id == claim.execution_id, execution_table.c.status == ExecutionStatus.PENDING_START.value, execution_table.c.revision == claim.expected_execution_revision, execution_table.c.sequence == claim.expected_event_sequence, execution_table.c.agent_run_sequence == 0))).mappings().first()
                identity_row = (await session.execute(select(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == claim.tenant_id, idempotency_table.c.scope == claim.scope, idempotency_table.c.key_hash == claim.key_hash))).mappings().first()
                if execution_row is None or identity_row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                current = _decode_payload("executions", execution_row["payload"])
                identity = _decode_payload("idempotency", identity_row["payload"])
                if not isinstance(current, ExecutionRecord) or not isinstance(identity, IdempotencyRecord):
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if identity.status is not IdempotencyStatus.RESERVED or identity.execution_id != claim.execution_id or identity.request_digest != claim.request_digest:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                started = replace(current, status=ExecutionStatus.STARTED, snapshot_revision=current.snapshot_revision + 1, event_sequence=current.event_sequence + 1, updated_at=claim.started_at, agent_run_sequence=1)
                identity_started = replace(identity, status=IdempotencyStatus.STARTED, updated_at=claim.started_at)
                result = await session.execute(update(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == claim.tenant_id, execution_table.c.record_id == claim.execution_id, execution_table.c.revision == claim.expected_execution_revision, execution_table.c.status == ExecutionStatus.PENDING_START.value, execution_table.c.sequence == claim.expected_event_sequence, execution_table.c.agent_run_sequence == 0).values(payload=_encode_payload(started), revision=started.snapshot_revision, sequence=started.event_sequence, status=started.status.value, agent_run_sequence=1, updated_at=claim.started_at))
                if result.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                await session.execute(update(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == claim.tenant_id, idempotency_table.c.scope == claim.scope, idempotency_table.c.key_hash == claim.key_hash, idempotency_table.c.status == IdempotencyStatus.RESERVED.value).values(payload=_encode_payload(identity_started), status=identity_started.status.value, updated_at=claim.started_at))
                event = ExecutionEventRecord(claim.execution_id, claim.tenant_id, claim.expected_event_sequence + 1, ExecutionEventType.EXECUTION_STARTED, {})
                await session.execute(insert(event_table).values(namespace_key=self.namespace_key, tenant_id=claim.tenant_id, record_id=f"{claim.execution_id}:{event.sequence}", sequence=event.sequence, revision=0, status=event.event_type.value, payload=_encode_payload(event), created_at=claim.started_at, updated_at=claim.started_at))
                return started

    async def reserve_start(self, reservation: ExecutionStartReservation) -> ExecutionStartReservationResult:
        from sqlalchemy import insert, select
        if self._table_name != "executions" or reservation.execution.tenant_id != reservation.idempotency.tenant_id or reservation.execution.execution_id != reservation.idempotency.execution_id:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution_table = self._table
        idempotency_table = self._owner.tables["idempotency"]
        execution = reservation.execution
        identity = reservation.idempotency
        try:
            async with self._owner.database.session_factory() as session:
                async with session.begin():
                    await session.execute(insert(idempotency_table).values(namespace_key=self.namespace_key, tenant_id=identity.tenant_id, record_id=f"{identity.scope}:{identity.key_hash}", scope=identity.scope, key_hash=identity.key_hash, sequence=0, revision=0, status=identity.status.value, payload=_encode_payload(identity), created_at=identity.created_at, updated_at=identity.updated_at))
                    values = {"namespace_key": self.namespace_key, "tenant_id": execution.tenant_id, "record_id": execution.execution_id, "sequence": execution.event_sequence, "revision": execution.snapshot_revision, "status": execution.status.value, "payload": _encode_payload(execution), "created_at": execution.created_at, "updated_at": execution.updated_at, "session_id": execution.session_id, "parent_execution_id": execution.parent_execution_id, "source_execution_id": execution.source_execution_id, "base_execution_id": execution.base_execution_id, "lineage_kind": execution.lineage_kind, "agent_run_sequence": execution.agent_run_sequence}
                    await session.execute(insert(execution_table).values(values))
        except Exception as error:
            if not _is_integrity(error):
                raise LinktoolsAIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            async with self._owner.database.session_factory() as session:
                identity_row = (await session.execute(select(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == identity.tenant_id, idempotency_table.c.scope == identity.scope, idempotency_table.c.key_hash == identity.key_hash))).mappings().first()
                if identity_row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT) from error
                existing_identity = _decode_payload("idempotency", identity_row["payload"])
                if not isinstance(existing_identity, IdempotencyRecord):
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if existing_identity.request_digest != identity.request_digest:
                    raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                execution_row = (await session.execute(select(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == identity.tenant_id, execution_table.c.record_id == existing_identity.execution_id))).mappings().first()
                if execution_row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                existing_execution = _decode_payload("executions", execution_row["payload"])
                if not isinstance(existing_execution, ExecutionRecord):
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return ExecutionStartReservationResult(existing_execution, existing_identity, False)
        return ExecutionStartReservationResult(execution, identity, True)

    async def claim_next_agent_run(self, execution_id: str, *, tenant_id: str, expected_revision: int, expected_agent_run_sequence: int) -> ExecutionRecord:
        from sqlalchemy import select, update
        if self._table_name != "executions":
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        now = datetime.now(timezone.utc)
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(self._table).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == tenant_id, self._table.c.record_id == execution_id, self._table.c.revision == expected_revision, self._table.c.status == ExecutionStatus.STARTED.value, self._table.c.agent_run_sequence == expected_agent_run_sequence))).mappings().first()
                if row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                current = _decode_payload("executions", row["payload"])
                if not isinstance(current, ExecutionRecord):
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                updated = replace(current, snapshot_revision=current.snapshot_revision + 1, agent_run_sequence=current.agent_run_sequence + 1, updated_at=now)
                result = await session.execute(update(self._table).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == tenant_id, self._table.c.record_id == execution_id, self._table.c.revision == expected_revision, self._table.c.status == ExecutionStatus.STARTED.value, self._table.c.agent_run_sequence == expected_agent_run_sequence).values(payload=_encode_payload(updated), revision=updated.snapshot_revision, agent_run_sequence=updated.agent_run_sequence, updated_at=now))
                if result.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                return updated

    async def mark_start_unknown(self, commit: ExecutionStartUnknownCommit) -> ExecutionRecord:
        from sqlalchemy import select, update, insert
        if self._table_name != "executions":
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        event_table = self._owner.tables["execution_events"]
        idempotency_table = self._owner.tables["idempotency"]
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(self._table).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == commit.tenant_id, self._table.c.record_id == commit.execution_id, self._table.c.revision == commit.expected_execution_revision, self._table.c.status == ExecutionStatus.STARTED.value))).mappings().first()
                identity_row = (await session.execute(select(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == commit.tenant_id, idempotency_table.c.scope == commit.scope, idempotency_table.c.key_hash == commit.key_hash))).mappings().first()
                if row is None or identity_row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                current = _decode_payload("executions", row["payload"])
                identity = _decode_payload("idempotency", identity_row["payload"])
                if not isinstance(current, ExecutionRecord) or not isinstance(identity, IdempotencyRecord):
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if identity.status is not IdempotencyStatus.STARTED:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                unknown = replace(current, status=ExecutionStatus.START_UNKNOWN, snapshot_revision=current.snapshot_revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.started_at)
                await session.execute(update(self._table).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == commit.tenant_id, self._table.c.record_id == commit.execution_id, self._table.c.revision == commit.expected_execution_revision, self._table.c.status == ExecutionStatus.STARTED.value).values(payload=_encode_payload(unknown), revision=unknown.snapshot_revision, status=unknown.status.value, sequence=unknown.event_sequence, updated_at=commit.started_at))
                await session.execute(update(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == commit.tenant_id, idempotency_table.c.scope == commit.scope, idempotency_table.c.key_hash == commit.key_hash, idempotency_table.c.status == IdempotencyStatus.STARTED.value).values(payload=_encode_payload(replace(identity, status=IdempotencyStatus.START_UNKNOWN, updated_at=commit.started_at)), status=IdempotencyStatus.START_UNKNOWN.value, updated_at=commit.started_at))
                event = ExecutionEventRecord(commit.execution_id, commit.tenant_id, current.event_sequence + 1, ExecutionEventType.EXECUTION_START_UNKNOWN, {})
                await session.execute(insert(event_table).values(namespace_key=self.namespace_key, tenant_id=commit.tenant_id, record_id=f"{commit.execution_id}:{event.sequence}", sequence=event.sequence, revision=0, status=event.event_type.value, payload=_encode_payload(event), created_at=commit.started_at, updated_at=commit.started_at))
                return unknown

    async def commit_terminal(self, execution_id: "str | ExecutionTerminalCommit", *, tenant_id: "str | None" = None, expected_revision: "int | None" = None, next_record: "ExecutionRecord | None" = None) -> "ExecutionRecord | ExecutionTerminalCommitResult":
        if self._table_name == "results":
            if not isinstance(execution_id, ExecutionTerminalCommit):
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._commit_result(execution_id)
        if not isinstance(execution_id, str) or tenant_id is None or expected_revision is None or next_record is None:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self._get(execution_id, tenant_id=tenant_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        return await self._replace(next_record, record_id=execution_id, tenant_id=tenant_id, expected_revision=expected_revision, revision=expected_revision + 1, status=next_record.status.value)

    async def _commit_result(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult:
        from sqlalchemy import insert, select, update
        execution_table = self._owner.tables["executions"]
        result_table = self._owner.tables["results"]
        event_table = self._owner.tables["execution_events"]
        idempotency_table = self._owner.tables["idempotency"]
        operation_table = self._owner.tables["operation_ledger"]
        session_table = self._owner.tables["sessions"]
        execution = commit.terminal_execution
        if execution.event_sequence != commit.expected_event_sequence + 1 or commit.terminal_event_type not in {ExecutionEventType.EXECUTION_SUCCEEDED, ExecutionEventType.EXECUTION_FAILED, ExecutionEventType.EXECUTION_CANCELLED}:
            raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        if commit.session_head is not None and (execution.status is not ExecutionStatus.SUCCEEDED or execution.lineage_kind not in {"SESSION_RESUME", "RETRY"}):
            raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == execution.tenant_id, execution_table.c.record_id == execution.execution_id).with_for_update())).mappings().first()
                if row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                current = _decode_payload("executions", row["payload"])
                existing_result_row = (await session.execute(select(result_table).where(result_table.c.namespace_key == self.namespace_key, result_table.c.tenant_id == execution.tenant_id, result_table.c.record_id == execution.execution_id))).mappings().first()
                if existing_result_row is not None:
                    existing_result = _decode_payload("results", existing_result_row["payload"])
                    terminal_event = (await session.execute(select(event_table).where(event_table.c.namespace_key == self.namespace_key, event_table.c.tenant_id == execution.tenant_id, event_table.c.record_id == f"{execution.execution_id}:{commit.expected_event_sequence + 1}"))).mappings().first()
                    identity_ok = True
                    if commit.idempotency is not None:
                        identity_row = (await session.execute(select(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == execution.tenant_id, idempotency_table.c.scope == commit.idempotency.scope, idempotency_table.c.key_hash == commit.idempotency.key_hash))).mappings().first()
                        identity_value = None if identity_row is None else _decode_payload("idempotency", identity_row["payload"])
                        identity_ok = isinstance(identity_value, IdempotencyRecord) and identity_value.execution_id == execution.execution_id and identity_value.request_digest == commit.idempotency.request_digest and identity_value.status is commit.idempotency.next_status and identity_value.result_digest == commit.idempotency.result_digest and identity_value.error_code == commit.idempotency.error_code
                    operation_ok = True
                    if commit.operation is not None:
                        operation_row = (await session.execute(select(operation_table).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == execution.tenant_id, operation_table.c.record_id == commit.operation.operation_id))).mappings().first()
                        operation_value = None if operation_row is None else _decode_payload("operation_ledger", operation_row["payload"])
                        operation_ok = isinstance(operation_value, OperationLedgerRecord) and operation_value.execution_id == execution.execution_id and operation_value.status is commit.operation.next_status and operation_value.result_ref == commit.operation.result_ref and operation_value.result_digest == commit.operation.result_digest and operation_value.error_code == commit.operation.error_code
                    if isinstance(existing_result, ResultRecord) and existing_result == commit.result and isinstance(current, ExecutionRecord) and current == execution and terminal_event is not None and _decode_payload("execution_events", terminal_event["payload"]) == ExecutionEventRecord(execution.execution_id, execution.tenant_id, commit.expected_event_sequence + 1, commit.terminal_event_type, commit.terminal_event_payload) and identity_ok and operation_ok:
                        return ExecutionTerminalCommitResult(current, existing_result)
                    raise LinktoolsAIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
                if not isinstance(current, ExecutionRecord) or current.snapshot_revision != commit.expected_execution_revision or current.event_sequence != commit.expected_event_sequence or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                identity = None
                if commit.idempotency is not None:
                    identity_row = (await session.execute(select(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == execution.tenant_id, idempotency_table.c.scope == commit.idempotency.scope, idempotency_table.c.key_hash == commit.idempotency.key_hash).with_for_update())).mappings().first()
                    if identity_row is None:
                        raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                    identity = _decode_payload("idempotency", identity_row["payload"])
                    if not isinstance(identity, IdempotencyRecord) or identity.execution_id != execution.execution_id or identity.request_digest != commit.idempotency.request_digest or identity.status is not commit.idempotency.expected_status:
                        raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                else:
                    identity_rows = (await session.execute(select(idempotency_table.c.payload).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == execution.tenant_id))).all()
                    identities = tuple(_decode_payload("idempotency", row[0]) for row in identity_rows)
                    if sum(isinstance(identity, IdempotencyRecord) and identity.execution_id == execution.execution_id for identity in identities) > 1:
                        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                operation = None
                if commit.operation is not None:
                    operation_row = (await session.execute(select(operation_table).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == execution.tenant_id, operation_table.c.record_id == commit.operation.operation_id).with_for_update())).mappings().first()
                    if operation_row is None:
                        raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                    operation = _decode_payload("operation_ledger", operation_row["payload"])
                    if not isinstance(operation, OperationLedgerRecord) or operation.execution_id != execution.execution_id or operation.status is not commit.operation.expected_status:
                        raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                updated = await session.execute(update(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == execution.tenant_id, execution_table.c.record_id == execution.execution_id, execution_table.c.revision == commit.expected_execution_revision, execution_table.c.sequence == commit.expected_event_sequence).values(payload=_encode_payload(execution), revision=execution.snapshot_revision, sequence=execution.event_sequence, status=execution.status.value, updated_at=_record_time(execution)))
                if updated.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                await session.execute(insert(result_table).values(namespace_key=self.namespace_key, tenant_id=execution.tenant_id, record_id=execution.execution_id, sequence=0, revision=0, status=commit.result.status.value, payload=_encode_payload(commit.result), created_at=_record_time(commit.result), updated_at=_record_time(commit.result)))
                terminal_event = ExecutionEventRecord(execution.execution_id, execution.tenant_id, commit.expected_event_sequence + 1, commit.terminal_event_type, commit.terminal_event_payload)
                await session.execute(insert(event_table).values(namespace_key=self.namespace_key, tenant_id=execution.tenant_id, record_id=f"{execution.execution_id}:{terminal_event.sequence}", sequence=terminal_event.sequence, revision=0, status=terminal_event.event_type.value, payload=_encode_payload(terminal_event), created_at=_record_time(commit.result), updated_at=_record_time(commit.result)))
                if commit.idempotency is not None and identity is not None:
                    updated_identity = replace(identity, status=commit.idempotency.next_status, result_digest=commit.idempotency.result_digest, error_code=commit.idempotency.error_code, updated_at=_record_time(execution))
                    outcome = await session.execute(update(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == execution.tenant_id, idempotency_table.c.scope == commit.idempotency.scope, idempotency_table.c.key_hash == commit.idempotency.key_hash, idempotency_table.c.status == commit.idempotency.expected_status.value).values(payload=_encode_payload(updated_identity), revision=idempotency_table.c.revision + 1, status=updated_identity.status.value, updated_at=updated_identity.updated_at))
                    if outcome.rowcount != 1:
                        raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                if commit.operation is not None and operation is not None:
                    updated_operation = replace(operation, status=commit.operation.next_status, result_ref=commit.operation.result_ref, result_digest=commit.operation.result_digest, error_code=commit.operation.error_code, updated_at=_record_time(execution))
                    outcome = await session.execute(update(operation_table).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == execution.tenant_id, operation_table.c.record_id == commit.operation.operation_id, operation_table.c.status == commit.operation.expected_status.value).values(payload=_encode_payload(updated_operation), status=updated_operation.status.value, updated_at=updated_operation.updated_at))
                    if outcome.rowcount != 1:
                        raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                if commit.session_head is not None:
                    expected_head = commit.session_head.expected_head_execution_id
                    head_match = session_table.c.head_execution_id.is_(None) if expected_head is None else session_table.c.head_execution_id == expected_head
                    session_row = (await session.execute(select(session_table).where(session_table.c.namespace_key == self.namespace_key, session_table.c.tenant_id == execution.tenant_id, session_table.c.session_id == commit.session_head.session_id, session_table.c.status == SessionStatus.OPEN.value, head_match).with_for_update())).mappings().first()
                    if session_row is not None:
                        current_session = _decode_payload("sessions", session_row["payload"])
                        if not isinstance(current_session, SessionRecord):
                            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        updated_session = replace(current_session, revision=current_session.revision + 1, head_execution_id=commit.session_head.next_head_execution_id, updated_at=_record_time(execution))
                        await session.execute(update(session_table).where(session_table.c.namespace_key == self.namespace_key, session_table.c.tenant_id == execution.tenant_id, session_table.c.session_id == commit.session_head.session_id, session_table.c.revision == current_session.revision).values(payload=_encode_payload(updated_session), head_execution_id=updated_session.head_execution_id, revision=updated_session.revision, updated_at=updated_session.updated_at))
        return ExecutionTerminalCommitResult(execution, commit.result)

    async def _append_event(self, execution_id: str, *, tenant_id: str, expected_sequence: int, event_type: "ExecutionEventType | None" = None, payload: JsonValue) -> "ExecutionEventRecord":
        if self._table_name != "execution_events" or event_type is None:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        from sqlalchemy import insert, select, update
        execution_table = self._owner.tables["executions"]
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == tenant_id, execution_table.c.record_id == execution_id))).mappings().first()
                if row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                current = _decode_payload("executions", row["payload"])
                field = "event_sequence"
                current_sequence = current.event_sequence
                if current_sequence != expected_sequence:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                next_value = expected_sequence + 1
                updated_execution = replace(current, **{field: next_value, "snapshot_revision": current.snapshot_revision + 1, "updated_at": datetime.now(timezone.utc)})
                outcome = await session.execute(update(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == tenant_id, execution_table.c.record_id == execution_id, execution_table.c.revision == current.snapshot_revision).values(payload=_encode_payload(updated_execution), revision=updated_execution.snapshot_revision, updated_at=updated_execution.updated_at))
                if outcome.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                item = ExecutionEventRecord(execution_id, tenant_id, next_value, event_type, payload)
                await session.execute(insert(self._table).values(namespace_key=self.namespace_key, tenant_id=tenant_id, record_id=f"{execution_id}:{next_value}", sequence=next_value, revision=0, status="", payload=_encode_payload(item), created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)))
                return item

    async def _list_event(self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int) -> Page[object]:
        values = tuple(item for item in await self._list_records(tenant_id=tenant_id) if item.execution_id == execution_id and item.sequence > after_sequence)
        page = values[:limit]
        return Page(page, str(page[-1].sequence) if len(values) > len(page) else None)

    async def get_idempotency(self, scope: str, key_hash: str, *, tenant_id: str) -> IdempotencyRecord | None:
        return await self._get(f"{scope}:{key_hash}", tenant_id=tenant_id)

    async def compare_idempotency(self, scope: str, key_hash: str, *, tenant_id: str, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord:
        from sqlalchemy import select, update
        table = self._owner.tables["idempotency"]
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(table).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.scope == scope, table.c.key_hash == key_hash))).mappings().first()
                if row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                current = _decode_payload("idempotency", row["payload"])
                if not isinstance(current, IdempotencyRecord) or current.status is not expected_status:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                if next_record.tenant_id != tenant_id or next_record.scope != scope or next_record.key_hash != key_hash:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                outcome = await session.execute(update(table).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.scope == scope, table.c.key_hash == key_hash, table.c.revision == row["revision"], table.c.status == expected_status.value).values(payload=_encode_payload(next_record), revision=int(row["revision"]) + 1, status=next_record.status.value, updated_at=next_record.updated_at))
                if outcome.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
        return next_record

    async def append(self, record: "OperationLedgerInput | str", *, tenant_id: "str | None" = None, expected_sequence: "int | None" = None, event_type: "ExecutionEventType | None" = None, payload: "JsonValue | None" = None) -> "OperationLedgerRecord | ExecutionEventRecord | object":
        if self._table_name == "execution_events":
            if not isinstance(record, str) or tenant_id is None or expected_sequence is None or payload is None:
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._append_event(record, tenant_id=tenant_id, expected_sequence=expected_sequence, event_type=event_type, payload=payload)
        if not isinstance(record, OperationLedgerInput):
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        for attempt in range(32):
            try:
                return await self._append_operation(record)
            except Exception as error:
                retryable = _is_retryable_transaction(error) or (isinstance(error, LinktoolsAIError) and error.code is ErrorCode.STORAGE_CONFLICT)
                if not retryable or attempt == 31:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT) from error
                await asyncio.sleep(0.01 * (attempt + 1))

    async def _append_operation(self, record: OperationLedgerInput) -> OperationLedgerRecord:
        from sqlalchemy import insert, select, update
        counter_table = self._owner.tables["operation_counters"]
        ledger_table = self._owner.tables["operation_ledger"]
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(ledger_table).where(ledger_table.c.namespace_key == self.namespace_key, ledger_table.c.tenant_id == record.tenant_id, ledger_table.c.record_id == record.operation_id))).mappings().first()
                if row is not None:
                    existing = _decode_payload("operation_ledger", row["payload"])
                    if _operation_identity(existing) != _operation_identity(record):
                        raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                    return existing
                counter = (await session.execute(select(counter_table).where(counter_table.c.namespace_key == self.namespace_key, counter_table.c.tenant_id == record.tenant_id, counter_table.c.resource_kind == record.resource_kind.value, counter_table.c.resource_id == record.resource_id).with_for_update())).mappings().first()
                sequence = 1 if counter is None else int(counter["revision"]) + 1
                counter_values = {"namespace_key": self.namespace_key, "tenant_id": record.tenant_id, "record_id": f"{record.resource_kind.value}:{record.resource_id}", "resource_kind": record.resource_kind.value, "resource_id": record.resource_id, "sequence": sequence, "revision": sequence, "status": "", "payload": _encode_payload(sequence), "created_at": record.created_at, "updated_at": record.updated_at}
                if counter is None:
                    await session.execute(insert(counter_table).values(counter_values))
                else:
                    result = await session.execute(update(counter_table).where(counter_table.c.id == counter["id"], counter_table.c.revision == sequence - 1).values(sequence=sequence, revision=sequence, payload=_encode_payload(sequence), updated_at=record.updated_at))
                    if result.rowcount != 1:
                        raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
                created = OperationLedgerRecord(record.operation_id, record.tenant_id, record.resource_kind, record.resource_id, record.execution_id, record.kind, record.status, record.request_digest, record.result_ref, record.result_digest, record.error_code, record.compactable, sequence, record.created_at, record.updated_at)
                await session.execute(insert(ledger_table).values(namespace_key=self.namespace_key, tenant_id=record.tenant_id, record_id=record.operation_id, resource_kind=record.resource_kind.value, resource_id=record.resource_id, sequence=sequence, revision=0, status=record.status.value, payload=_encode_payload(created), created_at=record.created_at, updated_at=record.updated_at))
                return created

    async def list_pending(self, resource_kind: "ResourceKind | str", resource_id: "str | None" = None, *, tenant_id: str, limit: int = 100) -> "tuple[OperationLedgerRecord, ...] | tuple[object, ...]":
        if self._table_name in {"approvals", "external_results"}:
            return await self._list_external(str(resource_kind), tenant_id=tenant_id)
        if resource_id is None or not isinstance(resource_kind, ResourceKind):
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        values = await self._list_records(tenant_id=tenant_id, table=self._owner.tables["operation_ledger"])
        return tuple(item for item in values if item.resource_kind is resource_kind and item.resource_id == resource_id and item.status in {OperationStatus.PENDING, OperationStatus.RUNNING})[:limit]

    async def compact_terminal(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, through_sequence: int) -> str:
        return hashlib.sha256(f"{self.namespace}:{tenant_id}:{resource_kind.value}:{resource_id}:{through_sequence}".encode()).hexdigest()

    async def reserve(self, record: "IdempotencyRecord | ToolOperationRecord") -> "IdempotencyRecord | ToolOperationRecord":
        if self._table_name == "idempotency":
            if not isinstance(record, IdempotencyRecord):
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            try:
                return await self._insert(record, record_id=f"{record.scope}:{record.key_hash}", tenant_id=record.tenant_id, status=record.status.value, table=self._owner.tables["idempotency"])
            except Exception as error:
                if not _is_integrity(error):
                    raise
                existing = await self.get_idempotency(record.scope, record.key_hash, tenant_id=record.tenant_id)
                if existing is not None and existing.request_digest == record.request_digest and existing.execution_id == record.execution_id:
                    return existing
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
        if not isinstance(record, ToolOperationRecord):
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        if re.fullmatch(r"[0-9a-f]{64}", record.idempotency_key_hash) is None:
            raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        existing = await self.get_operation(record.operation_id, tenant_id=record.tenant_id)
        if existing is not None:
            if _tool_identity(existing) != _tool_identity(record):
                raise LinktoolsAIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            return existing
        try:
            return await self.create(record)
        except LinktoolsAIError as error:
            if error.code is ErrorCode.STORAGE_CONFLICT:
                existing = await self.get_operation(record.operation_id, tenant_id=record.tenant_id)
                if existing is not None and _tool_identity(existing) == _tool_identity(record):
                    return existing
                raise LinktoolsAIError(ErrorCode.TOOL_OPERATION_CONFLICT) from error
            raise

    async def get_operation(self, operation_id: str, *, tenant_id: str) -> ToolOperationRecord | None:
        return await self._get(operation_id, tenant_id=tenant_id)

    async def claim(self, operation_id: str, task_id: "str | None" = None, *, tenant_id: str, owner: str, lease_seconds: int) -> "ToolOperationRecord | TaskLease":
        if self._table_name == "task_graphs":
            if task_id is None:
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._claim_task(operation_id, task_id, tenant_id=tenant_id, owner=owner, lease_seconds=lease_seconds)
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        now = datetime.now(timezone.utc)
        if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
            raise LinktoolsAIError(ErrorCode.TASK_TERMINAL_CONFLICT)
        if current.status is ToolOperationStatus.CLAIMED and current.lease_expires_at is not None and current.lease_expires_at > now:
            raise LinktoolsAIError(ErrorCode.TASK_OWNER_CONFLICT)
        if current.status is ToolOperationStatus.CLAIMED and not current.replay_safe:
            unknown = replace(current, status=ToolOperationStatus.EFFECT_UNKNOWN, owner=None, lease_expires_at=None, updated_at=now)
            await self._tool_update(unknown, tenant_id=tenant_id, expected_status=ToolOperationStatus.CLAIMED, expected_owner=current.owner, expected_fence=current.fence, expected_lease_before=now)
            raise LinktoolsAIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        updated = replace(current, status=ToolOperationStatus.CLAIMED, owner=owner, fence=current.fence + 1, lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now)
        return await self._tool_update(updated, tenant_id=tenant_id, expected_status=current.status, expected_owner=current.owner, expected_fence=current.fence if current.status is ToolOperationStatus.CLAIMED else None, expected_lease_before=now if current.status is ToolOperationStatus.CLAIMED else None)

    async def renew(self, operation_id: "str | TaskLease", *, tenant_id: str, owner: "str | None" = None, fence: "int | None" = None, lease_seconds: "int | None" = None) -> "ToolOperationRecord | TaskLease":
        if self._table_name == "task_graphs":
            if not isinstance(operation_id, TaskLease):
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._renew_task(operation_id, tenant_id=tenant_id)
        if owner is None or fence is None or lease_seconds is None or not isinstance(operation_id, str):
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self._owned(operation_id, tenant_id, owner, fence)
        now = datetime.now(timezone.utc)
        return await self._tool_update(replace(current, lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now), tenant_id=tenant_id, expected_status=ToolOperationStatus.CLAIMED, expected_owner=owner, expected_fence=fence, expected_lease_after=now)

    async def complete(self, operation_id: "str | TaskLease", *, tenant_id: str, owner: "str | None" = None, fence: "int | None" = None, result_ref: "str | None" = None, result_digest: str) -> "ToolOperationRecord | TaskTerminalRecord":
        if self._table_name == "task_graphs":
            if not isinstance(operation_id, TaskLease):
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._complete_task(operation_id, tenant_id=tenant_id, result_digest=result_digest)
        if not isinstance(operation_id, str) or owner is None or fence is None:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status is ToolOperationStatus.COMPLETED and current.result_digest == result_digest:
            return current
        if current.status is ToolOperationStatus.COMPLETED:
            raise LinktoolsAIError(ErrorCode.TOOL_RESULT_CONFLICT)
        current = await self._owned(operation_id, tenant_id, owner, fence)
        now = datetime.now(timezone.utc)
        return await self._tool_update(replace(current, status=ToolOperationStatus.COMPLETED, result_ref=result_ref, result_digest=result_digest, lease_expires_at=None, updated_at=now), tenant_id=tenant_id, expected_status=ToolOperationStatus.CLAIMED, expected_owner=owner, expected_fence=fence, expected_lease_after=now)

    async def fail(self, operation_id: "str | TaskLease", *, tenant_id: str, owner: "str | None" = None, fence: "int | None" = None, error_code: str, error_digest: "str | None" = None) -> "ToolOperationRecord | TaskTerminalRecord":
        if self._table_name == "task_graphs":
            if not isinstance(operation_id, TaskLease) or error_digest is None:
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._fail_task(operation_id, tenant_id=tenant_id, error_code=error_code, error_digest=error_digest)
        if not isinstance(operation_id, str) or owner is None or fence is None:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status is ToolOperationStatus.FAILED and current.error_code == error_code:
            return current
        if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
            raise LinktoolsAIError(ErrorCode.TASK_TERMINAL_CONFLICT)
        current = await self._owned(operation_id, tenant_id, owner, fence)
        now = datetime.now(timezone.utc)
        return await self._tool_update(replace(current, status=ToolOperationStatus.FAILED, error_code=error_code, lease_expires_at=None, updated_at=now), tenant_id=tenant_id, expected_status=ToolOperationStatus.CLAIMED, expected_owner=owner, expected_fence=fence, expected_lease_after=now)

    async def _tool_update(self, record: ToolOperationRecord, *, tenant_id: str, expected_status: ToolOperationStatus, expected_owner: "str | None", expected_fence: "int | None", expected_lease_after: "datetime | None" = None, expected_lease_before: "datetime | None" = None) -> ToolOperationRecord:
        from sqlalchemy import select, update
        table = self._owner.tables["tool_operations"]
        async with self._owner.database.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(table).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record.operation_id))).mappings().first()
                if row is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                predicate = [table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record.operation_id, table.c.revision == row["revision"], table.c.status == expected_status.value]
                predicate.append(table.c.owner.is_(None) if expected_owner is None else table.c.owner == expected_owner)
                if expected_fence is not None:
                    predicate.append(table.c.fence == expected_fence)
                if expected_lease_after is not None:
                    predicate.append(table.c.lease_expires_at > expected_lease_after)
                if expected_lease_before is not None:
                    predicate.append(table.c.lease_expires_at <= expected_lease_before)
                current = _decode_payload("tool_operations", row["payload"])
                if not isinstance(current, ToolOperationRecord) or (expected_owner is not None and current.owner != expected_owner) or (expected_fence is not None and current.fence != expected_fence) or (expected_lease_after is not None and (current.lease_expires_at is None or current.lease_expires_at <= expected_lease_after)) or (expected_lease_before is not None and (current.lease_expires_at is None or current.lease_expires_at > expected_lease_before)):
                    raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
                outcome = await session.execute(update(table).where(*predicate).values(payload=_encode_payload(record), revision=int(row["revision"]) + 1, status=record.status.value, owner=record.owner, tool_call_id=record.tool_call_id, run_id=record.run_id, fence=record.fence, lease_expires_at=record.lease_expires_at, updated_at=record.updated_at))
                if outcome.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        return record

    async def _owned(self, operation_id: str, tenant_id: str, owner: str, fence: int) -> ToolOperationRecord:
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None or current.owner != owner or current.fence != fence or current.lease_expires_at is None or current.lease_expires_at <= datetime.now(timezone.utc):
            raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        return current

    async def put_bytes(self, *, tenant_id: str, data: bytes, expected_digest: "str | None" = None) -> BlobRef:
        if len(data) > _MAX_SQL_INLINE_BLOB_BYTES:
            raise ValueError("put_stream is required for blobs larger than 4 MiB")
        digest = hashlib.sha256(data).hexdigest()
        if expected_digest is not None and expected_digest != digest:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async def chunks() -> AsyncIterator[bytes]:
            for offset in range(0, len(data), _MAX_SQL_BLOB_CHUNK):
                yield data[offset:offset + _MAX_SQL_BLOB_CHUNK]

        return await self.put_stream(tenant_id=tenant_id, chunks=chunks(), expected_size=len(data), expected_digest=digest)

    async def put_stream(self, *, tenant_id: str, chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> BlobRef:
        if expected_size < 0 or expected_size > _MAX_SQL_BLOB_BYTES or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        existing = await self._get(expected_digest, tenant_id=tenant_id, table=self._owner.tables["blobs"])
        if existing is not None:
            if not isinstance(existing, BlobRef) or existing.size != expected_size:
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        from sqlalchemy import insert, select
        blob_table = self._owner.tables["blobs"]
        chunk_table = self._owner.tables["blob_chunks"]
        digest = hashlib.sha256()
        size = 0
        index = 0
        try:
            async with self._owner.database.session_factory() as session:
                async with session.begin():
                    manifest = (await session.execute(select(blob_table).where(blob_table.c.namespace_key == self.namespace_key, blob_table.c.tenant_id == tenant_id, blob_table.c.record_id == expected_digest))).mappings().first()
                    if manifest is not None:
                        existing = _decode_payload("blobs", manifest["payload"])
                        if not isinstance(existing, BlobRef) or existing.size != expected_size:
                            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        return existing
                    async for chunk in chunks:
                        chunk = bytes(chunk)
                        if not 1 <= len(chunk) <= _MAX_SQL_BLOB_CHUNK or size + len(chunk) > expected_size:
                            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
                        digest.update(chunk)
                        size += len(chunk)
                        now = datetime.now(timezone.utc)
                        await session.execute(insert(chunk_table).values(namespace_key=self.namespace_key, tenant_id=tenant_id, record_id=f"{expected_digest}:{index}", digest=expected_digest, chunk_index=index, content=chunk, sequence=index, revision=0, status="", payload=_encode_payload(chunk), created_at=now, updated_at=now))
                        index += 1
                    if size != expected_size or digest.hexdigest() != expected_digest:
                        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    ref = BlobRef(tenant_id, expected_digest, size, f"sql:{self.namespace}:{expected_digest}")
                    now = datetime.now(timezone.utc)
                    await session.execute(insert(blob_table).values(namespace_key=self.namespace_key, tenant_id=tenant_id, record_id=expected_digest, digest=expected_digest, size=size, sequence=0, revision=0, status="COMPLETED", payload=_encode_payload(ref), created_at=now, updated_at=now))
        except Exception as error:
            if not _is_integrity(error):
                raise
            existing = await self._get(expected_digest, tenant_id=tenant_id, table=blob_table)
            if isinstance(existing, BlobRef) and existing.size == expected_size:
                return existing
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        _logger.info("SQL blob committed: namespace=%s tenant=%s digest=%s size=%s", self.namespace, tenant_id, expected_digest, expected_size)
        return ref

    async def stat(self, ref: BlobRef, *, tenant_id: str) -> BlobRef | None:
        value = await self._get(ref.digest, tenant_id=tenant_id, table=self._owner.tables["blobs"])
        if value is None:
            return None
        if not isinstance(value, BlobRef):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    def open(self, ref: BlobRef, *, tenant_id: str) -> AsyncIterator[bytes]:
        return self._open_blob(ref, tenant_id)

    async def _open_blob(self, ref: BlobRef, tenant_id: str) -> AsyncIterator[bytes]:
        from sqlalchemy import select
        manifest = await self._get(ref.digest, tenant_id=tenant_id, table=self._owner.tables["blobs"])
        if not isinstance(manifest, BlobRef):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with self._owner.database.session_factory() as session:
            rows = (await session.execute(select(self._owner.tables["blob_chunks"].c.payload).where(self._owner.tables["blob_chunks"].c.namespace_key == self.namespace_key, self._owner.tables["blob_chunks"].c.tenant_id == tenant_id, self._owner.tables["blob_chunks"].c.record_id.like(f"{ref.digest}:%")).order_by(self._owner.tables["blob_chunks"].c.sequence))).scalars()
            total = 0
            digest = hashlib.sha256()
            for payload in rows:
                chunk = _decode_payload("blob_chunks", payload)
                if not isinstance(chunk, bytes):
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                total += len(chunk)
                digest.update(chunk)
                yield chunk
        if total != manifest.size or digest.hexdigest() != manifest.digest:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


class _SqlRuntimeOwner:
    def __init__(self, database: StorageDatabase, tables: SqlRuntimeTables, *, backend: RuntimeBackend, namespace: str, atomic_domain_id: str) -> None:
        self.database = database
        self.tables = tables.tables
        self.backend = backend
        self.namespace = namespace
        self.namespace_key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        self.atomic_domain_id = atomic_domain_id

    def table_name(self, table: "Table") -> str:
        for name, value in self.tables.items():
            if value is table:
                return name
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _encode_payload(value: object) -> JsonValue:
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if is_dataclass(value):
        return {"type": type(value).__name__, "value": json.loads(canonical_json_bytes(asdict(value)))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _decode_payload(table_name: str, value: object) -> object:
    if not isinstance(value, dict) or "type" not in value:
        return value
    kind = value["type"]
    raw = value.get("value")
    if kind == "bytes":
        if not isinstance(raw, str):
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return base64.b64decode(raw.encode("ascii"))
    if not isinstance(raw, dict):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    decoder = {
        "SessionRecord": _session_record,
        "ExecutionRecord": _execution_record,
        "ResultRecord": _result_record,
        "IdempotencyRecord": _idempotency_record,
        "ExecutionEventRecord": _event_record,
        "TaskGraphView": _task_graph,
        "TaskNodeView": _task_node,
        "EvaluationRecord": _evaluation,
        "MemoryRecord": _memory,
        "ArtifactRecord": _artifact,
        "ApprovalRecord": _approval,
        "ExternalResultRecord": _external,
        "OperationLedgerRecord": _operation,
        "ToolOperationRecord": _tool,
        "BlobRef": _blob_ref,
    }.get(str(kind))
    if decoder is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR, f"unknown SQL payload type for {table_name}")
    return decoder(raw)


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _enum(enum_type: "type[object]", value: object) -> object:
    return enum_type(str(value))


def _session_record(value: "dict[str, JsonValue]") -> SessionRecord:
    return SessionRecord(str(value["session_id"]), str(value["tenant_id"]), str(value["owner_principal_id"]), str(value["binding_digest"]), _enum(SessionStatus, value["status"]), int(value["revision"]), int(value["resource_generation"]), None if value.get("cwd") is None else str(value["cwd"]), value.get("metadata", {}), _utc(value["created_at"]), _utc(value["updated_at"]), None if value.get("closed_at") is None else _utc(value["closed_at"]), _enum(ExecutionProfile, value.get("profile", ExecutionProfile.LOCAL_CODING.value)), None if value.get("head_execution_id") is None else str(value["head_execution_id"]))


def _execution_record(value: "dict[str, JsonValue]") -> ExecutionRecord:
    return ExecutionRecord(str(value["execution_id"]), str(value["tenant_id"]), None if value.get("session_id") is None else str(value["session_id"]), _enum(ExecutionProfile, value["profile"]), str(value["binding_digest"]), None if value.get("parent_execution_id") is None else str(value["parent_execution_id"]), str(value["root_execution_id"]), _enum(ExecutionStatus, value["status"]), int(value["snapshot_revision"]), int(value["event_sequence"]), None if value.get("result_ref") is None else str(value["result_ref"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), value.get("safe_error_details", {}), _utc(value["created_at"]), _utc(value["updated_at"]), None if value.get("source_execution_id") is None else str(value["source_execution_id"]), None if value.get("base_execution_id") is None else str(value["base_execution_id"]), _enum(ExecutionLineageKind, value.get("lineage_kind", "RUN")), int(value.get("agent_run_sequence", 0)))


def _result_record(value: "dict[str, JsonValue]") -> ResultRecord:
    return ResultRecord(str(value["execution_id"]), str(value["tenant_id"]), _enum(ExecutionStatus, value["status"]), str(value["output_schema_id"]), int(value["output_schema_revision"]), str(value["output_schema_fingerprint"]), None if value.get("payload_ref") is None else str(value["payload_ref"]), None if value.get("payload_digest") is None else str(value["payload_digest"]), _enum(StopReason, value["stop_reason"]), int(value["input_tokens"]), int(value["output_tokens"]), int(value["total_cost_micros"]), _utc(value["created_at"]))


def _idempotency_record(value: "dict[str, JsonValue]") -> IdempotencyRecord:
    return IdempotencyRecord(str(value["tenant_id"]), str(value["scope"]), str(value["key_hash"]), str(value["request_digest"]), str(value["execution_id"]), _enum(IdempotencyStatus, value["status"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), _utc(value["created_at"]), _utc(value["updated_at"]))


def _event_record(value: "dict[str, JsonValue]") -> ExecutionEventRecord:
    return ExecutionEventRecord(str(value["execution_id"]), str(value["tenant_id"]), int(value["sequence"]), _enum(ExecutionEventType, value["event_type"]), value.get("payload", {}))


def _task_graph(value: "dict[str, JsonValue]") -> TaskGraphView:
    nodes = tuple(TaskNode(str(node["task_id"]), tuple(node.get("dependencies", [])), None if node.get("binding_digest") is None else str(node["binding_digest"]), int(node.get("budget_cost", 1))) for node in value.get("nodes", []))
    return TaskGraphView(str(value["graph_id"]), _enum(TaskStatus, value["status"]), nodes)


def _task_node(value: "dict[str, JsonValue]") -> TaskNodeView:
    return TaskNodeView(str(value["graph_id"]), str(value["task_id"]), tuple(value.get("dependencies", [])), _enum(TaskStatus, value["status"]), None if value.get("owner") is None else str(value["owner"]), int(value["fence"]), None if value.get("lease_expires_at") is None else _utc(value["lease_expires_at"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), None if value.get("error_digest") is None else str(value["error_digest"]))


def _evaluation(value: "dict[str, JsonValue]") -> EvaluationRecord:
    return EvaluationRecord(str(value["evaluation_id"]), str(value["tenant_id"]), str(value["execution_id"]), str(value["dataset_id"]), int(value["dataset_revision"]), str(value["evaluator_id"]), int(value["evaluator_revision"]), str(value["binding_digest"]), str(value["output_schema_fingerprint"]), None if value.get("artifact_digest") is None else str(value["artifact_digest"]), _enum(EvaluationStatus, value["status"]), int(value["revision"]), value.get("metrics", {}), _utc(value["created_at"]), _utc(value["updated_at"]))


def _memory(value: "dict[str, JsonValue]") -> MemoryRecord:
    return MemoryRecord(str(value["memory_id"]), str(value["tenant_id"]), str(value["owner_id"]), str(value["kind"]), str(value["content_ref"]), str(value["content_digest"]), value.get("metadata", {}), int(value["revision"]), _utc(value["created_at"]), _utc(value["updated_at"]))


def _artifact(value: "dict[str, JsonValue]") -> ArtifactRecord:
    return ArtifactRecord(str(value["artifact_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["producer"]), str(value["media_type"]), int(value["size"]), str(value["digest"]), str(value["blob_ref"]), _utc(value["created_at"]))


def _approval(value: "dict[str, JsonValue]") -> ApprovalRecord:
    return ApprovalRecord(str(value["approval_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["operation_id"]), _enum(ApprovalStatus, value["status"]), None if value.get("decision_id") is None else str(value["decision_id"]), None if value.get("decision") is None else _enum(ApprovalDecision, value["decision"]), None if value.get("decided_by") is None else str(value["decided_by"]), None if value.get("decision_digest") is None else str(value["decision_digest"]), _utc(value["created_at"]), None if value.get("decided_at") is None else _utc(value["decided_at"]))


def _external(value: "dict[str, JsonValue]") -> ExternalResultRecord:
    return ExternalResultRecord(str(value["call_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["operation_id"]), _enum(ExternalCallStatus, value["status"]), None if value.get("result_id") is None else str(value["result_id"]), None if value.get("payload_ref") is None else str(value["payload_ref"]), None if value.get("payload_digest") is None else str(value["payload_digest"]), _utc(value["created_at"]), None if value.get("supplied_at") is None else _utc(value["supplied_at"]))


def _operation(value: "dict[str, JsonValue]") -> OperationLedgerRecord:
    return OperationLedgerRecord(str(value["operation_id"]), str(value["tenant_id"]), _enum(ResourceKind, value["resource_kind"]), str(value["resource_id"]), None if value.get("execution_id") is None else str(value["execution_id"]), _enum(OperationKind, value["kind"]), _enum(OperationStatus, value["status"]), str(value["request_digest"]), None if value.get("result_ref") is None else str(value["result_ref"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), bool(value["compactable"]), int(value["sequence"]), _utc(value["created_at"]), _utc(value["updated_at"]))


def _tool(value: "dict[str, JsonValue]") -> ToolOperationRecord:
    return ToolOperationRecord(str(value["operation_id"]), str(value["tenant_id"]), str(value["run_id"]), str(value["tool_call_id"]), str(value["idempotency_key_hash"]), str(value["tool_name"]), str(value["arguments_hash"]), str(value["binding_fingerprint"]), bool(value["replay_safe"]), _enum(ToolOperationStatus, value["status"]), None if value.get("owner") is None else str(value["owner"]), int(value["fence"]), None if value.get("lease_expires_at") is None else _utc(value["lease_expires_at"]), None if value.get("result_ref") is None else str(value["result_ref"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), _utc(value["created_at"]), _utc(value["updated_at"]))


def _blob_ref(value: "dict[str, JsonValue]") -> BlobRef:
    return BlobRef(str(value["tenant_id"]), str(value["digest"]), int(value["size"]), str(value["locator"]))


async def open_sql_runtime(database: StorageDatabase, *, backend: RuntimeBackend, namespace: str, deployment_id: str, tables: SqlRuntimeTables) -> RuntimePersistence:
    if backend not in {RuntimeBackend.SQLITE, RuntimeBackend.MYSQL, RuntimeBackend.POSTGRESQL}:
        raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
    target = database_target(database)
    atomic_domain_id = hashlib.sha256(f"{backend.value}{target}{namespace}{deployment_id}{database.schema_manifest_digest}".encode("utf-8")).hexdigest()
    owner = _SqlRuntimeOwner(database, tables, backend=backend, namespace=namespace, atomic_domain_id=atomic_domain_id)
    components = tuple(_SqlRuntimeRepository(owner, name) for name in ("sessions", "executions", "results", "idempotency", "execution_events", "task_graphs", "evaluations", "memories", "artifacts", "approvals", "external_results", "operation_ledger", "tool_operations", "blobs"))
    return RuntimePersistence(
        mode=RuntimePersistenceMode.SQL,
        backend=backend,
        namespace=namespace,
        sessions=components[0],
        executions=components[1],
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
    )


def database_target(database: StorageDatabase) -> str:
    return database.target_identity


def _record_id(record: object) -> str:
    if isinstance(record, IdempotencyRecord):
        return f"{record.scope}:{record.key_hash}"
    if isinstance(record, (SessionRecord, ExecutionRecord, ResultRecord)):
        return record.session_id if isinstance(record, SessionRecord) else record.execution_id
    if isinstance(record, (EvaluationRecord, MemoryRecord, ArtifactRecord, ApprovalRecord, ExternalResultRecord, OperationLedgerRecord, ToolOperationRecord)):
        if isinstance(record, EvaluationRecord):
            return record.evaluation_id
        if isinstance(record, MemoryRecord):
            return record.memory_id
        if isinstance(record, ArtifactRecord):
            return record.artifact_id
        if isinstance(record, ApprovalRecord):
            return record.approval_id
        if isinstance(record, ExternalResultRecord):
            return record.call_id
        return record.operation_id
    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _record_time(record: object) -> datetime:
    value = record.updated_at if isinstance(record, (SessionRecord, ExecutionRecord, IdempotencyRecord, EvaluationRecord, MemoryRecord, OperationLedgerRecord, ToolOperationRecord)) else record.created_at if isinstance(record, (ResultRecord, ArtifactRecord, ApprovalRecord, ExternalResultRecord)) else datetime.now(timezone.utc)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _revision(record: object) -> int:
    if isinstance(record, ExecutionRecord):
        return record.snapshot_revision
    if isinstance(record, (SessionRecord, EvaluationRecord, MemoryRecord)):
        return record.revision
    return 0


def _status(record: object) -> str:
    value = record.status if isinstance(record, (SessionRecord, ExecutionRecord, ResultRecord, IdempotencyRecord, EvaluationRecord, ApprovalRecord, ExternalResultRecord, OperationLedgerRecord, ToolOperationRecord)) else ""
    return value.value if isinstance(value, StrEnum) else str(value)


def _operation_identity(record: object) -> tuple[object, ...]:
    return (
        record.tenant_id, record.resource_kind, record.resource_id, record.execution_id,
        record.kind, record.request_digest, record.result_ref, record.result_digest,
        record.error_code, record.compactable,
    ) if isinstance(record, (OperationLedgerInput, OperationLedgerRecord)) else ()


def _tool_identity(record: ToolOperationRecord) -> tuple[object, ...]:
    return (
        record.tenant_id, record.run_id, record.tool_call_id, record.idempotency_key_hash,
        record.tool_name, record.arguments_hash, record.binding_fingerprint, record.replay_safe,
    )


def _resource_kind(table_name: str) -> ResourceKind:
    return {"sessions": ResourceKind.SESSION, "executions": ResourceKind.EXECUTION, "evaluations": ResourceKind.EVALUATION, "artifacts": ResourceKind.ARTIFACT, "approvals": ResourceKind.APPROVAL, "task_graphs": ResourceKind.TASK_GRAPH}.get(table_name, ResourceKind.EXECUTION)


def _is_integrity(error: BaseException) -> bool:
    try:
        from sqlalchemy.exc import IntegrityError
        return isinstance(error, IntegrityError)
    except ModuleNotFoundError:
        return False


def _is_retryable_transaction(error: BaseException) -> bool:
    if _is_integrity(error):
        return True
    try:
        from sqlalchemy.exc import DBAPIError
    except ModuleNotFoundError:
        return False
    if not isinstance(error, DBAPIError):
        return False
    original = error.orig
    message = str(original).lower()
    return any(token in message for token in ("40001", "40p01", "1205", "1213", "deadlock", "database is locked"))


__all__ = ["open_sql_runtime"]
