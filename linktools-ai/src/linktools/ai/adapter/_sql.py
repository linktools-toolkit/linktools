#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL RuntimePersistence owner shared by SQLite, MySQL and PostgreSQL."""

import asyncio
import base64
import hashlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import TYPE_CHECKING

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
    canonical_json_bytes,
    canonical_sha256,
)
from ..errors import AIError, ErrorCode
from ..runtime import (
    ApprovalRecord,
    ArtifactRecord,
    BlobRef,
    BlobStore,
    EvaluationRecord,
    ExecutionCancelRequestCommit,
    ExecutionEventRecord,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionStartUnknownCommit,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExternalResultRecord,
    IdempotencyRecord,
    MemoryRecord,
    OperationLedgerInput,
    OperationLedgerRecord,
    ResultRecord,
    RuntimeBackend,
    RuntimePersistence,
    RuntimePersistenceMode,
    RuntimeRepository,
    SessionRecord,
    TaskLease,
    TaskNodeView,
    ToolOperationRecord,
    ToolStateStore,
)
from ..storage import StorageDatabase, resolve_dialect
from ..task import TaskGraph, TaskGraphView, TaskNode, TaskTerminalRecord
from ._schema import SqlRuntimeTables

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from sqlalchemy.sql.elements import ColumnElement


_MAX_SQL_BLOB_BYTES = 2 * 1024 * 1024 * 1024
_MAX_SQL_BLOB_CHUNK = 1024 * 1024
_MAX_SQL_INLINE_BLOB_BYTES = 4 * 1024 * 1024
_SQL_CHUNK_INSERT_ROWS = 64
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
        async with self._owner.session_factory() as session:
            payload = await session.scalar(
                select(target.c.payload).where(
                    target.c.namespace_key == self.namespace_key,
                    target.c.tenant_id == tenant_id,
                    target.c.record_id == record_id,
                )
            )
        return None if payload is None else _decode_payload(self._owner.table_name(target), payload)

    async def _exists(self, record_id: str, *, tenant_id: str) -> bool:
        from sqlalchemy import select
        async with self._owner.session_factory() as session:
            row_id = await session.scalar(
                select(self._table.c.id).where(
                    self._table.c.namespace_key == self.namespace_key,
                    self._table.c.tenant_id == tenant_id,
                    self._table.c.record_id == record_id,
                )
            )
        return row_id is not None

    async def _insert(self, record: object, *, record_id: str, tenant_id: str, sequence: int = 0, revision: int = 0, status: str = "", table: "Table | None" = None) -> object:
        target = self._table if table is None else table
        now = _record_time(record)
        values = {"namespace_key": self.namespace_key, "tenant_id": tenant_id, "record_id": record_id, "sequence": sequence, "revision": revision, "status": status, "payload": _encode_payload(record), "created_at": now, "updated_at": now}
        if isinstance(record, SessionRecord):
            values.update(session_id=record.session_id, profile="", head_execution_id=record.head_execution_id)
        if isinstance(record, ExecutionRecord):
            values.update(session_id=record.session_id, parent_execution_id=record.parent_execution_id, source_execution_id=record.source_execution_id, base_execution_id=record.base_execution_id, lineage_kind=record.lineage_kind, agent_run_sequence=record.agent_run_sequence)
        if isinstance(record, ToolOperationRecord):
            values.update(run_id=record.run_id, tool_call_id=record.tool_call_id, call_key=_composite_key(record.run_id, record.tool_call_id), owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at)
        if isinstance(record, TaskNodeView):
            values.update(owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at)
        if isinstance(record, IdempotencyRecord):
            values.update(scope=record.scope, key_hash=record.key_hash, identity_key=_composite_key(record.scope, record.key_hash))
        if isinstance(record, OperationLedgerRecord):
            values.update(resource_kind=record.resource_kind.value, resource_id=record.resource_id)
        from sqlalchemy import insert
        async with self._owner.session_factory() as session:
            async with session.begin():
                await session.execute(insert(target).values(values))
        return record

    async def _replace(self, record: object, *, record_id: str, tenant_id: str, expected_revision: int, revision: int, status: str = "", table: "Table | None" = None) -> object:
        target = self._table if table is None else table
        from sqlalchemy import update
        now = _record_time(record)
        async with self._owner.session_factory() as session:
            async with session.begin():
                values = {"payload": _encode_payload(record), "revision": revision, "status": status, "updated_at": now}
                if isinstance(record, SessionRecord):
                    values.update(session_id=record.session_id, profile="", head_execution_id=record.head_execution_id)
                if isinstance(record, ExecutionRecord):
                    values.update(sequence=record.event_sequence, session_id=record.session_id, parent_execution_id=record.parent_execution_id, source_execution_id=record.source_execution_id, base_execution_id=record.base_execution_id, lineage_kind=record.lineage_kind, agent_run_sequence=record.agent_run_sequence)
                if isinstance(record, ToolOperationRecord):
                    values.update(run_id=record.run_id, tool_call_id=record.tool_call_id, call_key=_composite_key(record.run_id, record.tool_call_id), owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at)
                if isinstance(record, TaskNodeView):
                    values.update(owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at)
                if isinstance(record, IdempotencyRecord):
                    values.update(scope=record.scope, key_hash=record.key_hash, identity_key=_composite_key(record.scope, record.key_hash))
                if isinstance(record, OperationLedgerRecord):
                    values.update(resource_kind=record.resource_kind.value, resource_id=record.resource_id)
                updated = await session.execute(
                    update(target)
                    .where(
                        target.c.namespace_key == self.namespace_key,
                        target.c.tenant_id == tenant_id,
                        target.c.record_id == record_id,
                        target.c.revision == expected_revision,
                    )
                    .values(values)
                )
                if updated.rowcount != 1:
                    current = await self._get(record_id, tenant_id=tenant_id, table=target)
                    if current is None:
                        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
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
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            raise

    async def get_header(self, record_id: str, *, tenant_id: str) -> ResourceRef | None:
        if not await self._exists(record_id, tenant_id=tenant_id):
            return None
        return ResourceRef(_resource_kind(self._table_name), record_id, tenant_id)

    async def get(self, record_id: str, key_hash: "str | None" = None, *, tenant_id: str) -> "object | None":
        if self._table_name == "idempotency":
            if key_hash is None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self.get_idempotency(record_id, key_hash, tenant_id=tenant_id)
        return await self._get(record_id, tenant_id=tenant_id)

    async def list(self, record_id: "str | None" = None, *, tenant_id: str, owner_principal_id: "str | None" = None, owner_id: "str | None" = None, cursor: "str | None" = None, after_sequence: int = 0, limit: int = 100) -> "tuple[SessionRecord, ...] | Page[object]":
        if self._table_name == "execution_events":
            if record_id is None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._list_event(record_id, tenant_id=tenant_id, after_sequence=after_sequence, limit=limit)
        if self._table_name == "memories":
            return await self.list_memories(
                tenant_id=tenant_id,
                owner_id="" if owner_id is None else owner_id,
                cursor=cursor,
                limit=limit,
            )
        predicates = ()
        if owner_principal_id is not None:
            predicates = (self._table.c.payload["value"]["owner_principal_id"].as_string() == owner_principal_id,)
        values = await self._list_records(tenant_id=tenant_id, predicates=predicates)
        return tuple(item for item in values if isinstance(item, SessionRecord))

    async def _list_records(
        self,
        *,
        tenant_id: str,
        table: "Table | None" = None,
        predicates: "tuple[ColumnElement[bool], ...]" = (),
        order_by: "tuple[ColumnElement[object], ...]" = (),
        limit: "int | None" = None,
    ) -> tuple[object, ...]:
        from sqlalchemy import select
        target = self._table if table is None else table
        statement = select(target.c.payload).where(
            target.c.namespace_key == self.namespace_key,
            target.c.tenant_id == tenant_id,
            *predicates,
        ).order_by(*(order_by or (target.c.record_id,)))
        if limit is not None:
            statement = statement.limit(limit)
        async with self._owner.session_factory() as session:
            payloads = (await session.execute(statement)).scalars().all()
        logical_name = self._owner.table_name(target)
        return tuple(_decode_payload(logical_name, payload) for payload in payloads)

    async def compare_and_swap(self, record_id: str, *, tenant_id: str, expected_revision: int = 0, expected_status: "IdempotencyStatus | None" = None, next_record: object) -> object:
        if self._table_name == "idempotency":
            if not isinstance(next_record, IdempotencyRecord) or expected_status is None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self.compare_idempotency(record_id, next_record.key_hash, tenant_id=tenant_id, expected_status=expected_status, next_record=next_record)
        if self._table_name == "operation_ledger":
            if not isinstance(next_record, OperationLedgerRecord) or not isinstance(expected_status, OperationStatus):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            current = await self._get(record_id, tenant_id=tenant_id)
            if not isinstance(current, OperationLedgerRecord) or current.status is not expected_status or next_record.sequence != current.sequence or next_record.tenant_id != current.tenant_id or next_record.resource_kind is not current.resource_kind or next_record.resource_id != current.resource_id or next_record.operation_id != current.operation_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await self._replace_status(next_record, table_name="operation_ledger", tenant_id=tenant_id, record_id=record_id, expected_status=expected_status)
        if isinstance(next_record, ExecutionRecord):
            if next_record.revision != expected_revision + 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        if isinstance(next_record, SessionRecord) and next_record.revision != expected_revision + 1:
            raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
        return await self._replace(next_record, record_id=record_id, tenant_id=tenant_id, expected_revision=expected_revision, revision=expected_revision + 1, status=_status(next_record))

    async def list_by_session(self, session_id: str, *, tenant_id: str, statuses: frozenset[ExecutionStatus] | None = None) -> tuple[ExecutionRecord, ...]:
        predicates = [self._table.c.session_id == session_id]
        if statuses is not None:
            predicates.append(self._table.c.status.in_(tuple(status.value for status in statuses)))
        values = await self._list_records(tenant_id=tenant_id, predicates=tuple(predicates))
        return tuple(item for item in values if isinstance(item, ExecutionRecord))

    async def list_children(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionRecord, ...]:
        values = await self._list_records(
            tenant_id=tenant_id,
            predicates=(self._table.c.parent_execution_id == execution_id,),
        )
        return tuple(item for item in values if isinstance(item, ExecutionRecord))

    async def create_plan(self, graph: TaskGraph, *, tenant_id: str) -> TaskGraphView:
        from sqlalchemy import insert
        view = TaskGraphView(graph.graph_id, TaskStatus.PENDING, graph.nodes)
        graph_table = self._owner.tables["task_graphs"]
        node_table = self._owner.tables["task_nodes"]
        now = datetime.now(timezone.utc)
        try:
            async with self._owner.session_factory() as session:
                async with session.begin():
                    await session.execute(insert(graph_table).values(namespace_key=self.namespace_key, tenant_id=tenant_id, record_id=graph.graph_id, sequence=0, revision=0, status=view.status.value, payload=_encode_payload(view), created_at=now, updated_at=now))
                    node_values = []
                    for node in graph.nodes:
                        record = TaskNodeView(graph.graph_id, node.task_id, node.dependencies, TaskStatus.PENDING, None, 0, None, None, None, None)
                        node_values.append({"namespace_key": self.namespace_key, "tenant_id": tenant_id, "record_id": f"{graph.graph_id}:{node.task_id}", "sequence": 0, "revision": 0, "status": TaskStatus.PENDING.value, "owner": None, "fence": 0, "lease_expires_at": None, "payload": _encode_payload(record), "created_at": now, "updated_at": now})
                    for offset in range(0, len(node_values), _SQL_CHUNK_INSERT_ROWS):
                        await session.execute(
                            insert(node_table).values(
                                node_values[offset:offset + _SQL_CHUNK_INSERT_ROWS]
                            )
                        )
        except Exception as error:
            if _is_integrity(error):
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            raise
        return view

    async def get_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None:
        value = await self._get(graph_id, tenant_id=tenant_id, table=self._owner.tables["task_graphs"])
        return value if isinstance(value, TaskGraphView) else None

    async def reconcile_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        view = await self.get_plan(graph_id, tenant_id=tenant_id)
        if view is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if view.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
            return await self.cancel_plan(graph_id, tenant_id=tenant_id)
        nodes = {item.task_id: item for item in await self.list_nodes(graph_id, tenant_id=tenant_id)}
        changed: list[TaskNodeView] = []
        for task_id, node in nodes.items():
            dependencies = tuple(nodes[dependency] for dependency in node.dependencies)
            next_status = node.status
            error_code = node.error_code
            error_digest = node.error_digest
            if node.status in {TaskStatus.PENDING, TaskStatus.READY} and any(item.status in {TaskStatus.FAILED, TaskStatus.BLOCKED} for item in dependencies):
                next_status = TaskStatus.BLOCKED
                error_code = ErrorCode.TASK_DEPENDENCY_FAILED.value
                error_digest = canonical_sha256({"graph_id": graph_id, "task_id": task_id, "reason": "dependency_failed"})
            elif node.status is TaskStatus.PENDING and all(item.status is TaskStatus.SUCCEEDED for item in dependencies):
                next_status = TaskStatus.READY
            if next_status is not node.status or error_code != node.error_code or error_digest != node.error_digest:
                updated = replace(node, status=next_status, error_code=error_code, error_digest=error_digest)
                await self._task_update(updated, tenant_id=tenant_id, expected_status=node.status, expected_owner=node.owner, expected_fence=node.fence)
                changed.append(updated)
                nodes[task_id] = updated
        statuses = tuple(item.status for item in nodes.values())
        if not statuses:
            graph_status = TaskStatus.SUCCEEDED
        elif all(item in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED} for item in statuses):
            graph_status = TaskStatus.FAILED if TaskStatus.FAILED in statuses else TaskStatus.BLOCKED if TaskStatus.BLOCKED in statuses else TaskStatus.CANCELLED if TaskStatus.CANCELLED in statuses else TaskStatus.SUCCEEDED
        elif TaskStatus.RUNNING in statuses:
            graph_status = TaskStatus.RUNNING
        elif TaskStatus.READY in statuses:
            graph_status = TaskStatus.READY
        else:
            graph_status = TaskStatus.PENDING
        updated_view = replace(view, status=graph_status)
        if updated_view != view:
            await self._replace(updated_view, record_id=graph_id, tenant_id=tenant_id, expected_revision=0, revision=0, status=graph_status.value, table=self._owner.tables["task_graphs"])
        return updated_view

    async def cancel_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        from sqlalchemy import select, update

        graph_table = self._owner.tables["task_graphs"]
        node_table = self._owner.tables["task_nodes"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                graph_row = (await session.execute(select(graph_table.c.payload, graph_table.c.revision).where(graph_table.c.namespace_key == self.namespace_key, graph_table.c.tenant_id == tenant_id, graph_table.c.record_id == graph_id).with_for_update())).first()
                if graph_row is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                view = _decode_payload("task_graphs", graph_row.payload)
                if not isinstance(view, TaskGraphView):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                status = view.status if view.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED} else TaskStatus.CANCELLED
                updated = replace(view, status=status)
                rows = (await session.execute(select(node_table.c.record_id, node_table.c.payload, node_table.c.revision).where(node_table.c.namespace_key == self.namespace_key, node_table.c.tenant_id == tenant_id, node_table.c.record_id.startswith(f"{graph_id}:", autoescape=True)).with_for_update())).all()
                for row in rows:
                    node = _decode_payload("task_nodes", row.payload)
                    if not isinstance(node, TaskNodeView) or node.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                        continue
                    next_node = replace(node, status=TaskStatus.CANCELLED, owner=None, lease_expires_at=None)
                    outcome = await session.execute(update(node_table).where(node_table.c.namespace_key == self.namespace_key, node_table.c.tenant_id == tenant_id, node_table.c.record_id == row.record_id, node_table.c.revision == row.revision).values(payload=_encode_payload(next_node), status=next_node.status.value, owner=None, lease_expires_at=None, revision=node_table.c.revision + 1, updated_at=_record_time(next_node)))
                    if outcome.rowcount != 1:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                outcome = await session.execute(update(graph_table).where(graph_table.c.namespace_key == self.namespace_key, graph_table.c.tenant_id == tenant_id, graph_table.c.record_id == graph_id, graph_table.c.revision == graph_row.revision).values(payload=_encode_payload(updated), status=updated.status.value, revision=graph_table.c.revision + 1, updated_at=datetime.now(timezone.utc)))
                if outcome.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return updated

    async def _claim_task(self, graph_id: str, task_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease:
        if not owner.strip() or not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        from sqlalchemy import select

        graph_table = self._owner.tables["task_graphs"]
        node_table = self._owner.tables["task_nodes"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                graph_payload = await session.scalar(
                    select(graph_table.c.payload).where(
                        graph_table.c.namespace_key == self.namespace_key,
                        graph_table.c.tenant_id == tenant_id,
                        graph_table.c.record_id == graph_id,
                    )
                )
                if graph_payload is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                graph = _decode_payload("task_graphs", graph_payload)
                if not isinstance(graph, TaskGraphView):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if graph.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                    raise AIError(ErrorCode.TASK_NOT_READY)
                payloads = (
                    await session.execute(
                        select(node_table.c.payload).where(
                            node_table.c.namespace_key == self.namespace_key,
                            node_table.c.tenant_id == tenant_id,
                            node_table.c.record_id.startswith(f"{graph_id}:", autoescape=True),
                        )
                    )
                ).scalars().all()
                graph_nodes = tuple(_decode_payload("task_nodes", payload) for payload in payloads)
                node = next(
                    (item for item in graph_nodes if isinstance(item, TaskNodeView) and item.task_id == task_id),
                    None,
                )
                if not isinstance(node, TaskNodeView):
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                dependency_ids = frozenset(node.dependencies)
                dependency_nodes = {
                    dependency.task_id: dependency
                    for dependency in graph_nodes
                    if isinstance(dependency, TaskNodeView) and dependency.task_id in dependency_ids
                }
                dependencies_ready = len(dependency_nodes) == len(dependency_ids) and all(
                    dependency.status is TaskStatus.SUCCEEDED
                    for dependency in dependency_nodes.values()
                )
                db_now = await resolve_dialect(session).database_now(session)
                expired = node.status is TaskStatus.RUNNING and node.lease_expires_at is not None and node.lease_expires_at <= db_now
                if (node.status not in {TaskStatus.PENDING, TaskStatus.READY} and not expired) or not dependencies_ready:
                    raise AIError(ErrorCode.TASK_NOT_READY)
                lease = TaskLease(graph_id, task_id, tenant_id, owner, node.fence + 1, db_now + timedelta(seconds=lease_seconds))
                updated = replace(node, status=TaskStatus.RUNNING, owner=owner, fence=lease.fence, lease_expires_at=lease.lease_expires_at)
                await self._task_update(
                    session,
                    updated,
                    tenant_id=tenant_id,
                    expected_status=node.status,
                    expected_owner=node.owner if expired else None,
                    expected_fence=node.fence,
                    db_now=db_now,
                    expected_lease_before=db_now if expired else None,
                )
                _logger.debug("SQL task lease claimed: graph=%s task=%s owner=%s fence=%s", graph_id, task_id, owner, lease.fence)
                return lease

    async def _renew_task(self, lease: TaskLease, *, tenant_id: str, lease_seconds: int) -> TaskLease:
        if lease.tenant_id != tenant_id or not lease.owner.strip() or not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        async with self._owner.session_factory() as session:
            async with session.begin():
                db_now = await resolve_dialect(session).database_now(session)
                node = await self._task_node(session, lease, tenant_id, db_now)
                renewed = replace(lease, lease_expires_at=db_now + timedelta(seconds=lease_seconds))
                await self._task_update(
                    session,
                    replace(node, lease_expires_at=renewed.lease_expires_at),
                    tenant_id=tenant_id,
                    expected_status=TaskStatus.RUNNING,
                    expected_owner=lease.owner,
                    expected_fence=lease.fence,
                    db_now=db_now,
                    expected_lease_after=db_now,
                )
                return renewed

    async def _complete_task(self, lease: TaskLease, *, tenant_id: str, execution_id: "str | None", result_digest: str) -> TaskTerminalRecord:
        if lease.tenant_id != tenant_id:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        async with self._owner.session_factory() as session:
            async with session.begin():
                db_now = await resolve_dialect(session).database_now(session)
                node = await self._task_node(session, lease, tenant_id, db_now)
                terminal = TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, TaskStatus.SUCCEEDED, result_digest, None, None, execution_id=execution_id)
                await self._task_update(
                    session,
                    replace(node, status=TaskStatus.SUCCEEDED, owner=None, lease_expires_at=None, result_digest=result_digest, execution_id=execution_id),
                    tenant_id=tenant_id,
                    expected_status=TaskStatus.RUNNING,
                    expected_owner=lease.owner,
                    expected_fence=lease.fence,
                    db_now=db_now,
                    expected_lease_after=db_now,
                )
                return terminal

    async def _fail_task(self, lease: TaskLease, *, tenant_id: str, error_code: str, error_digest: str) -> TaskTerminalRecord:
        if lease.tenant_id != tenant_id:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        async with self._owner.session_factory() as session:
            async with session.begin():
                db_now = await resolve_dialect(session).database_now(session)
                node = await self._task_node(session, lease, tenant_id, db_now)
                terminal = TaskTerminalRecord(lease.task_id, lease.owner, lease.fence, TaskStatus.FAILED, None, error_code, error_digest)
                await self._task_update(
                    session,
                    replace(node, status=TaskStatus.FAILED, owner=None, lease_expires_at=None, error_code=error_code, error_digest=error_digest),
                    tenant_id=tenant_id,
                    expected_status=TaskStatus.RUNNING,
                    expected_owner=lease.owner,
                    expected_fence=lease.fence,
                    db_now=db_now,
                    expected_lease_after=db_now,
                )
                return terminal

    async def _task_node(self, session: "AsyncSession", lease: TaskLease, tenant_id: str, db_now: datetime) -> TaskNodeView:
        from sqlalchemy import select

        table = self._owner.tables["task_nodes"]
        payload = await session.scalar(
            select(table.c.payload).where(
                table.c.namespace_key == self.namespace_key,
                table.c.tenant_id == tenant_id,
                table.c.record_id == f"{lease.graph_id}:{lease.task_id}",
            )
        )
        node = None if payload is None else _decode_payload("task_nodes", payload)
        if not isinstance(node, TaskNodeView) or node.owner != lease.owner or node.fence != lease.fence or node.lease_expires_at is None or node.lease_expires_at <= db_now:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        return node

    async def _task_update(self, session: "AsyncSession", record: TaskNodeView, *, tenant_id: str, expected_status: TaskStatus, expected_owner: "str | None", expected_fence: int, db_now: datetime, expected_lease_after: "datetime | None" = None, expected_lease_before: "datetime | None" = None) -> TaskNodeView:
        from sqlalchemy import update
        table = self._owner.tables["task_nodes"]
        record_id = f"{record.graph_id}:{record.task_id}"
        predicate = [table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record_id, table.c.status == expected_status.value, table.c.fence == expected_fence]
        predicate.append(table.c.owner.is_(None) if expected_owner is None else table.c.owner == expected_owner)
        if expected_lease_after is not None:
            predicate.append(table.c.lease_expires_at > expected_lease_after)
        if expected_lease_before is not None:
            predicate.append(table.c.lease_expires_at <= expected_lease_before)
        outcome = await session.execute(update(table).where(*predicate).values(payload=_encode_payload(record), revision=table.c.revision + 1, status=record.status.value, owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at, updated_at=db_now))
        if outcome.rowcount != 1:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        return record

    async def _replace_status(self, record: object, *, table_name: str, tenant_id: str, record_id: str, expected_status: object) -> object:
        from sqlalchemy import update
        table = self._owner.tables[table_name]
        status = expected_status.value if isinstance(expected_status, StrEnum) else str(expected_status)
        async with self._owner.session_factory() as session:
            async with session.begin():
                outcome = await session.execute(update(table).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record_id, table.c.status == status).values(payload=_encode_payload(record), revision=table.c.revision + 1, status=_status(record), updated_at=_record_time(record)))
                if outcome.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        return record

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[TaskNodeView, ...]:
        table = self._owner.tables["task_nodes"]
        values = await self._list_records(
            tenant_id=tenant_id,
            table=table,
            predicates=(table.c.record_id.startswith(f"{graph_id}:", autoescape=True),),
        )
        return tuple(item for item in values if isinstance(item, TaskNodeView))

    async def _list_evaluations(self, execution_id: str, *, tenant_id: str) -> tuple[EvaluationRecord, ...]:
        values = await self._list_records(
            tenant_id=tenant_id,
            predicates=(self._table.c.payload["value"]["execution_id"].as_string() == execution_id,),
        )
        return tuple(item for item in values if isinstance(item, EvaluationRecord) and item.execution_id == execution_id)

    async def put(self, record: MemoryRecord, *, expected_revision: "int | None") -> MemoryRecord:
        stored, replayed = await self.put_with_operation(record, expected_revision=expected_revision, operation=None)
        if replayed or stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return stored

    async def put_with_operation(self, record: MemoryRecord, *, expected_revision: "int | None", operation: "OperationLedgerInput | None") -> "tuple[MemoryRecord | None, bool]":
        from sqlalchemy import insert, select, update

        memory_table = self._owner.tables["memories"]
        operation_table = self._owner.tables["operation_ledger"]
        for attempt in range(32):
            try:
                async with self._owner.session_factory() as session:
                    async with session.begin():
                        if operation is not None:
                            operation_row = (await session.execute(select(operation_table.c.payload).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == operation.tenant_id, operation_table.c.record_id == operation.operation_id).with_for_update())).mappings().first()
                            if operation_row is not None:
                                existing_operation = _decode_payload("operation_ledger", operation_row["payload"])
                                if not isinstance(existing_operation, OperationLedgerRecord) or _operation_identity(existing_operation) != _operation_identity(operation):
                                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                                memory_row = (await session.execute(select(memory_table.c.payload).where(memory_table.c.namespace_key == self.namespace_key, memory_table.c.tenant_id == record.tenant_id, memory_table.c.record_id == record.memory_id))).mappings().first()
                                current = None if memory_row is None else _decode_payload("memories", memory_row["payload"])
                                return current if isinstance(current, MemoryRecord) else None, True
                        memory_row = (await session.execute(select(memory_table.c.payload).where(memory_table.c.namespace_key == self.namespace_key, memory_table.c.tenant_id == record.tenant_id, memory_table.c.record_id == record.memory_id).with_for_update())).mappings().first()
                        current = None if memory_row is None else _decode_payload("memories", memory_row["payload"])
                        if current is not None and not isinstance(current, MemoryRecord):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if current is None and expected_revision not in (None, 0):
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
                        if current is not None and current.revision != expected_revision:
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
                        now = _record_time(record)
                        values = {"namespace_key": self.namespace_key, "tenant_id": record.tenant_id, "record_id": record.memory_id, "sequence": 0, "revision": record.revision, "status": "", "payload": _encode_payload(record), "created_at": record.created_at, "updated_at": now}
                        if current is None:
                            await session.execute(insert(memory_table).values(values))
                        else:
                            outcome = await session.execute(update(memory_table).where(memory_table.c.namespace_key == self.namespace_key, memory_table.c.tenant_id == record.tenant_id, memory_table.c.record_id == record.memory_id, memory_table.c.revision == expected_revision).values(payload=values["payload"], revision=record.revision, updated_at=now))
                            if outcome.rowcount != 1:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
                        if operation is not None:
                            await self._append_operation_in_session(session, operation)
                        return record, False
            except Exception as error:
                if not _is_retryable_transaction(error) or attempt == 31:
                    raise
                await asyncio.sleep(0.01 * (attempt + 1))

    async def list_memories(self, *, tenant_id: str, owner_id: str, cursor: "str | None", limit: int) -> Page[MemoryRecord]:
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        predicates = [self._table.c.payload["value"]["owner_id"].as_string() == owner_id]
        if cursor is not None:
            predicates.append(self._table.c.record_id > cursor)
        values = await self._list_records(
            tenant_id=tenant_id,
            predicates=tuple(predicates),
            limit=limit + 1,
        )
        memories = tuple(item for item in values if isinstance(item, MemoryRecord))
        page = memories[:limit]
        return Page(page, page[-1].memory_id if len(memories) > limit else None)

    async def delete(self, memory_id: str, *, tenant_id: str, expected_revision: int) -> None:
        deleted, replayed = await self.delete_with_operation(memory_id, tenant_id=tenant_id, expected_revision=expected_revision, operation=None)
        if replayed or not deleted:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def delete_with_operation(self, memory_id: str, *, tenant_id: str, expected_revision: "int | None", operation: "OperationLedgerInput | None") -> "tuple[bool, bool]":
        from sqlalchemy import and_, select

        memory_table = self._owner.tables["memories"]
        operation_table = self._owner.tables["operation_ledger"]
        for attempt in range(32):
            try:
                async with self._owner.session_factory() as session:
                    async with session.begin():
                        if operation is not None:
                            operation_row = (await session.execute(select(operation_table.c.payload).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == operation.tenant_id, operation_table.c.record_id == operation.operation_id).with_for_update())).mappings().first()
                            if operation_row is not None:
                                existing_operation = _decode_payload("operation_ledger", operation_row["payload"])
                                if not isinstance(existing_operation, OperationLedgerRecord) or _operation_identity(existing_operation) != _operation_identity(operation):
                                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                                return False, True
                        if expected_revision is None:
                            memory_row = (await session.execute(select(memory_table.c.payload).where(memory_table.c.namespace_key == self.namespace_key, memory_table.c.tenant_id == tenant_id, memory_table.c.record_id == memory_id).with_for_update())).mappings().first()
                            current = None if memory_row is None else _decode_payload("memories", memory_row["payload"])
                            if current is not None and not isinstance(current, MemoryRecord):
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            if current is not None:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
                            deleted = False
                        else:
                            rows = await resolve_dialect(session).delete_returning(
                                session,
                                table=memory_table,
                                where=and_(
                                    memory_table.c.namespace_key == self.namespace_key,
                                    memory_table.c.tenant_id == tenant_id,
                                    memory_table.c.record_id == memory_id,
                                    memory_table.c.revision == expected_revision,
                                ),
                                returning=("payload",),
                            )
                            if not rows:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
                            current = _decode_payload("memories", rows[0]["payload"])
                            if not isinstance(current, MemoryRecord):
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            deleted = True
                        if operation is not None:
                            await self._append_operation_in_session(session, operation)
                        return deleted, False
            except Exception as error:
                if not _is_retryable_transaction(error) or attempt == 31:
                    raise
                await asyncio.sleep(0.01 * (attempt + 1))

    async def _append_operation_in_session(self, session: "AsyncSession", record: OperationLedgerInput) -> OperationLedgerRecord:
        from sqlalchemy import insert, select, update

        counter_table = self._owner.tables["operation_counters"]
        ledger_table = self._owner.tables["operation_ledger"]
        row = (await session.execute(select(ledger_table.c.payload).where(ledger_table.c.namespace_key == self.namespace_key, ledger_table.c.tenant_id == record.tenant_id, ledger_table.c.record_id == record.operation_id).with_for_update())).mappings().first()
        if row is not None:
            existing = _decode_payload("operation_ledger", row["payload"])
            if not isinstance(existing, OperationLedgerRecord) or _operation_identity(existing) != _operation_identity(record):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return existing
        counter = (await session.execute(select(counter_table.c.id, counter_table.c.revision).where(counter_table.c.namespace_key == self.namespace_key, counter_table.c.tenant_id == record.tenant_id, counter_table.c.resource_kind == record.resource_kind.value, counter_table.c.resource_id == record.resource_id).with_for_update())).mappings().first()
        sequence = 1 if counter is None else int(counter["revision"]) + 1
        counter_values = {"namespace_key": self.namespace_key, "tenant_id": record.tenant_id, "record_id": f"{record.resource_kind.value}:{record.resource_id}", "resource_kind": record.resource_kind.value, "resource_id": record.resource_id, "partition_key": _composite_key(record.resource_kind.value, record.resource_id), "sequence": sequence, "revision": sequence, "status": "", "payload": _encode_payload(sequence), "created_at": record.created_at, "updated_at": record.updated_at}
        if counter is None:
            await session.execute(insert(counter_table).values(counter_values))
        else:
            outcome = await session.execute(update(counter_table).where(counter_table.c.id == counter["id"], counter_table.c.revision == sequence - 1).values(sequence=sequence, revision=sequence, payload=_encode_payload(sequence), updated_at=record.updated_at))
            if outcome.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        created = OperationLedgerRecord(record.operation_id, record.tenant_id, record.resource_kind, record.resource_id, record.execution_id, record.kind, record.status, record.request_digest, record.result_ref, record.result_digest, record.error_code, record.compactable, sequence, record.created_at, record.updated_at)
        await session.execute(insert(ledger_table).values(namespace_key=self.namespace_key, tenant_id=record.tenant_id, record_id=record.operation_id, resource_kind=record.resource_kind.value, resource_id=record.resource_id, sequence=sequence, revision=0, status=record.status.value, payload=_encode_payload(created), created_at=record.created_at, updated_at=record.updated_at))
        return created

    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord:
        return await self.create(record)

    async def get_metadata(self, artifact_id: str, *, tenant_id: str) -> ArtifactRecord | None:
        value = await self._get(artifact_id, tenant_id=tenant_id, table=self._owner.tables["artifacts"])
        return value if isinstance(value, ArtifactRecord) else None

    async def list_by_execution(self, execution_id: str, *, tenant_id: str, cursor: "str | None" = None, limit: int = 100) -> "Page[ArtifactRecord] | tuple[EvaluationRecord, ...]":
        execution_predicate = self._table.c.payload["value"]["execution_id"].as_string() == execution_id
        if self._table_name == "idempotency":
            values = await self._list_records(
                tenant_id=tenant_id,
                predicates=(execution_predicate,),
                order_by=(self._table.c.scope, self._table.c.key_hash),
            )
            return tuple(item for item in values if isinstance(item, IdempotencyRecord))
        if self._table_name == "evaluations":
            return await self._list_evaluations(execution_id, tenant_id=tenant_id)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        predicates = [execution_predicate]
        if cursor is not None:
            predicates.append(self._table.c.record_id > cursor)
        values = await self._list_records(
            tenant_id=tenant_id,
            predicates=tuple(predicates),
            limit=limit + 1,
        )
        artifacts = tuple(item for item in values if isinstance(item, ArtifactRecord))
        page = artifacts[:limit]
        return Page(page, page[-1].artifact_id if len(artifacts) > limit else None)

    async def decide(self, approval_id: str, *, tenant_id: str, expected_status: ApprovalStatus, decision_id: str, decision: ApprovalDecision, principal_id: str, decision_digest: str, decided_at: datetime) -> ApprovalRecord:
        current = await self._get(approval_id, tenant_id=tenant_id)
        if not isinstance(current, ApprovalRecord) or current.status is not expected_status:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        updated = replace(current, status=ApprovalStatus.APPROVED if decision is ApprovalDecision.APPROVE else ApprovalStatus.DENIED, decision_id=decision_id, decision=decision, decided_by=principal_id, decision_digest=decision_digest, decided_at=decided_at)
        return await self._replace_status(updated, table_name="approvals", tenant_id=tenant_id, record_id=approval_id, expected_status=expected_status)

    async def _list_external(self, execution_id: str, *, tenant_id: str) -> tuple[object, ...]:
        pending_status = (
            ApprovalStatus.PENDING.value
            if self._table_name == "approvals"
            else ExternalCallStatus.PENDING.value
        )
        values = await self._list_records(
            tenant_id=tenant_id,
            predicates=(
                self._table.c.payload["value"]["execution_id"].as_string() == execution_id,
                self._table.c.status == pending_status,
            ),
        )
        return tuple(
            item for item in values
            if isinstance(item, (ApprovalRecord, ExternalResultRecord))
        )

    async def create_call(self, record: ExternalResultRecord) -> ExternalResultRecord:
        return await self.create(record)

    async def supply(self, call_id: str, *, tenant_id: str, expected_status: ExternalCallStatus, result_id: str, payload_ref: str, payload_digest: str, supplied_at: datetime) -> ExternalResultRecord:
        current = await self._get(call_id, tenant_id=tenant_id, table=self._owner.tables["external_results"])
        if not isinstance(current, ExternalResultRecord) or current.status is not expected_status:
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        updated = replace(current, status=ExternalCallStatus.SUPPLIED, result_id=result_id, payload_ref=payload_ref, payload_digest=payload_digest, supplied_at=supplied_at)
        return await self._replace_status(updated, table_name="external_results", tenant_id=tenant_id, record_id=call_id, expected_status=expected_status)

    async def advance_sequence(self, execution_id: str, *, tenant_id: str, kind: str, expected_sequence: int) -> ExecutionRecord:
        current = await self._get(execution_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if kind != "event":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        field = "event_sequence"
        current_sequence = current.event_sequence
        if current_sequence != expected_sequence:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        updated = replace(current, **{field: expected_sequence + 1, "revision": current.revision + 1, "updated_at": datetime.now(timezone.utc)})
        return await self._replace(updated, record_id=execution_id, tenant_id=tenant_id, expected_revision=current.revision, revision=updated.revision)

    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord:
        from sqlalchemy import insert, select, update
        if self._table_name != "executions":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        execution_table = self._table
        idempotency_table = self._owner.tables["idempotency"]
        event_table = self._owner.tables["execution_events"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                execution_row = (await session.execute(select(execution_table.c.payload).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == claim.tenant_id, execution_table.c.record_id == claim.execution_id, execution_table.c.status == ExecutionStatus.PENDING_START.value, execution_table.c.revision == claim.expected_revision, execution_table.c.sequence == claim.expected_event_sequence, execution_table.c.agent_run_sequence == 0))).mappings().first()
                identity_row = (await session.execute(select(idempotency_table.c.payload).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == claim.tenant_id, idempotency_table.c.scope == claim.scope, idempotency_table.c.key_hash == claim.key_hash))).mappings().first()
                if execution_row is None or identity_row is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                current = _decode_payload("executions", execution_row["payload"])
                identity = _decode_payload("idempotency", identity_row["payload"])
                if not isinstance(current, ExecutionRecord) or not isinstance(identity, IdempotencyRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if identity.status is not IdempotencyStatus.RESERVED or identity.execution_id != claim.execution_id or identity.request_digest != claim.request_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                started = replace(current, status=ExecutionStatus.STARTED, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=claim.started_at, agent_run_sequence=1)
                identity_started = replace(identity, status=IdempotencyStatus.STARTED, updated_at=claim.started_at)
                result = await session.execute(update(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == claim.tenant_id, execution_table.c.record_id == claim.execution_id, execution_table.c.revision == claim.expected_revision, execution_table.c.status == ExecutionStatus.PENDING_START.value, execution_table.c.sequence == claim.expected_event_sequence, execution_table.c.agent_run_sequence == 0).values(payload=_encode_payload(started), revision=started.revision, sequence=started.event_sequence, status=started.status.value, agent_run_sequence=1, updated_at=claim.started_at))
                if result.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await session.execute(update(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == claim.tenant_id, idempotency_table.c.scope == claim.scope, idempotency_table.c.key_hash == claim.key_hash, idempotency_table.c.status == IdempotencyStatus.RESERVED.value).values(payload=_encode_payload(identity_started), status=identity_started.status.value, updated_at=claim.started_at))
                event = ExecutionEventRecord(claim.execution_id, claim.tenant_id, claim.expected_event_sequence + 1, ExecutionEventType.EXECUTION_STARTED, {})
                await session.execute(insert(event_table).values(namespace_key=self.namespace_key, tenant_id=claim.tenant_id, record_id=f"{claim.execution_id}:{event.sequence}", sequence=event.sequence, revision=0, status=event.event_type.value, payload=_encode_payload(event), created_at=claim.started_at, updated_at=claim.started_at))
                return started

    async def reserve_start(self, reservation: ExecutionStartReservation) -> ExecutionStartReservationResult:
        from sqlalchemy import insert, select
        if self._table_name != "executions" or reservation.execution.tenant_id != reservation.idempotency.tenant_id or reservation.execution.execution_id != reservation.idempotency.execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution_table = self._table
        idempotency_table = self._owner.tables["idempotency"]
        execution = reservation.execution
        identity = reservation.idempotency
        try:
            async with self._owner.session_factory() as session:
                async with session.begin():
                    await session.execute(insert(idempotency_table).values(namespace_key=self.namespace_key, tenant_id=identity.tenant_id, record_id=f"{identity.scope}:{identity.key_hash}", scope=identity.scope, key_hash=identity.key_hash, identity_key=_composite_key(identity.scope, identity.key_hash), sequence=0, revision=0, status=identity.status.value, payload=_encode_payload(identity), created_at=identity.created_at, updated_at=identity.updated_at))
                    values = {"namespace_key": self.namespace_key, "tenant_id": execution.tenant_id, "record_id": execution.execution_id, "sequence": execution.event_sequence, "revision": execution.revision, "status": execution.status.value, "payload": _encode_payload(execution), "created_at": execution.created_at, "updated_at": execution.updated_at, "session_id": execution.session_id, "parent_execution_id": execution.parent_execution_id, "source_execution_id": execution.source_execution_id, "base_execution_id": execution.base_execution_id, "lineage_kind": execution.lineage_kind, "agent_run_sequence": execution.agent_run_sequence}
                    await session.execute(insert(execution_table).values(values))
        except Exception as error:
            if not _is_integrity(error):
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            async with self._owner.session_factory() as session:
                identity_row = (await session.execute(select(idempotency_table.c.payload).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == identity.tenant_id, idempotency_table.c.scope == identity.scope, idempotency_table.c.key_hash == identity.key_hash))).mappings().first()
                if identity_row is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                existing_identity = _decode_payload("idempotency", identity_row["payload"])
                if not isinstance(existing_identity, IdempotencyRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if existing_identity.request_digest != identity.request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                execution_row = (await session.execute(select(execution_table.c.payload).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == identity.tenant_id, execution_table.c.record_id == existing_identity.execution_id))).mappings().first()
                if execution_row is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                existing_execution = _decode_payload("executions", execution_row["payload"])
                if not isinstance(existing_execution, ExecutionRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return ExecutionStartReservationResult(existing_execution, existing_identity, False)
        return ExecutionStartReservationResult(execution, identity, True)

    async def claim_next_agent_run(self, execution_id: str, *, tenant_id: str, expected_revision: int, expected_agent_run_sequence: int) -> ExecutionRecord:
        from sqlalchemy import select, update
        if self._table_name != "executions":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        now = datetime.now(timezone.utc)
        async with self._owner.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(self._table.c.payload).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == tenant_id, self._table.c.record_id == execution_id, self._table.c.revision == expected_revision, self._table.c.status == ExecutionStatus.STARTED.value, self._table.c.agent_run_sequence == expected_agent_run_sequence))).mappings().first()
                if row is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                current = _decode_payload("executions", row["payload"])
                if not isinstance(current, ExecutionRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                updated = replace(current, revision=current.revision + 1, agent_run_sequence=current.agent_run_sequence + 1, updated_at=now)
                result = await session.execute(update(self._table).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == tenant_id, self._table.c.record_id == execution_id, self._table.c.revision == expected_revision, self._table.c.status == ExecutionStatus.STARTED.value, self._table.c.agent_run_sequence == expected_agent_run_sequence).values(payload=_encode_payload(updated), revision=updated.revision, agent_run_sequence=updated.agent_run_sequence, updated_at=now))
                if result.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return updated

    async def mark_start_unknown(self, commit: ExecutionStartUnknownCommit) -> ExecutionRecord:
        from sqlalchemy import insert, select, update
        if self._table_name != "executions":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        event_table = self._owner.tables["execution_events"]
        idempotency_table = self._owner.tables["idempotency"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(self._table.c.payload).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == commit.tenant_id, self._table.c.record_id == commit.execution_id, self._table.c.revision == commit.expected_revision, self._table.c.status == ExecutionStatus.STARTED.value))).mappings().first()
                identity_row = (await session.execute(select(idempotency_table.c.payload).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == commit.tenant_id, idempotency_table.c.scope == commit.scope, idempotency_table.c.key_hash == commit.key_hash))).mappings().first()
                if row is None or identity_row is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                current = _decode_payload("executions", row["payload"])
                identity = _decode_payload("idempotency", identity_row["payload"])
                if not isinstance(current, ExecutionRecord) or not isinstance(identity, IdempotencyRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if identity.status is not IdempotencyStatus.STARTED:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                unknown = replace(current, status=ExecutionStatus.START_UNKNOWN, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.occurred_at)
                await session.execute(update(self._table).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == commit.tenant_id, self._table.c.record_id == commit.execution_id, self._table.c.revision == commit.expected_revision, self._table.c.status == ExecutionStatus.STARTED.value).values(payload=_encode_payload(unknown), revision=unknown.revision, status=unknown.status.value, sequence=unknown.event_sequence, updated_at=commit.occurred_at))
                await session.execute(update(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == commit.tenant_id, idempotency_table.c.scope == commit.scope, idempotency_table.c.key_hash == commit.key_hash, idempotency_table.c.status == IdempotencyStatus.STARTED.value).values(payload=_encode_payload(replace(identity, status=IdempotencyStatus.START_UNKNOWN, updated_at=commit.occurred_at)), status=IdempotencyStatus.START_UNKNOWN.value, updated_at=commit.occurred_at))
                event = ExecutionEventRecord(commit.execution_id, commit.tenant_id, current.event_sequence + 1, ExecutionEventType.EXECUTION_START_UNKNOWN, {})
                await session.execute(insert(event_table).values(namespace_key=self.namespace_key, tenant_id=commit.tenant_id, record_id=f"{commit.execution_id}:{event.sequence}", sequence=event.sequence, revision=0, status=event.event_type.value, payload=_encode_payload(event), created_at=commit.occurred_at, updated_at=commit.occurred_at))
                return unknown

    async def request_cancel(self, commit: ExecutionCancelRequestCommit) -> ExecutionRecord:
        from sqlalchemy import insert, select, update
        if self._table_name != "executions":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        operation_table = self._owner.tables["operation_ledger"]
        event_table = self._owner.tables["execution_events"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                operation_row = (await session.execute(select(operation_table.c.payload).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == commit.tenant_id, operation_table.c.record_id == commit.operation_id).with_for_update())).mappings().first()
                execution_row = (await session.execute(select(self._table.c.payload).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == commit.tenant_id, self._table.c.record_id == commit.execution_id).with_for_update())).mappings().first()
                if operation_row is None or execution_row is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                operation = _decode_payload("operation_ledger", operation_row["payload"])
                current = _decode_payload("executions", execution_row["payload"])
                if not isinstance(operation, OperationLedgerRecord) or not isinstance(current, ExecutionRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if operation.status is not OperationStatus.PENDING or operation.execution_id != commit.execution_id:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if current.status is ExecutionStatus.CANCELLING:
                    return current
                if current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} or current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                updated = replace(current, status=ExecutionStatus.CANCELLING, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.requested_at)
                outcome = await session.execute(update(self._table).where(self._table.c.namespace_key == self.namespace_key, self._table.c.tenant_id == commit.tenant_id, self._table.c.record_id == commit.execution_id, self._table.c.revision == commit.expected_revision, self._table.c.sequence == commit.expected_event_sequence, self._table.c.status.notin_({ExecutionStatus.SUCCEEDED.value, ExecutionStatus.FAILED.value, ExecutionStatus.CANCELLED.value})).values(payload=_encode_payload(updated), revision=updated.revision, sequence=updated.event_sequence, status=updated.status.value, updated_at=updated.updated_at))
                if outcome.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                event = ExecutionEventRecord(commit.execution_id, commit.tenant_id, updated.event_sequence, ExecutionEventType.CANCEL_REQUESTED, {})
                await session.execute(insert(event_table).values(namespace_key=self.namespace_key, tenant_id=commit.tenant_id, record_id=f"{commit.execution_id}:{event.sequence}", sequence=event.sequence, revision=0, status=event.event_type.value, payload=_encode_payload(event), created_at=commit.requested_at, updated_at=commit.requested_at))
                return updated

    async def commit_terminal(self, execution_id: "str | ExecutionTerminalCommit", *, tenant_id: "str | None" = None, expected_revision: "int | None" = None, next_record: "ExecutionRecord | None" = None) -> "ExecutionRecord | ExecutionTerminalCommitResult":
        if self._table_name == "results":
            if not isinstance(execution_id, ExecutionTerminalCommit):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._commit_result(execution_id)
        if not isinstance(execution_id, str) or tenant_id is None or expected_revision is None or next_record is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self._get(execution_id, tenant_id=tenant_id)
        if current is None or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self._replace(next_record, record_id=execution_id, tenant_id=tenant_id, expected_revision=expected_revision, revision=expected_revision + 1, status=next_record.status.value)

    async def _commit_result(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult:
        from sqlalchemy import insert, select, update
        execution_table = self._owner.tables["executions"]
        result_table = self._owner.tables["results"]
        event_table = self._owner.tables["execution_events"]
        idempotency_table = self._owner.tables["idempotency"]
        operation_table = self._owner.tables["operation_ledger"]
        session_table = self._owner.tables["sessions"]
        execution = commit.execution
        if execution.event_sequence != commit.expected_event_sequence + 1 or commit.terminal_event_type not in {ExecutionEventType.EXECUTION_SUCCEEDED, ExecutionEventType.EXECUTION_FAILED, ExecutionEventType.EXECUTION_CANCELLED}:
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        if commit.session_head is not None and (execution.status is not ExecutionStatus.SUCCEEDED or execution.lineage_kind not in {ExecutionLineageKind.SESSION_RESUME, ExecutionLineageKind.RETRY}):
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        async with self._owner.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(execution_table.c.payload).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == execution.tenant_id, execution_table.c.record_id == execution.execution_id).with_for_update())).mappings().first()
                if row is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                current = _decode_payload("executions", row["payload"])
                existing_result_row = (await session.execute(select(result_table.c.payload).where(result_table.c.namespace_key == self.namespace_key, result_table.c.tenant_id == execution.tenant_id, result_table.c.record_id == execution.execution_id))).mappings().first()
                if existing_result_row is not None:
                    existing_result = _decode_payload("results", existing_result_row["payload"])
                    terminal_event = (await session.execute(select(event_table.c.payload).where(event_table.c.namespace_key == self.namespace_key, event_table.c.tenant_id == execution.tenant_id, event_table.c.record_id == f"{execution.execution_id}:{commit.expected_event_sequence + 1}"))).mappings().first()
                    identity_ok = True
                    if commit.idempotency is not None:
                        identity_row = (await session.execute(select(idempotency_table.c.payload).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == execution.tenant_id, idempotency_table.c.scope == commit.idempotency.scope, idempotency_table.c.key_hash == commit.idempotency.key_hash))).mappings().first()
                        identity_value = None if identity_row is None else _decode_payload("idempotency", identity_row["payload"])
                        identity_ok = isinstance(identity_value, IdempotencyRecord) and identity_value.execution_id == execution.execution_id and identity_value.request_digest == commit.idempotency.request_digest and identity_value.status is commit.idempotency.next_status and identity_value.result_digest == commit.idempotency.result_digest and identity_value.error_code == commit.idempotency.error_code
                    operation_ok = True
                    if commit.operation is not None:
                        operation_row = (await session.execute(select(operation_table.c.payload).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == execution.tenant_id, operation_table.c.record_id == commit.operation.operation_id))).mappings().first()
                        operation_value = None if operation_row is None else _decode_payload("operation_ledger", operation_row["payload"])
                        operation_ok = isinstance(operation_value, OperationLedgerRecord) and operation_value.execution_id == execution.execution_id and operation_value.status is commit.operation.next_status and operation_value.result_ref == commit.operation.result_ref and operation_value.result_digest == commit.operation.result_digest and operation_value.error_code == commit.operation.error_code
                    if isinstance(existing_result, ResultRecord) and existing_result == commit.result and isinstance(current, ExecutionRecord) and current == execution and terminal_event is not None and _decode_payload("execution_events", terminal_event["payload"]) == ExecutionEventRecord(execution.execution_id, execution.tenant_id, commit.expected_event_sequence + 1, commit.terminal_event_type, commit.terminal_event_payload) and identity_ok and operation_ok:
                        return ExecutionTerminalCommitResult(current, existing_result)
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
                if not isinstance(current, ExecutionRecord) or current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence or current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} or execution.revision != commit.expected_revision + 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                locked_session = None
                if commit.session_head is not None:
                    locked_session_row = (await session.execute(select(session_table.c.payload).where(session_table.c.namespace_key == self.namespace_key, session_table.c.tenant_id == execution.tenant_id, session_table.c.session_id == commit.session_head.session_id).with_for_update())).mappings().first()
                    if locked_session_row is None:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    locked_session = _decode_payload("sessions", locked_session_row["payload"])
                    if not isinstance(locked_session, SessionRecord) or locked_session.status is not SessionStatus.OPEN or locked_session.head_execution_id != commit.session_head.expected_head_execution_id:
                        _logger.warning("session head CAS rejected: session=%s expected=%s", commit.session_head.session_id, commit.session_head.expected_head_execution_id)
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                identity = None
                if commit.idempotency is not None:
                    identity_row = (await session.execute(select(idempotency_table.c.payload).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == execution.tenant_id, idempotency_table.c.scope == commit.idempotency.scope, idempotency_table.c.key_hash == commit.idempotency.key_hash).with_for_update())).mappings().first()
                    if identity_row is None:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    identity = _decode_payload("idempotency", identity_row["payload"])
                    if not isinstance(identity, IdempotencyRecord) or identity.execution_id != execution.execution_id or identity.request_digest != commit.idempotency.request_digest or identity.status is not commit.idempotency.expected_status:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                else:
                    from sqlalchemy import func
                    matching_identities = (
                        select(idempotency_table.c.id)
                        .where(
                            idempotency_table.c.namespace_key == self.namespace_key,
                            idempotency_table.c.tenant_id == execution.tenant_id,
                            idempotency_table.c.payload["value"]["execution_id"].as_string() == execution.execution_id,
                        )
                        .limit(2)
                        .subquery()
                    )
                    identity_count = await session.scalar(
                        select(func.count()).select_from(matching_identities)
                    )
                    if identity_count is not None and identity_count > 1:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                operation = None
                if commit.operation is not None:
                    operation_row = (await session.execute(select(operation_table.c.payload).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == execution.tenant_id, operation_table.c.record_id == commit.operation.operation_id).with_for_update())).mappings().first()
                    if operation_row is None:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    operation = _decode_payload("operation_ledger", operation_row["payload"])
                    if not isinstance(operation, OperationLedgerRecord) or operation.execution_id != execution.execution_id or operation.status is not commit.operation.expected_status:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                updated = await session.execute(update(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == execution.tenant_id, execution_table.c.record_id == execution.execution_id, execution_table.c.revision == commit.expected_revision, execution_table.c.sequence == commit.expected_event_sequence).values(payload=_encode_payload(execution), revision=execution.revision, sequence=execution.event_sequence, status=execution.status.value, updated_at=_record_time(execution)))
                if updated.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await session.execute(insert(result_table).values(namespace_key=self.namespace_key, tenant_id=execution.tenant_id, record_id=execution.execution_id, sequence=0, revision=0, status=commit.result.status.value, payload=_encode_payload(commit.result), created_at=_record_time(commit.result), updated_at=_record_time(commit.result)))
                terminal_event = ExecutionEventRecord(execution.execution_id, execution.tenant_id, commit.expected_event_sequence + 1, commit.terminal_event_type, commit.terminal_event_payload)
                await session.execute(insert(event_table).values(namespace_key=self.namespace_key, tenant_id=execution.tenant_id, record_id=f"{execution.execution_id}:{terminal_event.sequence}", sequence=terminal_event.sequence, revision=0, status=terminal_event.event_type.value, payload=_encode_payload(terminal_event), created_at=_record_time(commit.result), updated_at=_record_time(commit.result)))
                if commit.idempotency is not None and identity is not None:
                    updated_identity = replace(identity, status=commit.idempotency.next_status, result_digest=commit.idempotency.result_digest, error_code=commit.idempotency.error_code, updated_at=_record_time(execution))
                    outcome = await session.execute(update(idempotency_table).where(idempotency_table.c.namespace_key == self.namespace_key, idempotency_table.c.tenant_id == execution.tenant_id, idempotency_table.c.scope == commit.idempotency.scope, idempotency_table.c.key_hash == commit.idempotency.key_hash, idempotency_table.c.status == commit.idempotency.expected_status.value).values(payload=_encode_payload(updated_identity), revision=idempotency_table.c.revision + 1, status=updated_identity.status.value, updated_at=updated_identity.updated_at))
                    if outcome.rowcount != 1:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                if commit.operation is not None and operation is not None:
                    updated_operation = replace(operation, status=commit.operation.next_status, result_ref=commit.operation.result_ref, result_digest=commit.operation.result_digest, error_code=commit.operation.error_code, updated_at=_record_time(execution))
                    outcome = await session.execute(update(operation_table).where(operation_table.c.namespace_key == self.namespace_key, operation_table.c.tenant_id == execution.tenant_id, operation_table.c.record_id == commit.operation.operation_id, operation_table.c.status == commit.operation.expected_status.value).values(payload=_encode_payload(updated_operation), status=updated_operation.status.value, updated_at=updated_operation.updated_at))
                    if outcome.rowcount != 1:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                if commit.session_head is not None:
                    if not isinstance(locked_session, SessionRecord):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    updated_session = replace(locked_session, revision=locked_session.revision + 1, head_execution_id=commit.session_head.next_head_execution_id, updated_at=_record_time(execution))
                    outcome = await session.execute(update(session_table).where(session_table.c.namespace_key == self.namespace_key, session_table.c.tenant_id == execution.tenant_id, session_table.c.session_id == commit.session_head.session_id, session_table.c.revision == locked_session.revision).values(payload=_encode_payload(updated_session), head_execution_id=updated_session.head_execution_id, revision=updated_session.revision, updated_at=updated_session.updated_at))
                    if outcome.rowcount != 1:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
        return ExecutionTerminalCommitResult(execution, commit.result)

    async def _append_event(self, execution_id: str, *, tenant_id: str, expected_sequence: int, event_type: "ExecutionEventType | None" = None, payload: JsonValue) -> "ExecutionEventRecord":
        if self._table_name != "execution_events" or event_type is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        from sqlalchemy import insert, select, update
        execution_table = self._owner.tables["executions"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(execution_table.c.payload).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == tenant_id, execution_table.c.record_id == execution_id))).mappings().first()
                if row is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                current = _decode_payload("executions", row["payload"])
                field = "event_sequence"
                current_sequence = current.event_sequence
                if current_sequence != expected_sequence:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                next_value = expected_sequence + 1
                updated_execution = replace(current, **{field: next_value, "revision": current.revision + 1, "updated_at": datetime.now(timezone.utc)})
                outcome = await session.execute(update(execution_table).where(execution_table.c.namespace_key == self.namespace_key, execution_table.c.tenant_id == tenant_id, execution_table.c.record_id == execution_id, execution_table.c.revision == current.revision).values(payload=_encode_payload(updated_execution), revision=updated_execution.revision, sequence=updated_execution.event_sequence, updated_at=updated_execution.updated_at))
                if outcome.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                item = ExecutionEventRecord(execution_id, tenant_id, next_value, event_type, payload)
                await session.execute(insert(self._table).values(namespace_key=self.namespace_key, tenant_id=tenant_id, record_id=f"{execution_id}:{next_value}", sequence=next_value, revision=0, status="", payload=_encode_payload(item), created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)))
                return item

    async def _list_event(self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int) -> Page[object]:
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        values = await self._list_records(
            tenant_id=tenant_id,
            predicates=(
                self._table.c.record_id.startswith(f"{execution_id}:", autoescape=True),
                self._table.c.sequence > after_sequence,
            ),
            order_by=(self._table.c.sequence,),
            limit=limit + 1,
        )
        page = values[:limit]
        return Page(page, str(page[-1].sequence) if len(values) > len(page) else None)

    async def get_idempotency(self, scope: str, key_hash: str, *, tenant_id: str) -> IdempotencyRecord | None:
        return await self._get(f"{scope}:{key_hash}", tenant_id=tenant_id)

    async def compare_idempotency(self, scope: str, key_hash: str, *, tenant_id: str, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord:
        from sqlalchemy import select, update
        table = self._owner.tables["idempotency"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(table.c.payload, table.c.revision).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.scope == scope, table.c.key_hash == key_hash))).mappings().first()
                if row is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                current = _decode_payload("idempotency", row["payload"])
                if not isinstance(current, IdempotencyRecord) or current.status is not expected_status:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if next_record.tenant_id != tenant_id or next_record.scope != scope or next_record.key_hash != key_hash:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                outcome = await session.execute(update(table).where(table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.scope == scope, table.c.key_hash == key_hash, table.c.revision == row["revision"], table.c.status == expected_status.value).values(payload=_encode_payload(next_record), revision=int(row["revision"]) + 1, status=next_record.status.value, updated_at=next_record.updated_at))
                if outcome.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        return next_record

    async def append(self, record: "OperationLedgerInput | str", *, tenant_id: "str | None" = None, expected_sequence: "int | None" = None, event_type: "ExecutionEventType | None" = None, payload: "JsonValue | None" = None) -> "OperationLedgerRecord | ExecutionEventRecord | object":
        if self._table_name == "execution_events":
            if not isinstance(record, str) or tenant_id is None or expected_sequence is None or payload is None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._append_event(record, tenant_id=tenant_id, expected_sequence=expected_sequence, event_type=event_type, payload=payload)
        if not isinstance(record, OperationLedgerInput):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        for attempt in range(32):
            try:
                return await self._append_operation(record)
            except Exception as error:
                retryable = _is_retryable_transaction(error) or (isinstance(error, AIError) and error.code is ErrorCode.STORAGE_CONFLICT)
                if not retryable or attempt == 31:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                await asyncio.sleep(0.01 * (attempt + 1))

    async def _append_operation(self, record: OperationLedgerInput) -> OperationLedgerRecord:
        from sqlalchemy import insert, select, update
        counter_table = self._owner.tables["operation_counters"]
        ledger_table = self._owner.tables["operation_ledger"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                row = (await session.execute(select(ledger_table.c.payload).where(ledger_table.c.namespace_key == self.namespace_key, ledger_table.c.tenant_id == record.tenant_id, ledger_table.c.record_id == record.operation_id))).mappings().first()
                if row is not None:
                    existing = _decode_payload("operation_ledger", row["payload"])
                    if _operation_identity(existing) != _operation_identity(record):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    return existing
                counter = (await session.execute(select(counter_table.c.id, counter_table.c.revision).where(counter_table.c.namespace_key == self.namespace_key, counter_table.c.tenant_id == record.tenant_id, counter_table.c.resource_kind == record.resource_kind.value, counter_table.c.resource_id == record.resource_id).with_for_update())).mappings().first()
                sequence = 1 if counter is None else int(counter["revision"]) + 1
                counter_values = {"namespace_key": self.namespace_key, "tenant_id": record.tenant_id, "record_id": f"{record.resource_kind.value}:{record.resource_id}", "resource_kind": record.resource_kind.value, "resource_id": record.resource_id, "partition_key": _composite_key(record.resource_kind.value, record.resource_id), "sequence": sequence, "revision": sequence, "status": "", "payload": _encode_payload(sequence), "created_at": record.created_at, "updated_at": record.updated_at}
                if counter is None:
                    await session.execute(insert(counter_table).values(counter_values))
                else:
                    result = await session.execute(update(counter_table).where(counter_table.c.id == counter["id"], counter_table.c.revision == sequence - 1).values(sequence=sequence, revision=sequence, payload=_encode_payload(sequence), updated_at=record.updated_at))
                    if result.rowcount != 1:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                created = OperationLedgerRecord(record.operation_id, record.tenant_id, record.resource_kind, record.resource_id, record.execution_id, record.kind, record.status, record.request_digest, record.result_ref, record.result_digest, record.error_code, record.compactable, sequence, record.created_at, record.updated_at)
                await session.execute(insert(ledger_table).values(namespace_key=self.namespace_key, tenant_id=record.tenant_id, record_id=record.operation_id, resource_kind=record.resource_kind.value, resource_id=record.resource_id, sequence=sequence, revision=0, status=record.status.value, payload=_encode_payload(created), created_at=record.created_at, updated_at=record.updated_at))
                return created

    async def list_pending(self, resource_kind: "ResourceKind | str", resource_id: "str | None" = None, *, tenant_id: str, limit: int = 100) -> "tuple[OperationLedgerRecord, ...] | tuple[object, ...]":
        if self._table_name in {"approvals", "external_results"}:
            return await self._list_external(str(resource_kind), tenant_id=tenant_id)
        if resource_id is None or not isinstance(resource_kind, ResourceKind):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not 1 <= limit <= 1000:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        table = self._owner.tables["operation_ledger"]
        values = await self._list_records(
            tenant_id=tenant_id,
            table=table,
            predicates=(
                table.c.resource_kind == resource_kind.value,
                table.c.resource_id == resource_id,
                table.c.status.in_((OperationStatus.PENDING.value, OperationStatus.RUNNING.value)),
            ),
            order_by=(table.c.sequence,),
            limit=limit,
        )
        return tuple(item for item in values if isinstance(item, OperationLedgerRecord))

    async def compact_terminal(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, through_sequence: int) -> str:
        return hashlib.sha256(f"{self.namespace}:{tenant_id}:{resource_kind.value}:{resource_id}:{through_sequence}".encode()).hexdigest()

    async def reserve(self, record: "IdempotencyRecord | ToolOperationRecord") -> "IdempotencyRecord | ToolOperationRecord":
        if self._table_name == "idempotency":
            if not isinstance(record, IdempotencyRecord):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            try:
                return await self._insert(record, record_id=f"{record.scope}:{record.key_hash}", tenant_id=record.tenant_id, status=record.status.value, table=self._owner.tables["idempotency"])
            except Exception as error:
                if not _is_integrity(error):
                    raise
                existing = await self.get_idempotency(record.scope, record.key_hash, tenant_id=record.tenant_id)
                if existing is not None and existing.request_digest == record.request_digest and existing.execution_id == record.execution_id:
                    return existing
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
        if not isinstance(record, ToolOperationRecord):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if re.fullmatch(r"[0-9a-f]{64}", record.idempotency_key_hash) is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        try:
            return await self.create(record)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_CONFLICT:
                existing = await self.get_operation(record.operation_id, tenant_id=record.tenant_id)
                if existing is not None and _tool_identity(existing) == _tool_identity(record):
                    return existing
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT) from error
            raise

    async def get_operation(self, operation_id: str, *, tenant_id: str) -> ToolOperationRecord | None:
        return await self._get(operation_id, tenant_id=tenant_id)

    async def claim(self, operation_id: str, task_id: "str | None" = None, *, tenant_id: str, owner: str, lease_seconds: int) -> "ToolOperationRecord | TaskLease":
        if self._table_name == "task_graphs":
            if task_id is None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._claim_task(operation_id, task_id, tenant_id=tenant_id, owner=owner, lease_seconds=lease_seconds)
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        now = datetime.now(timezone.utc)
        if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
            raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
        if current.status is ToolOperationStatus.CLAIMED and current.lease_expires_at is not None and current.lease_expires_at > now:
            raise AIError(ErrorCode.TASK_OWNER_CONFLICT)
        if current.status is ToolOperationStatus.CLAIMED and not current.replay_safe:
            unknown = replace(current, status=ToolOperationStatus.EFFECT_UNKNOWN, owner=None, lease_expires_at=None, updated_at=now)
            await self._tool_update(unknown, tenant_id=tenant_id, expected_status=ToolOperationStatus.CLAIMED, expected_owner=current.owner, expected_fence=current.fence, expected_lease_before=now)
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        updated = replace(current, status=ToolOperationStatus.CLAIMED, owner=owner, fence=current.fence + 1, lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now)
        return await self._tool_update(updated, tenant_id=tenant_id, expected_status=current.status, expected_owner=current.owner, expected_fence=current.fence if current.status is ToolOperationStatus.CLAIMED else None, expected_lease_before=now if current.status is ToolOperationStatus.CLAIMED else None)

    async def renew(self, operation_id: "str | TaskLease", *, tenant_id: str, owner: "str | None" = None, fence: "int | None" = None, lease_seconds: "int | None" = None) -> "ToolOperationRecord | TaskLease":
        if self._table_name == "task_graphs":
            if not isinstance(operation_id, TaskLease) or lease_seconds is None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._renew_task(operation_id, tenant_id=tenant_id, lease_seconds=lease_seconds)
        if owner is None or fence is None or lease_seconds is None or not isinstance(operation_id, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self._owned(operation_id, tenant_id, owner, fence)
        now = datetime.now(timezone.utc)
        return await self._tool_update(replace(current, lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now), tenant_id=tenant_id, expected_status=ToolOperationStatus.CLAIMED, expected_owner=owner, expected_fence=fence, expected_lease_after=now)

    async def complete(self, operation_id: "str | TaskLease", *, tenant_id: str, owner: "str | None" = None, fence: "int | None" = None, result_ref: "str | None" = None, execution_id: "str | None" = None, result_digest: str) -> "ToolOperationRecord | TaskTerminalRecord":
        if self._table_name == "task_graphs":
            if not isinstance(operation_id, TaskLease):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._complete_task(operation_id, tenant_id=tenant_id, execution_id=execution_id, result_digest=result_digest)
        if not isinstance(operation_id, str) or owner is None or fence is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status is ToolOperationStatus.COMPLETED and current.result_digest == result_digest:
            return current
        if current.status is ToolOperationStatus.COMPLETED:
            raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
        current = await self._owned(operation_id, tenant_id, owner, fence)
        now = datetime.now(timezone.utc)
        return await self._tool_update(replace(current, status=ToolOperationStatus.COMPLETED, result_ref=result_ref, result_digest=result_digest, lease_expires_at=None, updated_at=now), tenant_id=tenant_id, expected_status=ToolOperationStatus.CLAIMED, expected_owner=owner, expected_fence=fence, expected_lease_after=now)

    async def fail(self, operation_id: "str | TaskLease", *, tenant_id: str, owner: "str | None" = None, fence: "int | None" = None, error_code: str, error_digest: "str | None" = None) -> "ToolOperationRecord | TaskTerminalRecord":
        if self._table_name == "task_graphs":
            if not isinstance(operation_id, TaskLease) or error_digest is None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return await self._fail_task(operation_id, tenant_id=tenant_id, error_code=error_code, error_digest=error_digest)
        if not isinstance(operation_id, str) or owner is None or fence is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status is ToolOperationStatus.FAILED and current.error_code == error_code:
            return current
        if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
            raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
        current = await self._owned(operation_id, tenant_id, owner, fence)
        now = datetime.now(timezone.utc)
        return await self._tool_update(replace(current, status=ToolOperationStatus.FAILED, error_code=error_code, lease_expires_at=None, updated_at=now), tenant_id=tenant_id, expected_status=ToolOperationStatus.CLAIMED, expected_owner=owner, expected_fence=fence, expected_lease_after=now)

    async def _tool_update(self, record: ToolOperationRecord, *, tenant_id: str, expected_status: ToolOperationStatus, expected_owner: "str | None", expected_fence: "int | None", expected_lease_after: "datetime | None" = None, expected_lease_before: "datetime | None" = None) -> ToolOperationRecord:
        from sqlalchemy import update
        table = self._owner.tables["tool_operations"]
        async with self._owner.session_factory() as session:
            async with session.begin():
                predicate = [table.c.namespace_key == self.namespace_key, table.c.tenant_id == tenant_id, table.c.record_id == record.operation_id, table.c.status == expected_status.value]
                predicate.append(table.c.owner.is_(None) if expected_owner is None else table.c.owner == expected_owner)
                if expected_fence is not None:
                    predicate.append(table.c.fence == expected_fence)
                if expected_lease_after is not None:
                    predicate.append(table.c.lease_expires_at > expected_lease_after)
                if expected_lease_before is not None:
                    predicate.append(table.c.lease_expires_at <= expected_lease_before)
                outcome = await session.execute(update(table).where(*predicate).values(payload=_encode_payload(record), revision=table.c.revision + 1, status=record.status.value, owner=record.owner, tool_call_id=record.tool_call_id, run_id=record.run_id, fence=record.fence, lease_expires_at=record.lease_expires_at, updated_at=record.updated_at))
                if outcome.rowcount != 1:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
        return record

    async def _owned(self, operation_id: str, tenant_id: str, owner: str, fence: int) -> ToolOperationRecord:
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None or current.owner != owner or current.fence != fence or current.lease_expires_at is None or current.lease_expires_at <= datetime.now(timezone.utc):
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        return current

    async def put_bytes(self, *, tenant_id: str, data: bytes, expected_digest: "str | None" = None) -> BlobRef:
        if len(data) > _MAX_SQL_INLINE_BLOB_BYTES:
            raise ValueError("put_stream is required for blobs larger than 4 MiB")
        digest = hashlib.sha256(data).hexdigest()
        if expected_digest is not None and expected_digest != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async def chunks() -> AsyncIterator[bytes]:
            for offset in range(0, len(data), _MAX_SQL_BLOB_CHUNK):
                yield data[offset:offset + _MAX_SQL_BLOB_CHUNK]

        return await self.put_stream(tenant_id=tenant_id, chunks=chunks(), expected_size=len(data), expected_digest=digest)

    async def put_stream(self, *, tenant_id: str, chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> BlobRef:
        if expected_size < 0 or expected_size > _MAX_SQL_BLOB_BYTES or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        from sqlalchemy import insert, select
        blob_table = self._owner.tables["blobs"]
        chunk_table = self._owner.tables["blob_chunks"]
        digest = hashlib.sha256()
        size = 0
        index = 0
        chunk_values: list[dict[str, object]] = []
        try:
            async with self._owner.session_factory() as session:
                async with session.begin():
                    manifest = (await session.execute(select(blob_table.c.payload).where(blob_table.c.namespace_key == self.namespace_key, blob_table.c.tenant_id == tenant_id, blob_table.c.record_id == expected_digest))).mappings().first()
                    if manifest is not None:
                        existing = _decode_payload("blobs", manifest["payload"])
                        if not isinstance(existing, BlobRef) or existing.size != expected_size:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        return existing
                    async for chunk in chunks:
                        chunk = bytes(chunk)
                        if not 1 <= len(chunk) <= _MAX_SQL_BLOB_CHUNK or size + len(chunk) > expected_size:
                            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                        digest.update(chunk)
                        size += len(chunk)
                        now = datetime.now(timezone.utc)
                        chunk_values.append({"namespace_key": self.namespace_key, "tenant_id": tenant_id, "record_id": f"{expected_digest}:{index}", "digest": expected_digest, "chunk_index": index, "chunk_key": _composite_key(expected_digest, str(index)), "content": chunk, "sequence": index, "revision": 0, "status": "", "payload": _encode_payload(chunk), "created_at": now, "updated_at": now})
                        if len(chunk_values) >= _SQL_CHUNK_INSERT_ROWS:
                            await session.execute(insert(chunk_table).values(chunk_values))
                            chunk_values.clear()
                        index += 1
                    if size != expected_size or digest.hexdigest() != expected_digest:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if chunk_values:
                        await session.execute(insert(chunk_table).values(chunk_values))
                    ref = BlobRef(tenant_id, expected_digest, size, f"sql:{self.namespace}:{expected_digest}")
                    now = datetime.now(timezone.utc)
                    await session.execute(insert(blob_table).values(namespace_key=self.namespace_key, tenant_id=tenant_id, record_id=expected_digest, digest=expected_digest, size=size, sequence=0, revision=0, status="COMPLETED", payload=_encode_payload(ref), created_at=now, updated_at=now))
        except Exception as error:
            if not _is_integrity(error):
                raise
            existing = await self._get(expected_digest, tenant_id=tenant_id, table=blob_table)
            if isinstance(existing, BlobRef) and existing.size == expected_size:
                return existing
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        _logger.debug("SQL blob committed: namespace=%s tenant=%s digest=%s size=%s", self.namespace, tenant_id, expected_digest, expected_size)
        return ref

    async def stat(self, ref: BlobRef, *, tenant_id: str) -> BlobRef | None:
        value = await self._get(ref.digest, tenant_id=tenant_id, table=self._owner.tables["blobs"])
        if value is None:
            return None
        if not isinstance(value, BlobRef):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    def open(self, ref: BlobRef, *, tenant_id: str) -> AsyncIterator[bytes]:
        return self._open_blob(ref, tenant_id)

    async def _open_blob(self, ref: BlobRef, tenant_id: str) -> AsyncIterator[bytes]:
        from sqlalchemy import select
        manifest = await self._get(ref.digest, tenant_id=tenant_id, table=self._owner.tables["blobs"])
        if not isinstance(manifest, BlobRef):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with self._owner.session_factory() as session:
            rows = (await session.execute(select(self._owner.tables["blob_chunks"].c.payload).where(self._owner.tables["blob_chunks"].c.namespace_key == self.namespace_key, self._owner.tables["blob_chunks"].c.tenant_id == tenant_id, self._owner.tables["blob_chunks"].c.record_id.like(f"{ref.digest}:%")).order_by(self._owner.tables["blob_chunks"].c.sequence))).scalars()
            total = 0
            digest = hashlib.sha256()
            for payload in rows:
                chunk = _decode_payload("blob_chunks", payload)
                if not isinstance(chunk, bytes):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                total += len(chunk)
                digest.update(chunk)
                yield chunk
        if total != manifest.size or digest.hexdigest() != manifest.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


class _SqlRuntimeOwner:
    def __init__(self, database: StorageDatabase, session_factory: "async_sessionmaker[AsyncSession]", tables: SqlRuntimeTables, *, backend: RuntimeBackend, namespace: str, atomic_domain_id: str) -> None:
        self.database = database
        self.session_factory = session_factory
        self.tables = tables.tables
        self.backend = backend
        self.namespace = namespace
        self.namespace_key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        self.atomic_domain_id = atomic_domain_id

    def table_name(self, table: "Table") -> str:
        for name, value in self.tables.items():
            if value is table:
                return name
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _encode_payload(value: object) -> JsonValue:
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if is_dataclass(value):
        return {"type": type(value).__name__, "value": json.loads(canonical_json_bytes(asdict(value)))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _decode_payload(table_name: str, value: object) -> object:
    if not isinstance(value, dict) or "type" not in value:
        return value
    kind = value["type"]
    raw = value.get("value")
    if kind == "bytes":
        if not isinstance(raw, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return base64.b64decode(raw.encode("ascii"))
    if not isinstance(raw, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, f"unknown SQL payload type for {table_name}")
    return decoder(raw)


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _enum(enum_type: "type[object]", value: object) -> object:
    return enum_type(str(value))


def _session_record(value: "dict[str, JsonValue]") -> SessionRecord:
    return SessionRecord(session_id=str(value["session_id"]), tenant_id=str(value["tenant_id"]), owner_principal_id=str(value["owner_principal_id"]), binding_digest=str(value["binding_digest"]), status=_enum(SessionStatus, value["status"]), revision=int(value["revision"]), resource_generation=int(value["resource_generation"]), cwd=None if value.get("cwd") is None else str(value["cwd"]), metadata=value.get("metadata", {}), created_at=_utc(value["created_at"]), updated_at=_utc(value["updated_at"]), closed_at=None if value.get("closed_at") is None else _utc(value["closed_at"]), head_execution_id=None if value.get("head_execution_id") is None else str(value["head_execution_id"]))


def _execution_record(value: "dict[str, JsonValue]") -> ExecutionRecord:
    return ExecutionRecord(execution_id=str(value["execution_id"]), tenant_id=str(value["tenant_id"]), session_id=None if value.get("session_id") is None else str(value["session_id"]), binding_digest=str(value["binding_digest"]), parent_execution_id=None if value.get("parent_execution_id") is None else str(value["parent_execution_id"]), root_execution_id=str(value["root_execution_id"]), source_execution_id=None if value.get("source_execution_id") is None else str(value["source_execution_id"]), base_execution_id=None if value.get("base_execution_id") is None else str(value["base_execution_id"]), lineage_kind=_enum(ExecutionLineageKind, value.get("lineage_kind", "RUN")), status=_enum(ExecutionStatus, value["status"]), revision=int(value["revision"]), event_sequence=int(value["event_sequence"]), agent_run_sequence=int(value.get("agent_run_sequence", 0)), result_ref=None if value.get("result_ref") is None else str(value["result_ref"]), result_digest=None if value.get("result_digest") is None else str(value["result_digest"]), error_code=None if value.get("error_code") is None else str(value["error_code"]), safe_error_details=value.get("safe_error_details", {}), created_at=_utc(value["created_at"]), updated_at=_utc(value["updated_at"]), memory_namespace=None if value.get("memory_namespace") is None else str(value["memory_namespace"]))


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
    return TaskNodeView(str(value["graph_id"]), str(value["task_id"]), tuple(value.get("dependencies", [])), _enum(TaskStatus, value["status"]), None if value.get("owner") is None else str(value["owner"]), int(value["fence"]), None if value.get("lease_expires_at") is None else _utc(value["lease_expires_at"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), None if value.get("error_digest") is None else str(value["error_digest"]), None if value.get("execution_id") is None else str(value["execution_id"]))


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


async def open_sql_runtime(database: StorageDatabase, *, session_factory: "async_sessionmaker[AsyncSession]", backend: RuntimeBackend, namespace: str, deployment_id: str, tables: SqlRuntimeTables) -> RuntimePersistence:
    if backend not in {RuntimeBackend.SQLITE, RuntimeBackend.MYSQL, RuntimeBackend.POSTGRESQL}:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    atomic_domain_id = hashlib.sha256(f"{backend.value}{namespace}{deployment_id}{database.schema_manifest_digest}".encode("utf-8")).hexdigest()
    owner = _SqlRuntimeOwner(database, session_factory, tables, backend=backend, namespace=namespace, atomic_domain_id=atomic_domain_id)
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
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _composite_key(*values: str) -> str:
    return hashlib.sha256("\x00".join(values).encode("utf-8")).hexdigest()


def _record_time(record: object) -> datetime:
    value = record.updated_at if isinstance(record, (SessionRecord, ExecutionRecord, IdempotencyRecord, EvaluationRecord, MemoryRecord, OperationLedgerRecord, ToolOperationRecord)) else record.created_at if isinstance(record, (ResultRecord, ArtifactRecord, ApprovalRecord, ExternalResultRecord)) else datetime.now(timezone.utc)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _revision(record: object) -> int:
    if isinstance(record, ExecutionRecord):
        return record.revision
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
    from sqlalchemy.exc import IntegrityError
    return isinstance(error, IntegrityError)


def _is_retryable_transaction(error: BaseException) -> bool:
    if _is_integrity(error):
        return True
    from sqlalchemy.exc import DBAPIError
    if not isinstance(error, DBAPIError):
        return False
    original = error.orig
    message = str(original).lower()
    return any(token in message for token in ("40001", "40p01", "1205", "1213", "deadlock", "database is locked"))


__all__ = ["open_sql_runtime"]
