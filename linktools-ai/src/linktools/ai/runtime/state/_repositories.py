#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private database-current Runtime repositories."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from linktools.core import environ

from ...core import (
    ApprovalDecision,
    ApprovalStatus,
    EvaluationStatus,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    OperationLedgerInput,
    OperationLedgerRecord,
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
    validate_tenant_id,
)
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, SqlStorageContext, namespace_key
from .._tool import ToolOperationRecord
from ._contracts import (
    ApprovalRecord,
    ArtifactRecord,
    ConversationCursor,
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
    ExternalCallRecord,
    IdempotencyRecord,
    MemoryRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryConversationIntent,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    ResultRecord,
    SessionRecord,
)
from ._plan import RuntimeDomain

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager
    from typing import Protocol

    from sqlalchemy import MetaData, Table
    from sqlalchemy.ext.asyncio import AsyncSession

    from ...task import (
        TaskGraph,
        TaskGraphView,
        TaskLease,
        TaskNode,
        TaskNodeView,
        TaskTerminalRecord,
    )


    class _SqlTransactionProtocol(Protocol):
        def mutation(self) -> "AbstractAsyncContextManager[None]": ...

        def current_session(self) -> "AsyncSession": ...

        def mark_changed(self) -> None: ...


_logger = environ.get_logger("ai.runtime.state.repositories")


class _SqlRepositoryBase:
    def __init__(
        self,
        context: SqlStorageContext,
        metadata: "MetaData",
        *,
        namespace: str,
        tenant_id: str,
        owner_domain: RuntimeDomain,
        transaction: "_SqlTransactionProtocol",
    ) -> None:
        self._context = context
        self._metadata = metadata
        self._namespace_key = namespace_key(namespace)
        self._tenant_id = tenant_id
        validate_tenant_id(tenant_id)
        self._owner_domain = owner_domain
        self._transaction = transaction
        self._closed = False

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed or self._context.closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)

    def _check_tenant(self, tenant_id: str) -> None:
        validate_tenant_id(tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)

    def _table(self, name: str) -> "Table":
        try:
            return self._metadata.tables[name]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING) from error

    def _where(self, table: "Table", **values: object) -> tuple[object, ...]:
        clauses = [table.c.namespace_key == self._namespace_key, table.c.tenant_id == self._tenant_id]
        clauses.extend(table.c[name] == value for name, value in values.items())
        return tuple(clauses)

    @asynccontextmanager
    async def _read_session(self) -> AsyncIterator["AsyncSession"]:
        self._ensure_open()
        try:
            session = self._transaction.current_session()
        except RuntimeError:
            async with self._context.sessions() as session:
                yield session
        else:
            yield session

    @asynccontextmanager
    async def _mutation(self) -> AsyncIterator["AsyncSession"]:
        self._ensure_open()
        async with self._transaction.mutation():
            yield self._transaction.current_session()

    async def _one(self, session: "AsyncSession", table: "Table", *where: object) -> Mapping[str, object] | None:
        from sqlalchemy import select

        return (await session.execute(select(table).where(*where).limit(1))).mappings().first()

    async def _insert(self, session: "AsyncSession", table: "Table", values: Mapping[str, object]) -> bool:
        from sqlalchemy.exc import IntegrityError

        try:
            async with session.begin_nested():
                await session.execute(table.insert().values(**values))
                await session.flush()
        except IntegrityError as error:
            if self._context.dialect.classify_integrity_error(error).value != "unique_conflict":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            return False
        self._transaction.mark_changed()
        return True

    def _values(self, identity: Mapping[str, object]) -> dict[str, object]:
        return {"namespace_key": self._namespace_key, "tenant_id": self._tenant_id, **identity}


class _SqlSessionRepository(_SqlRepositoryBase):
    def _from_row(self, row: Mapping[str, object]) -> SessionRecord:
        return SessionRecord(
            str(row["session_id"]), self._tenant_id, str(row["owner_principal_id"]), str(row["binding_digest"]),
            SessionStatus(str(row["status"])), int(row["revision"]), int(row["resource_generation"]),
            None if row["cwd"] is None else str(row["cwd"]), row["metadata_json"] or {}, _utc(row["created_at"]),
            _utc(row["updated_at"]), None if row["closed_at"] is None else _utc(row["closed_at"]),
            None if row["continuation_step_run_id"] is None else ConversationCursor(str(row["continuation_step_run_id"])),
        )

    def _record_values(self, record: SessionRecord) -> dict[str, object]:
        return self._values({"session_id": record.session_id}) | {
            "owner_principal_id": record.owner_principal_id,
            "binding_digest": record.binding_digest,
            "status": record.status.value,
            "revision": record.revision,
            "resource_generation": record.resource_generation,
            "cwd": record.cwd,
            "metadata_json": dict(record.metadata),
            "continuation_step_run_id": None if record.continuation is None else record.continuation.step_run_id,
            "closed_at": record.closed_at,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    async def create(self, record: SessionRecord) -> SessionRecord:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            if await self._insert(session, self._table("runtime_sessions"), self._record_values(record)):
                return record
            raise AIError(ErrorCode.SESSION_CONFLICT)

    async def get_header(self, session_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._check_tenant(tenant_id)
        record = await self.get(session_id, tenant_id=tenant_id)
        return None if record is None else ResourceRef(ResourceKind.SESSION, session_id, tenant_id, record.owner_principal_id)

    async def get(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_sessions")
            row = await self._one(session, table, *self._where(table, session_id=session_id))
        return None if row is None else self._from_row(row)

    async def list(self, *, tenant_id: str, owner_principal_id: str | None = None) -> tuple[SessionRecord, ...]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        async with self._read_session() as session:
            table = self._table("runtime_sessions")
            clauses = list(self._where(table))
            if owner_principal_id is not None:
                clauses.append(table.c.owner_principal_id == owner_principal_id)
            rows = (await session.execute(select(table).where(*clauses).order_by(table.c.session_id))).mappings().all()
        return tuple(self._from_row(row) for row in rows)

    async def compare_and_swap(self, session_id: str, *, tenant_id: str, expected_revision: int, next_record: SessionRecord) -> SessionRecord:
        self._check_tenant(tenant_id)
        self._check_tenant(next_record.tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_sessions")
            values = self._record_values(next_record)
            for name in ("namespace_key", "tenant_id", "session_id", "created_at"):
                values.pop(name, None)
            result = await session.execute(update(table).where(*self._where(table, session_id=session_id, revision=expected_revision)).values(**values))
            if result.rowcount != 1:
                raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
            self._transaction.mark_changed()
            row = await self._one(session, table, *self._where(table, session_id=session_id))
        if row is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._from_row(row)

    async def advance_continuation(self, session_id: str, *, tenant_id: str, expected: ConversationCursor | None, next_cursor: ConversationCursor) -> SessionRecord:
        self._check_tenant(tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_sessions")
            row = await self._one(session, table, *self._where(table, session_id=session_id))
            if row is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = self._from_row(row)
            if current.status is not SessionStatus.OPEN or current.continuation != expected:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(current, continuation=next_cursor, revision=current.revision + 1, updated_at=datetime.now(timezone.utc))
            result = await session.execute(
                update(table)
                .where(*self._where(table, session_id=session_id, revision=current.revision, status=SessionStatus.OPEN.value, continuation_step_run_id=None if expected is None else expected.step_run_id))
                .values(continuation_step_run_id=next_cursor.step_run_id, revision=updated.revision, updated_at=updated.updated_at)
            )
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
        return updated


class _SqlIdempotencyRepository(_SqlRepositoryBase):
    def __init__(self, *args: object, runtime_domain: RuntimeDomain, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._runtime_domain = runtime_domain

    def _from_row(self, row: Mapping[str, object]) -> IdempotencyRecord:
        return IdempotencyRecord(
            self._tenant_id, self._runtime_domain, str(row["scope"]), str(row["key_hash"]),
            str(row["request_digest"]), ResourceKind(str(row["resource_kind"])), str(row["resource_id"]),
            IdempotencyStatus(str(row["status"])), None if row["result_digest"] is None else str(row["result_digest"]),
            None if row["error_code"] is None else str(row["error_code"]), _utc(row["created_at"]), _utc(row["updated_at"]),
        )

    def _record_values(self, record: IdempotencyRecord) -> dict[str, object]:
        return self._values({"runtime_domain": self._runtime_domain.value, "scope": record.scope, "key_hash": record.key_hash}) | {
            "request_digest": record.request_digest,
            "resource_kind": record.resource_kind.value,
            "resource_id": record.resource_id,
            "status": record.status.value,
            "result_digest": record.result_digest,
            "error_code": record.error_code,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord:
        self._check_tenant(record.tenant_id)
        expected_kind = {
            RuntimeDomain.EXECUTION: ResourceKind.EXECUTION,
            RuntimeDomain.EVALUATION: ResourceKind.EVALUATION,
        }.get(self._runtime_domain)
        if record.tenant_id != self._tenant_id or record.runtime_domain is not self._runtime_domain or record.resource_kind is not expected_kind:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        async with self._mutation() as session:
            table = self._table("runtime_idempotency")
            if await self._insert(session, table, self._record_values(record)):
                return record
            row = await self._one(session, table, *self._where(table, runtime_domain=self._runtime_domain.value, scope=record.scope, key_hash=record.key_hash))
        if row is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = self._from_row(row)
        if current.request_digest != record.request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        return current

    async def get(self, scope: str, key_hash: str, *, tenant_id: str) -> IdempotencyRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_idempotency")
            row = await self._one(session, table, *self._where(table, runtime_domain=self._runtime_domain.value, scope=scope, key_hash=key_hash))
        return None if row is None else self._from_row(row)

    async def list_by_resource(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str) -> tuple[IdempotencyRecord, ...]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        async with self._read_session() as session:
            table = self._table("runtime_idempotency")
            rows = (await session.execute(select(table).where(*self._where(table, runtime_domain=self._runtime_domain.value, resource_kind=resource_kind.value, resource_id=resource_id)).order_by(table.c.scope, table.c.key_hash))).mappings().all()
        return tuple(self._from_row(row) for row in rows)

    async def compare_and_swap(self, scope: str, key_hash: str, *, tenant_id: str, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord:
        from sqlalchemy import update

        self._check_tenant(tenant_id)
        if next_record.tenant_id != self._tenant_id or next_record.runtime_domain is not self._runtime_domain or next_record.scope != scope or next_record.key_hash != key_hash:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        async with self._mutation() as session:
            table = self._table("runtime_idempotency")
            values = self._record_values(next_record)
            for name in ("namespace_key", "tenant_id", "runtime_domain", "scope", "key_hash"):
                values.pop(name, None)
            result = await session.execute(update(table).where(*self._where(table, runtime_domain=self._runtime_domain.value, scope=scope, key_hash=key_hash, status=expected_status.value)).values(**values))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
            row = await self._one(session, table, *self._where(table, runtime_domain=self._runtime_domain.value, scope=scope, key_hash=key_hash))
        if row is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._from_row(row)


class _SqlExecutionRepository(_SqlRepositoryBase):
    def __init__(self, *args: object, idempotency: _SqlIdempotencyRepository, events: "_SqlEventRepository", operations: "_SqlOperationRepository", **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._idempotency = idempotency
        self._events = events
        self._operations = operations

    def _from_row(self, row: Mapping[str, object]) -> ExecutionRecord:
        return ExecutionRecord(
            str(row["execution_id"]), self._tenant_id, None if row["session_id"] is None else str(row["session_id"]),
            str(row["binding_digest"]), None if row["parent_execution_id"] is None else str(row["parent_execution_id"]),
            str(row["root_execution_id"]), None if row["source_execution_id"] is None else str(row["source_execution_id"]),
            None if row["base_execution_id"] is None else str(row["base_execution_id"]),
            ExecutionLineageKind(str(row["lineage_kind"])), ExecutionStatus(str(row["status"])), int(row["revision"]),
            int(row["event_sequence"]), int(row["agent_run_sequence"]), None if row["error_code"] is None else str(row["error_code"]),
            row["safe_error_details_json"] or {}, _utc(row["created_at"]), _utc(row["updated_at"]),
            None if row["memory_scope"] is None else str(row["memory_scope"]),
            None if row["conversation_step_run_id"] is None else str(row["conversation_step_run_id"]),
        )

    def _record_values(self, record: ExecutionRecord) -> dict[str, object]:
        return self._values({"execution_id": record.execution_id}) | {
            "session_id": record.session_id,
            "binding_digest": record.binding_digest,
            "parent_execution_id": record.parent_execution_id,
            "root_execution_id": record.root_execution_id,
            "source_execution_id": record.source_execution_id,
            "base_execution_id": record.base_execution_id,
            "lineage_kind": record.lineage_kind.value,
            "status": record.status.value,
            "revision": record.revision,
            "event_sequence": record.event_sequence,
            "agent_run_sequence": record.agent_run_sequence,
            "error_code": record.error_code,
            "safe_error_details_json": dict(record.safe_error_details),
            "memory_scope": record.memory_scope,
            "conversation_step_run_id": record.conversation_step_run_id,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    async def create(self, record: ExecutionRecord) -> ExecutionRecord:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            if await self._insert(session, self._table("runtime_executions"), self._record_values(record)):
                return record
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._check_tenant(tenant_id)
        return None if await self.get(execution_id, tenant_id=tenant_id) is None else ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_executions")
            row = await self._one(session, table, *self._where(table, execution_id=execution_id))
        return None if row is None else self._from_row(row)

    async def compare_and_swap(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: ExecutionRecord) -> ExecutionRecord:
        self._check_tenant(tenant_id)
        self._check_tenant(next_record.tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_executions")
            values = self._record_values(next_record)
            for name in ("namespace_key", "tenant_id", "execution_id", "created_at"):
                values.pop(name, None)
            result = await session.execute(update(table).where(*self._where(table, execution_id=execution_id, revision=expected_revision)).values(**values))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
            row = await self._one(session, table, *self._where(table, execution_id=execution_id))
        if row is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._from_row(row)

    async def list_by_session(self, session_id: str, *, tenant_id: str, statuses: frozenset[ExecutionStatus] | None = None) -> tuple[ExecutionRecord, ...]:
        self._check_tenant(tenant_id)
        return await self._list("session_id", session_id, statuses)

    async def list_children(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionRecord, ...]:
        self._check_tenant(tenant_id)
        return await self._list("parent_execution_id", execution_id, None)

    async def _list(self, field: str, value: str, statuses: frozenset[ExecutionStatus] | None) -> tuple[ExecutionRecord, ...]:
        from sqlalchemy import select

        async with self._read_session() as session:
            table = self._table("runtime_executions")
            clauses = list(self._where(table, **{field: value}))
            if statuses:
                clauses.append(table.c.status.in_([item.value for item in statuses]))
            rows = (await session.execute(select(table).where(*clauses).order_by(table.c.execution_id))).mappings().all()
        return tuple(self._from_row(row) for row in rows)

    async def reserve_start(self, reservation: ExecutionStartReservation) -> ExecutionStartReservationResult:
        self._check_tenant(reservation.execution.tenant_id)
        self._check_tenant(reservation.idempotency.tenant_id)
        if (
            reservation.execution.tenant_id != self._tenant_id
            or reservation.idempotency.tenant_id != self._tenant_id
            or reservation.idempotency.runtime_domain is not RuntimeDomain.EXECUTION
            or reservation.idempotency.resource_kind is not ResourceKind.EXECUTION
            or reservation.idempotency.resource_id != reservation.execution.execution_id
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        async with self._mutation() as session:
            idempotency_table = self._table("runtime_idempotency")
            identity_row = await self._one(
                session,
                idempotency_table,
                *self._where(
                    idempotency_table,
                    runtime_domain=RuntimeDomain.EXECUTION.value,
                    scope=reservation.idempotency.scope,
                    key_hash=reservation.idempotency.key_hash,
                ),
            )
            if identity_row is not None:
                identity = self._idempotency._from_row(identity_row)
                if (
                    identity.tenant_id != reservation.idempotency.tenant_id
                    or identity.runtime_domain is not RuntimeDomain.EXECUTION
                    or identity.resource_kind is not ResourceKind.EXECUTION
                    or identity.resource_id != reservation.execution.execution_id
                    or identity.request_digest != reservation.idempotency.request_digest
                ):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                execution_row = await self._one(
                    session,
                    self._table("runtime_executions"),
                    *self._where(self._table("runtime_executions"), execution_id=identity.resource_id),
                )
                if execution_row is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return ExecutionStartReservationResult(self._from_row(execution_row), identity, False)
            if not await self._insert(session, idempotency_table, self._idempotency._record_values(reservation.idempotency)):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if not await self._insert(session, self._table("runtime_executions"), self._record_values(reservation.execution)):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        return ExecutionStartReservationResult(reservation.execution, reservation.idempotency, True)

    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord:
        self._check_tenant(claim.tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            execution_table = self._table("runtime_executions")
            identity_table = self._table("runtime_idempotency")
            execution_row = await self._one(session, execution_table, *self._where(execution_table, execution_id=claim.execution_id))
            identity_row = await self._one(session, identity_table, *self._where(identity_table, runtime_domain=RuntimeDomain.EXECUTION.value, scope=claim.scope, key_hash=claim.key_hash))
            if execution_row is None or identity_row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = self._from_row(execution_row)
            identity = self._idempotency._from_row(identity_row)
            if current.status is not ExecutionStatus.PENDING_START or current.revision != claim.expected_revision or current.event_sequence != claim.expected_event_sequence or identity.status is not IdempotencyStatus.RESERVED or identity.resource_id != claim.execution_id or identity.request_digest != claim.request_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            started = replace(current, status=ExecutionStatus.STARTED, revision=current.revision + 1, event_sequence=current.event_sequence + 1, agent_run_sequence=1, updated_at=claim.started_at)
            values = self._record_values(started)
            for name in ("namespace_key", "tenant_id", "execution_id", "created_at"):
                values.pop(name, None)
            result = await session.execute(update(execution_table).where(*self._where(execution_table, execution_id=claim.execution_id, revision=claim.expected_revision, event_sequence=claim.expected_event_sequence)).values(**values))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
            await self._idempotency.compare_and_swap(claim.scope, claim.key_hash, tenant_id=claim.tenant_id, expected_status=IdempotencyStatus.RESERVED, next_record=replace(identity, status=IdempotencyStatus.STARTED, updated_at=claim.started_at))
            await self._events.append(claim.execution_id, tenant_id=claim.tenant_id, expected_sequence=claim.expected_event_sequence, event_type=ExecutionEventType.EXECUTION_STARTED, payload={})
            return started

    async def claim_next_agent_run(self, execution_id: str, *, tenant_id: str, expected_revision: int, expected_agent_run_sequence: int) -> ExecutionRecord:
        self._check_tenant(tenant_id)
        current = await self.get(execution_id, tenant_id=tenant_id)
        if current is None or current.status is not ExecutionStatus.STARTED or current.agent_run_sequence != expected_agent_run_sequence:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self.compare_and_swap(execution_id, tenant_id=tenant_id, expected_revision=expected_revision, next_record=replace(current, revision=current.revision + 1, agent_run_sequence=current.agent_run_sequence + 1, updated_at=datetime.now(timezone.utc)))

    async def mark_start_unknown(self, commit: ExecutionStartUnknownCommit) -> ExecutionRecord:
        self._check_tenant(commit.tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_executions")
            row = await self._one(session, table, *self._where(table, execution_id=commit.execution_id))
            identity = await self._idempotency.get(commit.scope, commit.key_hash, tenant_id=commit.tenant_id)
            if row is None or identity is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = self._from_row(row)
            if current.status is not ExecutionStatus.STARTED or current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence or identity.status is not IdempotencyStatus.STARTED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            unknown = replace(current, status=ExecutionStatus.START_UNKNOWN, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.occurred_at)
            values = self._record_values(unknown)
            for name in ("namespace_key", "tenant_id", "execution_id", "created_at"):
                values.pop(name, None)
            result = await session.execute(update(table).where(*self._where(table, execution_id=commit.execution_id, revision=commit.expected_revision, event_sequence=commit.expected_event_sequence)).values(**values))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
            await self._idempotency.compare_and_swap(commit.scope, commit.key_hash, tenant_id=commit.tenant_id, expected_status=IdempotencyStatus.STARTED, next_record=replace(identity, status=IdempotencyStatus.START_UNKNOWN, updated_at=commit.occurred_at))
            await self._events.append(commit.execution_id, tenant_id=commit.tenant_id, expected_sequence=commit.expected_event_sequence, event_type=ExecutionEventType.EXECUTION_START_UNKNOWN, payload={})
            return unknown

    async def request_cancel(self, commit: ExecutionCancelRequestCommit) -> ExecutionRecord:
        self._check_tenant(commit.tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_executions")
            row = await self._one(session, table, *self._where(table, execution_id=commit.execution_id))
            if row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = self._from_row(row)
            operation = await self._operations.get(commit.operation_id, tenant_id=commit.tenant_id)
            if operation is None or operation.status is not OperationStatus.PENDING or operation.execution_id != commit.execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current.status is ExecutionStatus.CANCELLING:
                event_table = self._table("runtime_events")
                event = await self._one(
                    session,
                    event_table,
                    *self._where(
                        event_table,
                        execution_id=commit.execution_id,
                        sequence=current.event_sequence,
                        event_type=ExecutionEventType.CANCEL_REQUESTED.value,
                    ),
                )
                if event is None or event["payload_json"] not in ({}, None):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return current
            if current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} or current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(current, status=ExecutionStatus.CANCELLING, revision=current.revision + 1, event_sequence=current.event_sequence + 1, updated_at=commit.requested_at)
            values = self._record_values(updated)
            for name in ("namespace_key", "tenant_id", "execution_id", "created_at"):
                values.pop(name, None)
            result = await session.execute(update(table).where(*self._where(table, execution_id=commit.execution_id, revision=commit.expected_revision, event_sequence=commit.expected_event_sequence)).values(**values))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
            await self._events.append(commit.execution_id, tenant_id=commit.tenant_id, expected_sequence=commit.expected_event_sequence, event_type=ExecutionEventType.CANCEL_REQUESTED, payload={})
            return updated

    async def advance_event_sequence(self, execution_id: str, *, tenant_id: str, expected_sequence: int) -> ExecutionRecord:
        self._check_tenant(tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_executions")
            row = await self._one(session, table, *self._where(table, execution_id=execution_id))
            if row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = self._from_row(row)
            if current.event_sequence != expected_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            updated = replace(current, revision=current.revision + 1, event_sequence=expected_sequence + 1, updated_at=datetime.now(timezone.utc))
            result = await session.execute(update(table).where(*self._where(table, execution_id=execution_id, revision=current.revision, event_sequence=expected_sequence)).values(revision=updated.revision, event_sequence=updated.event_sequence, updated_at=updated.updated_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
            return updated

    async def commit_terminal(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult:
        self._check_tenant(commit.execution.tenant_id)
        self._check_tenant(commit.result.tenant_id)
        from sqlalchemy import update

        if (
            commit.execution.execution_id != commit.result.execution_id
            or commit.execution.tenant_id != commit.result.tenant_id
            or commit.execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
            or commit.expected_revision < 0
            or commit.expected_event_sequence < 0
            or commit.execution.revision != commit.expected_revision + 1
            or commit.execution.event_sequence != commit.expected_event_sequence + 1
        ):
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        expected_event_type = {
            ExecutionStatus.SUCCEEDED: ExecutionEventType.EXECUTION_SUCCEEDED,
            ExecutionStatus.FAILED: ExecutionEventType.EXECUTION_FAILED,
            ExecutionStatus.CANCELLED: ExecutionEventType.EXECUTION_CANCELLED,
        }[commit.execution.status]
        if commit.terminal_event_type is not expected_event_type:
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        async with self._mutation() as session:
            table = self._table("runtime_executions")
            values = self._record_values(commit.execution)
            values.update({
                "output_schema_id": commit.result.output_schema_id,
                "output_schema_revision": commit.result.output_schema_revision,
                "output_schema_fingerprint": commit.result.output_schema_fingerprint,
                "result_store_id": None if commit.result.object_ref is None else commit.result.object_ref.store_id,
                "result_object_key": None if commit.result.object_ref is None else commit.result.object_ref.key,
                "result_digest": None if commit.result.object_ref is None else commit.result.object_ref.digest,
                "result_size": None if commit.result.object_ref is None else commit.result.object_ref.size,
                "stop_reason": commit.result.stop_reason.value,
                "input_tokens": commit.result.input_tokens,
                "output_tokens": commit.result.output_tokens,
                "total_cost_micros": commit.result.total_cost_micros,
                "result_created_at": commit.result.created_at,
            })
            for name in ("namespace_key", "tenant_id", "execution_id", "created_at"):
                values.pop(name, None)
            result = await session.execute(update(table).where(*self._where(table, execution_id=commit.execution.execution_id, revision=commit.expected_revision, event_sequence=commit.expected_event_sequence)).values(**values))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
            await self._events.append(commit.execution.execution_id, tenant_id=commit.execution.tenant_id, expected_sequence=commit.expected_event_sequence, event_type=commit.terminal_event_type, payload=commit.terminal_event_payload)
            if commit.idempotency is not None:
                identity = await self._idempotency.get(commit.idempotency.scope, commit.idempotency.key_hash, tenant_id=commit.execution.tenant_id)
                if identity is None or identity.resource_kind is not ResourceKind.EXECUTION or identity.resource_id != commit.execution.execution_id or identity.request_digest != commit.idempotency.request_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._idempotency.compare_and_swap(commit.idempotency.scope, commit.idempotency.key_hash, tenant_id=commit.execution.tenant_id, expected_status=commit.idempotency.expected_status, next_record=replace(identity, status=commit.idempotency.next_status, result_digest=commit.idempotency.result_digest, error_code=commit.idempotency.error_code, updated_at=commit.execution.updated_at))
            if commit.operation is not None:
                operation = await self._operations.get(commit.operation.operation_id, tenant_id=commit.execution.tenant_id)
                if operation is None or operation.execution_id not in {None, commit.execution.execution_id}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._operations.compare_and_swap(commit.operation.operation_id, tenant_id=commit.execution.tenant_id, expected_status=commit.operation.expected_status, next_record=replace(operation, status=commit.operation.next_status, result_ref=commit.operation.result_ref, result_digest=commit.operation.result_digest, error_code=commit.operation.error_code, updated_at=commit.execution.updated_at))
            return ExecutionTerminalCommitResult(commit.execution, commit.result)

    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_executions")
            row = await self._one(session, table, *self._where(table, execution_id=execution_id))
        if row is None:
            return None
        return _result(row, self._tenant_id)


class _SqlEventRepository(_SqlRepositoryBase):
    async def append(self, execution_id: str, *, tenant_id: str, expected_sequence: int, event_type: ExecutionEventType, payload: object) -> ExecutionEventRecord:
        self._check_tenant(tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_events")
            executions = self._table("runtime_executions")
            execution_row = await self._one(session, executions, *self._where(executions, execution_id=execution_id))
            if execution_row is None or int(execution_row["event_sequence"]) not in {expected_sequence, expected_sequence + 1}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            sequence = expected_sequence + 1
            values = self._values({"execution_id": execution_id, "sequence": sequence}) | {"event_type": event_type.value, "payload_json": payload}
            if not await self._insert(session, table, values):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if int(execution_row["event_sequence"]) == expected_sequence:
                result = await session.execute(
                    update(executions)
                    .where(*self._where(executions, execution_id=execution_id, revision=int(execution_row["revision"]), event_sequence=expected_sequence))
                    .values(event_sequence=expected_sequence + 1, revision=int(execution_row["revision"]) + 1, updated_at=datetime.now(timezone.utc))
                )
                if result.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                self._transaction.mark_changed()
        return ExecutionEventRecord(execution_id, self._tenant_id, sequence, event_type, payload)

    async def list(self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int) -> Page[ExecutionEventRecord]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        async with self._read_session() as session:
            table = self._table("runtime_events")
            rows = (await session.execute(select(table).where(*self._where(table, execution_id=execution_id), table.c.sequence > after_sequence).order_by(table.c.sequence).limit(limit + 1))).mappings().all()
        items = tuple(ExecutionEventRecord(execution_id, self._tenant_id, int(row["sequence"]), ExecutionEventType(str(row["event_type"])), row["payload_json"] or {}) for row in rows[:limit])
        return Page(items, str(items[-1].sequence) if len(rows) > limit and items else None)


class _SqlOperationRepository(_SqlRepositoryBase):
    def _from_row(self, row: Mapping[str, object]) -> OperationLedgerRecord:
        from ...core import OperationKind

        return OperationLedgerRecord(str(row["operation_id"]), self._tenant_id, ResourceKind(str(row["resource_kind"])), str(row["resource_id"]), None if row["execution_id"] is None else str(row["execution_id"]), OperationKind(str(row["operation_kind"])), OperationStatus(str(row["status"])), str(row["request_digest"]), None if row["result_ref"] is None else str(row["result_ref"]), None if row["result_digest"] is None else str(row["result_digest"]), None if row["error_code"] is None else str(row["error_code"]), bool(row["compactable"]), int(row["sequence"]), _utc(row["created_at"]), _utc(row["updated_at"]))

    async def append(self, record: OperationLedgerInput) -> OperationLedgerRecord:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            table = self._table("runtime_operations")
            row = await self._one(session, table, *self._where(table, runtime_domain=self._owner_domain.value, operation_id=record.operation_id))
            if row is not None:
                current = self._from_row(row)
                if (
                    current.resource_kind is not record.resource_kind
                    or current.resource_id != record.resource_id
                    or current.execution_id != record.execution_id
                    or current.operation_kind is not record.operation_kind
                    or current.request_digest != record.request_digest
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return current
            counter = self._table("runtime_operation_counters")
            sequence = await self._context.dialect.upsert_increment(
                session,
                table=counter,
                values=self._values({
                    "runtime_domain": self._owner_domain.value,
                    "resource_kind": record.resource_kind.value,
                    "resource_id": record.resource_id,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                }),
                column="last_sequence",
                index_elements=("namespace_key", "tenant_id", "runtime_domain", "resource_kind", "resource_id"),
            )
            self._transaction.mark_changed()
            values = self._values({"runtime_domain": self._owner_domain.value, "operation_id": record.operation_id}) | {"resource_kind": record.resource_kind.value, "resource_id": record.resource_id, "operation_kind": record.operation_kind.value, "sequence": sequence, "status": record.status.value, "execution_id": record.execution_id, "request_digest": record.request_digest, "result_ref": record.result_ref, "result_digest": record.result_digest, "error_code": record.error_code, "compactable": record.compactable, "created_at": record.created_at, "updated_at": record.updated_at}
            if not await self._insert(session, table, values):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        return OperationLedgerRecord(record.operation_id, record.tenant_id, record.resource_kind, record.resource_id, record.execution_id, record.operation_kind, record.status, record.request_digest, record.result_ref, record.result_digest, record.error_code, record.compactable, sequence, record.created_at, record.updated_at)

    async def get(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_operations")
            row = await self._one(session, table, *self._where(table, runtime_domain=self._owner_domain.value, operation_id=operation_id))
        return None if row is None else self._from_row(row)

    async def compare_and_swap(self, operation_id: str, *, tenant_id: str, expected_status: OperationStatus, next_record: OperationLedgerRecord) -> OperationLedgerRecord:
        from sqlalchemy import update

        self._check_tenant(tenant_id)
        self._check_tenant(next_record.tenant_id)
        async with self._mutation() as session:
            table = self._table("runtime_operations")
            result = await session.execute(update(table).where(*self._where(table, runtime_domain=self._owner_domain.value, operation_id=operation_id, status=expected_status.value)).values(status=next_record.status.value, result_ref=next_record.result_ref, result_digest=next_record.result_digest, error_code=next_record.error_code, updated_at=next_record.updated_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
        return next_record

    async def list_pending(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, limit: int) -> tuple[OperationLedgerRecord, ...]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        if not 1 <= limit <= 1000:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        async with self._read_session() as session:
            table = self._table("runtime_operations")
            rows = (await session.execute(select(table).where(*self._where(table, runtime_domain=self._owner_domain.value, resource_kind=resource_kind.value, resource_id=resource_id), table.c.status.in_([OperationStatus.PENDING.value, OperationStatus.RUNNING.value])).order_by(table.c.sequence).limit(limit))).mappings().all()
        return tuple(self._from_row(row) for row in rows)

    async def compact_terminal(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, through_sequence: int) -> str:
        self._check_tenant(tenant_id)
        from sqlalchemy import delete, select

        async with self._mutation() as session:
            table = self._table("runtime_operations")
            rows = (
                await session.execute(
                    select(table)
                    .where(
                        *self._where(
                            table,
                            runtime_domain=self._owner_domain.value,
                            resource_kind=resource_kind.value,
                            resource_id=resource_id,
                        ),
                        table.c.sequence <= through_sequence,
                        table.c.compactable.is_(True),
                        table.c.status.not_in((OperationStatus.PENDING.value, OperationStatus.RUNNING.value)),
                    )
                    .order_by(table.c.sequence)
                )
            ).mappings().all()
            records = tuple(self._from_row(row) for row in rows)
            digest = canonical_sha256([asdict(record) for record in records])
            if records:
                await session.execute(
                    delete(table).where(
                        *self._where(
                            table,
                            runtime_domain=self._owner_domain.value,
                            resource_kind=resource_kind.value,
                            resource_id=resource_id,
                        ),
                        table.c.sequence <= through_sequence,
                        table.c.compactable.is_(True),
                        table.c.status.not_in((OperationStatus.PENDING.value, OperationStatus.RUNNING.value)),
                    )
                )
                self._transaction.mark_changed()
            return digest


class _SqlMemoryRepository(_SqlRepositoryBase):
    def __init__(self, *args: object, operations: _SqlOperationRepository, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._operations = operations

    async def get_header(self, memory_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._check_tenant(tenant_id)
        return None if await self.get(memory_id, tenant_id=tenant_id) is None else ResourceRef(ResourceKind.MEMORY, memory_id, tenant_id)

    async def get(self, memory_id: str, *, tenant_id: str) -> MemoryRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_memories")
            row = await self._one(session, table, *self._where(table, memory_id=memory_id))
        return None if row is None else _memory(row, self._tenant_id)

    async def put(self, record: MemoryRecord, *, expected_revision: int | None) -> MemoryRecord:
        self._check_tenant(record.tenant_id)
        stored, replayed = await self.put_with_operation(record, expected_revision=expected_revision, operation=None)
        if replayed or stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return stored

    async def put_with_operation(self, record: MemoryRecord, *, expected_revision: int | None, operation: OperationLedgerInput | None) -> tuple[MemoryRecord | None, bool]:
        self._check_tenant(record.tenant_id)
        if operation is not None:
            self._check_tenant(operation.tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_memories")
            if operation is not None:
                existing_operation = await self._operations.get(operation.operation_id, tenant_id=record.tenant_id)
                if existing_operation is not None:
                    if _operation_matches(existing_operation, operation) is False:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    current = await self._one(session, table, *self._where(table, memory_id=record.memory_id))
                    return (None if current is None else _memory(current, self._tenant_id)), True
            current = await self._one(session, table, *self._where(table, memory_id=record.memory_id))
            if current is None and expected_revision not in (None, 0):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current is not None and int(current["revision"]) != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            revision = 0 if current is None else int(current["revision"]) + 1
            values = self._values({"memory_id": record.memory_id}) | {
                "memory_scope_key": record.memory_scope_key,
                "object_store_id": record.content_ref.store_id,
                "object_key": record.content_ref.key,
                "object_digest": record.content_digest,
                "object_size": record.content_ref.size,
                "metadata_json": dict(record.metadata),
                "revision": revision,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
            if current is None:
                if not await self._insert(session, table, values):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            else:
                result = await session.execute(update(table).where(*self._where(table, memory_id=record.memory_id, revision=int(current["revision"]))).values(**{key: value for key, value in values.items() if key not in {"namespace_key", "tenant_id", "memory_id", "created_at"}}))
                if result.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                self._transaction.mark_changed()
            if operation is not None:
                await self._operations.append(operation)
        return replace(record, revision=revision), False

    async def list(self, *, tenant_id: str, memory_scope_key: str, cursor: str | None, limit: int) -> Page[MemoryRecord]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        async with self._read_session() as session:
            table = self._table("runtime_memories")
            clauses = list(self._where(table, memory_scope_key=memory_scope_key))
            if cursor is not None:
                clauses.append(table.c.memory_id > cursor)
            rows = (await session.execute(select(table).where(*clauses).order_by(table.c.memory_id).limit(limit + 1))).mappings().all()
        items = tuple(_memory(row, self._tenant_id) for row in rows[:limit])
        return Page(items, items[-1].memory_id if len(rows) > limit and items else None)

    async def delete(self, memory_id: str, *, tenant_id: str, expected_revision: int) -> None:
        self._check_tenant(tenant_id)
        deleted, replayed = await self.delete_with_operation(memory_id, tenant_id=tenant_id, expected_revision=expected_revision, operation=None)
        if replayed or not deleted:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def delete_with_operation(self, memory_id: str, *, tenant_id: str, expected_revision: int | None, operation: OperationLedgerInput | None) -> tuple[bool, bool]:
        self._check_tenant(tenant_id)
        if operation is not None:
            self._check_tenant(operation.tenant_id)
        from sqlalchemy import delete

        async with self._mutation() as session:
            table = self._table("runtime_memories")
            if operation is not None:
                existing_operation = await self._operations.get(operation.operation_id, tenant_id=tenant_id)
                if existing_operation is not None:
                    if not _operation_matches(existing_operation, operation):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    return False, True
            clauses = list(self._where(table, memory_id=memory_id))
            if expected_revision is not None:
                clauses.append(table.c.revision == expected_revision)
            result = await session.execute(delete(table).where(*clauses))
            if result.rowcount:
                self._transaction.mark_changed()
            if operation is not None:
                await self._operations.append(operation)
            return bool(result.rowcount), False


class _SqlArtifactRepository(_SqlRepositoryBase):
    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            table = self._table("runtime_artifacts")
            values = self._values({"artifact_id": record.artifact_id}) | {
                "execution_id": record.execution_id, "producer": record.producer, "media_type": record.media_type,
                "object_store_id": record.object_ref.store_id, "object_key": record.object_ref.key,
                "object_digest": record.digest, "object_size": record.size, "created_at": record.created_at,
            }
            if await self._insert(session, table, values):
                return record
        current = await self.get_metadata(record.artifact_id, tenant_id=record.tenant_id)
        if current == record:
            return record
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def get_header(self, artifact_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._check_tenant(tenant_id)
        return None if await self.get_metadata(artifact_id, tenant_id=tenant_id) is None else ResourceRef(ResourceKind.ARTIFACT, artifact_id, tenant_id)

    async def get_metadata(self, artifact_id: str, *, tenant_id: str) -> ArtifactRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_artifacts")
            row = await self._one(session, table, *self._where(table, artifact_id=artifact_id))
        return None if row is None else _artifact(row, self._tenant_id)

    async def list_by_execution(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[ArtifactRecord]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        async with self._read_session() as session:
            table = self._table("runtime_artifacts")
            clauses = list(self._where(table, execution_id=execution_id))
            if cursor is not None:
                clauses.append(table.c.artifact_id > cursor)
            rows = (await session.execute(select(table).where(*clauses).order_by(table.c.artifact_id).limit(limit + 1))).mappings().all()
        items = tuple(_artifact(row, self._tenant_id) for row in rows[:limit])
        return Page(items, items[-1].artifact_id if len(rows) > limit and items else None)


def _task_node_from_sql(row: Mapping[str, object]) -> "TaskNode":
    from ...task import TaskNode

    try:
        raw_input = row["input_json"]
        raw_budget = row["budget_cost"]
        input_value = {} if raw_input is None else raw_input
        budget_value = 1 if raw_budget is None else raw_budget
        if not isinstance(budget_value, int) or isinstance(budget_value, bool):
            raise ValueError("budget_cost")
        return TaskNode(
            str(row["node_id"]),
            _task_dependencies_from_sql(row["dependencies_json"]),
            input=input_value,
            budget_cost=budget_value,
        )
    except (AIError, KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _task_dependencies_from_sql(value: object) -> tuple[str, ...]:
    try:
        if type(value) is not list:
            raise ValueError("task dependencies must be a JSON list")
        dependencies = tuple(value)
        if (
            any(not isinstance(item, str) or not item.strip() for item in dependencies)
            or len(set(dependencies)) != len(dependencies)
        ):
            raise ValueError("task dependencies are invalid")
        return dependencies
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


class _SqlTaskRepository(_SqlRepositoryBase):
    async def _graph_rows(self, session: "AsyncSession", graph_id: str, *, lock: bool) -> tuple[Mapping[str, object] | None, tuple[Mapping[str, object], ...]]:
        from sqlalchemy import select

        graphs = self._table("runtime_task_graphs")
        nodes = self._table("runtime_task_nodes")
        graph_query = select(graphs).where(*self._where(graphs, graph_id=graph_id))
        node_query = select(nodes).where(*self._where(nodes, graph_id=graph_id)).order_by(nodes.c.node_id)
        if lock:
            graph_query = graph_query.with_for_update()
            node_query = node_query.with_for_update()
        graph_row = (await session.execute(graph_query)).mappings().first()
        node_rows = tuple((await session.execute(node_query)).mappings().all()) if graph_row is not None else ()
        return graph_row, node_rows

    def _view(self, graph_row: Mapping[str, object], node_rows: tuple[Mapping[str, object], ...]) -> "TaskGraphView":
        from ...task import TaskGraphView

        return TaskGraphView(
            str(graph_row["graph_id"]),
            TaskStatus(str(graph_row["status"])),
            tuple(_task_node_from_sql(row) for row in node_rows),
        )

    def _node_view(self, row: Mapping[str, object]) -> "TaskNodeView":
        from ...task import TaskNodeView

        dependencies = _task_dependencies_from_sql(row["dependencies_json"])
        return TaskNodeView(
            str(row["graph_id"]),
            str(row["node_id"]),
            dependencies,
            TaskStatus(str(row["status"])),
            row["owner"],
            int(row["fence"]),
            _optional_utc(row["lease_expires_at"]),
            row["result_digest"],
            row["error_code"],
            row["error_digest"],
            row["execution_id"],
        )

    async def _locked_graph(self, session: "AsyncSession", graph_id: str) -> Mapping[str, object]:
        from sqlalchemy import select

        table = self._table("runtime_task_graphs")
        row = (
            await session.execute(
                select(table).where(*self._where(table, graph_id=graph_id)).with_for_update()
            )
        ).mappings().first()
        if row is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return row

    async def _locked_node(self, session: "AsyncSession", graph_id: str, node_id: str) -> Mapping[str, object]:
        from sqlalchemy import select

        table = self._table("runtime_task_nodes")
        row = (
            await session.execute(
                select(table)
                .where(*self._where(table, graph_id=graph_id, node_id=node_id))
                .with_for_update()
            )
        ).mappings().first()
        if row is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return row

    async def _locked_dependency_nodes(
        self,
        session: "AsyncSession",
        graph_id: str,
        dependencies: tuple[str, ...],
    ) -> dict[str, Mapping[str, object]]:
        from sqlalchemy import select

        if not dependencies:
            return {}
        table = self._table("runtime_task_nodes")
        rows = (
            await session.execute(
                select(table)
                .where(*self._where(table, graph_id=graph_id), table.c.node_id.in_(dependencies))
                .order_by(table.c.node_id)
                .with_for_update()
            )
        ).mappings().all()
        values = {str(row["node_id"]): row for row in rows}
        if len(values) != len(dependencies):
            raise AIError(ErrorCode.TASK_DEPENDENCY_UNKNOWN)
        return values

    async def _update_node(
        self,
        session: "AsyncSession",
        row: Mapping[str, object],
        **values: object,
    ) -> None:
        from sqlalchemy import update

        table = self._table("runtime_task_nodes")
        result = await session.execute(
            update(table)
            .where(
                *self._where(table, graph_id=str(row["graph_id"]), node_id=str(row["node_id"])),
                table.c.revision == int(row["revision"]),
            )
            .values(**values, revision=int(row["revision"]) + 1)
        )
        if result.rowcount != 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._transaction.mark_changed()

    async def _update_graph(
        self,
        session: "AsyncSession",
        row: Mapping[str, object],
        status: TaskStatus,
    ) -> Mapping[str, object]:
        from sqlalchemy import update

        table = self._table("runtime_task_graphs")
        result = await session.execute(
            update(table)
            .where(
                *self._where(table, graph_id=str(row["graph_id"])),
                table.c.revision == int(row["revision"]),
            )
            .values(status=status.value, revision=int(row["revision"]) + 1)
        )
        if result.rowcount != 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._transaction.mark_changed()
        value = dict(row)
        value["status"] = status.value
        value["revision"] = int(row["revision"]) + 1
        return value

    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._check_tenant(tenant_id)
        return None if await self.get_graph(graph_id, tenant_id=tenant_id) is None else ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def create_graph(self, graph: "TaskGraph", *, tenant_id: str) -> "TaskGraphView":
        self._check_tenant(tenant_id)
        from ...task import TaskGraphView

        async with self._mutation() as session:
            table = self._table("runtime_task_graphs")
            graph_id = graph.graph_id
            if not await self._insert(session, table, self._values({"graph_id": graph_id}) | {"status": TaskStatus.PENDING.value, "revision": 0}):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            node_table = self._table("runtime_task_nodes")
            for node in graph.nodes:
                values = self._values({"graph_id": graph_id, "node_id": node.node_id}) | {
                    "dependencies_json": list(node.dependencies),
                    "input_json": node.input,
                    "budget_cost": node.budget_cost,
                    "status": TaskStatus.PENDING.value,
                    "revision": 0,
                    "owner": None,
                    "fence": 0,
                    "lease_expires_at": None,
                    "execution_id": None,
                    "result_digest": None,
                    "error_code": None,
                    "error_digest": None,
                }
                if not await self._insert(session, node_table, values):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        return TaskGraphView(graph_id, TaskStatus.PENDING, graph.nodes)

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView | None":
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            row, nodes = await self._graph_rows(session, graph_id, lock=False)
        return None if row is None else self._view(row, nodes)

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView":
        self._check_tenant(tenant_id)
        terminal = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
        async with self._mutation() as session:
            graph_row = await self._locked_graph(session, graph_id)
            _, node_rows = await self._graph_rows(session, graph_id, lock=True)
            updated_nodes: list[Mapping[str, object]] = []
            if TaskStatus(str(graph_row["status"])) in terminal:
                for row in node_rows:
                    if TaskStatus(str(row["status"])) not in terminal:
                        await self._update_node(session, row, status=TaskStatus.CANCELLED.value, owner=None, lease_expires_at=None)
                        value = dict(row)
                        value.update(status=TaskStatus.CANCELLED.value, owner=None, lease_expires_at=None, revision=int(row["revision"]) + 1)
                        updated_nodes.append(value)
                    else:
                        updated_nodes.append(row)
                _logger.debug(
                    "SQL task graph reconciled: graph=%s status=%s",
                    graph_id,
                    graph_row["status"],
                )
                return self._view(graph_row, tuple(updated_nodes))

            by_id = {
                str(row["node_id"]): dict(row)
                for row in node_rows
            }
            changed = True
            while changed:
                changed = False
                for node_id in sorted(by_id):
                    row = by_id[node_id]
                    status = TaskStatus(str(row["status"]))
                    if status not in {TaskStatus.PENDING, TaskStatus.READY}:
                        continue
                    dependencies = _task_dependencies_from_sql(
                        row["dependencies_json"]
                    )
                    dependency_rows = tuple(by_id.get(item) for item in dependencies)
                    if any(item is None for item in dependency_rows):
                        raise AIError(ErrorCode.TASK_DEPENDENCY_UNKNOWN)
                    dependency_statuses = tuple(
                        TaskStatus(str(item["status"]))
                        for item in dependency_rows
                        if item is not None
                    )
                    if any(
                        item in {TaskStatus.FAILED, TaskStatus.BLOCKED}
                        for item in dependency_statuses
                    ):
                        digest = canonical_sha256(
                            {
                                "graph_id": graph_id,
                                "node_id": node_id,
                                "reason": "dependency_failed",
                            }
                        )
                        await self._update_node(
                            session,
                            row,
                            status=TaskStatus.BLOCKED.value,
                            error_code=ErrorCode.TASK_DEPENDENCY_FAILED.value,
                            error_digest=digest,
                        )
                        updated = dict(row)
                        updated.update(
                            status=TaskStatus.BLOCKED.value,
                            error_code=ErrorCode.TASK_DEPENDENCY_FAILED.value,
                            error_digest=digest,
                            revision=int(row["revision"]) + 1,
                        )
                    elif status is TaskStatus.PENDING and all(
                        item is TaskStatus.SUCCEEDED
                        for item in dependency_statuses
                    ):
                        await self._update_node(
                            session,
                            row,
                            status=TaskStatus.READY.value,
                        )
                        updated = dict(row)
                        updated.update(
                            status=TaskStatus.READY.value,
                            revision=int(row["revision"]) + 1,
                        )
                    else:
                        continue
                    by_id[node_id] = updated
                    changed = True
            updated_nodes = [by_id[node_id] for node_id in sorted(by_id)]
            next_status = _task_graph_status(
                tuple(TaskStatus(str(row["status"])) for row in updated_nodes)
            )
            if next_status is not TaskStatus(str(graph_row["status"])):
                graph_row = await self._update_graph(session, graph_row, next_status)
            _logger.debug(
                "SQL task graph reconciled: graph=%s status=%s",
                graph_id,
                graph_row["status"],
            )
            return self._view(graph_row, tuple(updated_nodes))

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView":
        self._check_tenant(tenant_id)
        terminal = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
        async with self._mutation() as session:
            graph_row = await self._locked_graph(session, graph_id)
            _, node_rows = await self._graph_rows(session, graph_id, lock=True)
            graph_status = TaskStatus(str(graph_row["status"]))
            if graph_status not in terminal:
                graph_row = await self._update_graph(session, graph_row, TaskStatus.CANCELLED)
            updated_nodes: list[Mapping[str, object]] = []
            for row in node_rows:
                if TaskStatus(str(row["status"])) not in terminal:
                    await self._update_node(session, row, status=TaskStatus.CANCELLED.value, owner=None, lease_expires_at=None)
                    value = dict(row)
                    value.update(status=TaskStatus.CANCELLED.value, owner=None, lease_expires_at=None, revision=int(row["revision"]) + 1)
                    updated_nodes.append(value)
                else:
                    updated_nodes.append(row)
            return self._view(graph_row, tuple(updated_nodes))

    async def claim(self, graph_id: str, node_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> "TaskLease":
        self._check_tenant(tenant_id)
        from ...core import validate_lease_owner
        from ...task import TaskLease

        validate_lease_owner(owner)
        if not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        terminal = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
        async with self._mutation() as session:
            graph_row = await self._locked_graph(session, graph_id)
            if TaskStatus(str(graph_row["status"])) in terminal:
                raise AIError(ErrorCode.TASK_NOT_READY)
            current = await self._locked_node(session, graph_id, node_id)
            status = TaskStatus(str(current["status"]))
            now = datetime.now(timezone.utc)
            expired = status is TaskStatus.RUNNING and _optional_utc(current["lease_expires_at"]) is not None and _optional_utc(current["lease_expires_at"]) <= now
            if status not in {TaskStatus.PENDING, TaskStatus.READY} and not expired:
                raise AIError(ErrorCode.TASK_NOT_READY)
            dependencies = _task_dependencies_from_sql(current["dependencies_json"])
            dependency_rows = await self._locked_dependency_nodes(session, graph_id, dependencies)
            if any(TaskStatus(str(row["status"])) is not TaskStatus.SUCCEEDED for row in dependency_rows.values()):
                raise AIError(ErrorCode.TASK_NOT_READY)
            fence = int(current["fence"]) + 1
            expiry = now + timedelta(seconds=lease_seconds)
            await self._update_node(session, current, status=TaskStatus.RUNNING.value, owner=owner, fence=fence, lease_expires_at=expiry)
        _logger.debug("SQL task node claimed: graph=%s node=%s fence=%s", graph_id, node_id, fence)
        return TaskLease(graph_id, node_id, tenant_id, owner, fence, expiry)

    async def renew(self, lease: "TaskLease", *, tenant_id: str, lease_seconds: int) -> "TaskLease":
        self._check_tenant(tenant_id)
        if lease.tenant_id != tenant_id or not 1 <= lease_seconds <= 3600:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        async with self._mutation() as session:
            await self._locked_graph(session, lease.graph_id)
            current = await self._locked_node(session, lease.graph_id, lease.node_id)
            if not _lease_matches(current, lease):
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            expiry = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            await self._update_node(session, current, lease_expires_at=expiry)
        return replace(lease, lease_expires_at=expiry)

    async def complete(self, lease: "TaskLease", *, tenant_id: str, execution_id: "str | None", result_digest: str) -> "TaskTerminalRecord":
        self._check_tenant(tenant_id)
        from ...task import TaskTerminalRecord

        async with self._mutation() as session:
            await self._locked_graph(session, lease.graph_id)
            current = await self._locked_node(session, lease.graph_id, lease.node_id)
            if not _lease_matches(current, lease):
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            await self._update_node(session, current, status=TaskStatus.SUCCEEDED.value, owner=None, execution_id=execution_id, result_digest=result_digest, lease_expires_at=None)
        return TaskTerminalRecord(lease.node_id, lease.owner, lease.fence, TaskStatus.SUCCEEDED, result_digest, None, None, execution_id=execution_id)

    async def fail(self, lease: "TaskLease", *, tenant_id: str, error_code: str, error_digest: str) -> "TaskTerminalRecord":
        self._check_tenant(tenant_id)
        from ...task import TaskTerminalRecord

        async with self._mutation() as session:
            await self._locked_graph(session, lease.graph_id)
            current = await self._locked_node(session, lease.graph_id, lease.node_id)
            if not _lease_matches(current, lease):
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            await self._update_node(session, current, status=TaskStatus.FAILED.value, owner=None, error_code=error_code, error_digest=error_digest, lease_expires_at=None)
        return TaskTerminalRecord(lease.node_id, lease.owner, lease.fence, TaskStatus.FAILED, None, error_code, error_digest)

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> "tuple[TaskNodeView, ...]":
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        async with self._read_session() as session:
            table = self._table("runtime_task_nodes")
            rows = (await session.execute(select(table).where(*self._where(table, graph_id=graph_id)).order_by(table.c.node_id))).mappings().all()
        return tuple(self._node_view(row) for row in rows)


class _SqlEvaluationRepository(_SqlRepositoryBase):
    async def get_header(self, evaluation_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._check_tenant(tenant_id)
        return None if await self.get(evaluation_id, tenant_id=tenant_id) is None else ResourceRef(ResourceKind.EVALUATION, evaluation_id, tenant_id)

    async def create(self, record: EvaluationRecord) -> EvaluationRecord:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            table = self._table("runtime_evaluations")
            values = self._values({"evaluation_id": record.evaluation_id}) | {"execution_id": record.execution_id, "dataset_id": record.dataset_id, "dataset_revision": record.dataset_revision, "evaluator_id": record.evaluator_id, "evaluator_revision": record.evaluator_revision, "binding_digest": record.binding_digest, "output_schema_fingerprint": record.output_schema_fingerprint, "artifact_digest": record.artifact_digest, "status": record.status.value, "revision": record.revision, "metrics_json": dict(record.metrics), "created_at": record.created_at, "updated_at": record.updated_at}
            if await self._insert(session, table, values):
                return record
        current = await self.get(record.evaluation_id, tenant_id=record.tenant_id)
        if current == record:
            return record
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def get(self, evaluation_id: str, *, tenant_id: str) -> EvaluationRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_evaluations")
            row = await self._one(session, table, *self._where(table, evaluation_id=evaluation_id))
        return None if row is None else _evaluation(row, self._tenant_id)

    async def compare_and_swap(self, evaluation_id: str, *, tenant_id: str, expected_revision: int, next_record: EvaluationRecord) -> EvaluationRecord:
        self._check_tenant(tenant_id)
        self._check_tenant(next_record.tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_evaluations")
            result = await session.execute(update(table).where(*self._where(table, evaluation_id=evaluation_id, revision=expected_revision)).values(execution_id=next_record.execution_id, dataset_id=next_record.dataset_id, dataset_revision=next_record.dataset_revision, evaluator_id=next_record.evaluator_id, evaluator_revision=next_record.evaluator_revision, binding_digest=next_record.binding_digest, output_schema_fingerprint=next_record.output_schema_fingerprint, artifact_digest=next_record.artifact_digest, status=next_record.status.value, revision=next_record.revision, metrics_json=dict(next_record.metrics), updated_at=next_record.updated_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
        return next_record

    async def list_by_execution(self, execution_id: str, *, tenant_id: str) -> tuple[EvaluationRecord, ...]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        async with self._read_session() as session:
            table = self._table("runtime_evaluations")
            rows = (await session.execute(select(table).where(*self._where(table, execution_id=execution_id)).order_by(table.c.evaluation_id))).mappings().all()
        return tuple(_evaluation(row, self._tenant_id) for row in rows)


class _SqlApprovalRepository(_SqlRepositoryBase):
    async def get_header(self, approval_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._check_tenant(tenant_id)
        return None if await self.get(approval_id, tenant_id=tenant_id) is None else ResourceRef(ResourceKind.APPROVAL, approval_id, tenant_id)

    async def create(self, record: ApprovalRecord) -> ApprovalRecord:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            table = self._table("runtime_approvals")
            values = self._values({"approval_id": record.approval_id}) | {"execution_id": record.execution_id, "operation_id": record.operation_id, "status": record.status.value, "idempotency_key_hash": record.idempotency_key_hash, "decision": None if record.decision is None else record.decision.value, "decided_by": record.decided_by, "decision_digest": record.decision_digest, "decided_at": record.decided_at, "created_at": record.created_at}
            if await self._insert(session, table, values):
                return record
        current = await self.get(record.approval_id, tenant_id=record.tenant_id)
        if current == record:
            return record
        raise AIError(ErrorCode.APPROVAL_CONFLICT)

    async def get(self, approval_id: str, *, tenant_id: str) -> ApprovalRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_approvals")
            row = await self._one(session, table, *self._where(table, approval_id=approval_id))
        return None if row is None else _approval(row, self._tenant_id)

    async def decide(self, approval_id: str, *, tenant_id: str, expected_status: ApprovalStatus, idempotency_key_hash: str, decision: ApprovalDecision, principal_id: str, decision_digest: str, decided_at: datetime) -> ApprovalRecord:
        self._check_tenant(tenant_id)
        current = await self.get(approval_id, tenant_id=tenant_id)
        if current is None or current.status is not expected_status:
            raise AIError(ErrorCode.APPROVAL_CONFLICT)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_approvals")
            status = ApprovalStatus.APPROVED if decision is ApprovalDecision.APPROVE else ApprovalStatus.DENIED
            result = await session.execute(update(table).where(*self._where(table, approval_id=approval_id, status=expected_status.value)).values(status=status.value, idempotency_key_hash=idempotency_key_hash, decision=decision.value, decided_by=principal_id, decision_digest=decision_digest, decided_at=decided_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.APPROVAL_CONFLICT)
            self._transaction.mark_changed()
        return replace(current, status=status, idempotency_key_hash=idempotency_key_hash, decision=decision, decided_by=principal_id, decision_digest=decision_digest, decided_at=decided_at)

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        async with self._read_session() as session:
            table = self._table("runtime_approvals")
            rows = (await session.execute(select(table).where(*self._where(table, execution_id=execution_id, status=ApprovalStatus.PENDING.value)))).mappings().all()
        return tuple(_approval(row, self._tenant_id) for row in rows)


class _SqlExternalCallRepository(_SqlRepositoryBase):
    async def get_header(self, call_id: str, *, tenant_id: str) -> ResourceRef | None:
        self._check_tenant(tenant_id)
        return None if await self.get(call_id, tenant_id=tenant_id) is None else ResourceRef(ResourceKind.EXTERNAL_CALL, call_id, tenant_id)

    async def create_call(self, record: ExternalCallRecord) -> ExternalCallRecord:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            table = self._table("runtime_external_calls")
            values = self._values({"call_id": record.call_id}) | {"execution_id": record.execution_id, "operation_id": record.operation_id, "status": record.status.value, "idempotency_key_hash": record.idempotency_key_hash, "object_store_id": None if record.object_ref is None else record.object_ref.store_id, "object_key": None if record.object_ref is None else record.object_ref.key, "object_digest": None if record.object_ref is None else record.object_ref.digest, "object_size": None if record.object_ref is None else record.object_ref.size, "payload_digest": record.payload_digest, "supplied_at": record.supplied_at, "created_at": record.created_at}
            if await self._insert(session, table, values):
                return record
        current = await self.get(record.call_id, tenant_id=record.tenant_id)
        if current == record:
            return record
        raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)

    async def get(self, call_id: str, *, tenant_id: str) -> ExternalCallRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_external_calls")
            row = await self._one(session, table, *self._where(table, call_id=call_id))
        return None if row is None else _external(row, self._tenant_id)

    async def supply(self, call_id: str, *, tenant_id: str, expected_status: ExternalCallStatus, idempotency_key_hash: str, object_ref: ObjectRef, payload_digest: str, supplied_at: datetime) -> ExternalCallRecord:
        self._check_tenant(tenant_id)
        current = await self.get(call_id, tenant_id=tenant_id)
        if current is None or current.status is not expected_status:
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_external_calls")
            result = await session.execute(update(table).where(*self._where(table, call_id=call_id, status=expected_status.value)).values(status=ExternalCallStatus.SUPPLIED.value, idempotency_key_hash=idempotency_key_hash, object_store_id=object_ref.store_id, object_key=object_ref.key, object_digest=object_ref.digest, object_size=object_ref.size, payload_digest=payload_digest, supplied_at=supplied_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
            self._transaction.mark_changed()
        return replace(current, status=ExternalCallStatus.SUPPLIED, idempotency_key_hash=idempotency_key_hash, object_ref=object_ref, payload_digest=payload_digest, supplied_at=supplied_at)

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ExternalCallRecord, ...]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        async with self._read_session() as session:
            table = self._table("runtime_external_calls")
            rows = (await session.execute(select(table).where(*self._where(table, execution_id=execution_id, status=ExternalCallStatus.PENDING.value)))).mappings().all()
        return tuple(_external(row, self._tenant_id) for row in rows)


class _SqlRecoveryCheckpointRepository(_SqlRepositoryBase):
    async def create(self, record: RecoveryCheckpoint) -> RecoveryCheckpoint:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            table = self._table("runtime_recovery_checkpoints")
            values = self._values({"execution_id": record.execution_id}) | {"step_run_id": record.step_run_id, "agent_run_sequence": record.agent_run_sequence, "state": record.state.value, "handoff_phase": record.handoff_phase.value, "input_json": _input_json(record.input), "terminal_handoff_json": _handoff_json(record.terminal_handoff), "handoff_contract_digest": record.handoff_contract_digest, "pending_operation_id": record.pending_operation_id, "revision": record.revision, "created_at": record.created_at, "updated_at": record.updated_at}
            if await self._insert(session, table, values):
                return record
        current = await self.get(record.execution_id, tenant_id=record.tenant_id)
        if current == record:
            return record
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def get(self, execution_id: str, *, tenant_id: str) -> RecoveryCheckpoint | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_recovery_checkpoints")
            row = await self._one(session, table, *self._where(table, execution_id=execution_id))
        return None if row is None else _checkpoint(row, self._tenant_id)

    async def list(self, *, tenant_id: str) -> tuple[RecoveryCheckpoint, ...]:
        self._check_tenant(tenant_id)
        from sqlalchemy import select

        async with self._read_session() as session:
            table = self._table("runtime_recovery_checkpoints")
            rows = (await session.execute(select(table).where(*self._where(table)).order_by(table.c.execution_id))).mappings().all()
        return tuple(_checkpoint(row, self._tenant_id) for row in rows)

    async def compare_and_swap(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: RecoveryCheckpoint) -> RecoveryCheckpoint:
        self._check_tenant(tenant_id)
        self._check_tenant(next_record.tenant_id)
        from sqlalchemy import update

        async with self._mutation() as session:
            table = self._table("runtime_recovery_checkpoints")
            result = await session.execute(update(table).where(*self._where(table, execution_id=execution_id, revision=expected_revision)).values(step_run_id=next_record.step_run_id, agent_run_sequence=next_record.agent_run_sequence, state=next_record.state.value, handoff_phase=next_record.handoff_phase.value, input_json=_input_json(next_record.input), terminal_handoff_json=_handoff_json(next_record.terminal_handoff), handoff_contract_digest=next_record.handoff_contract_digest, pending_operation_id=next_record.pending_operation_id, revision=next_record.revision, updated_at=next_record.updated_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
        return next_record


class _SqlToolRepository(_SqlRepositoryBase):
    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        self._check_tenant(record.tenant_id)
        async with self._mutation() as session:
            table = self._table("runtime_tool_operations")
            values = self._values({"tool_operation_id": record.tool_operation_id}) | {"step_run_id": record.step_run_id, "tool_call_id": record.tool_call_id, "idempotency_key_hash": record.idempotency_key_hash, "tool_name": record.tool_name, "arguments_hash": record.arguments_hash, "binding_fingerprint": record.binding_fingerprint, "replay_safe": record.replay_safe, "status": record.status.value, "owner": record.owner, "fence": record.fence, "lease_expires_at": record.lease_expires_at, "result_store_id": None if record.result_object_ref is None else record.result_object_ref.store_id, "result_object_key": None if record.result_object_ref is None else record.result_object_ref.key, "result_digest": None if record.result_object_ref is None else record.result_object_ref.digest, "result_size": None if record.result_object_ref is None else record.result_object_ref.size, "error_code": record.error_code, "created_at": record.created_at, "updated_at": record.updated_at}
            if await self._insert(session, table, values):
                return record
        current = await self.get_operation(record.tool_operation_id, tenant_id=record.tenant_id)
        if current == record:
            return record
        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)

    async def get_operation(self, tool_operation_id: str, *, tenant_id: str) -> ToolOperationRecord | None:
        self._check_tenant(tenant_id)
        async with self._read_session() as session:
            table = self._table("runtime_tool_operations")
            row = await self._one(session, table, *self._where(table, tool_operation_id=tool_operation_id))
        return None if row is None else _tool(row, self._tenant_id)

    async def claim(self, tool_operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> ToolOperationRecord:
        self._check_tenant(tenant_id)
        from sqlalchemy import update

        validate_lease_owner(owner)
        _validate_tool_lease(lease_seconds)
        effect_unknown = False
        updated: ToolOperationRecord | None = None
        async with self._mutation() as session:
            table = self._table("runtime_tool_operations")
            row = await self._one(session, table, *self._where(table, tool_operation_id=tool_operation_id))
            if row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = _tool(row, self._tenant_id)
            if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
                raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT)
            now = datetime.now(timezone.utc)
            if current.status is ToolOperationStatus.CLAIMED and current.lease_expires_at is not None and current.lease_expires_at <= now and not current.replay_safe:
                updated = replace(current, status=ToolOperationStatus.EFFECT_UNKNOWN, lease_expires_at=None, updated_at=now)
                effect_unknown = True
                values = {"status": updated.status.value, "lease_expires_at": None, "updated_at": updated.updated_at}
            else:
                if current.owner is not None and current.lease_expires_at is not None and current.lease_expires_at > now:
                    raise AIError(ErrorCode.TASK_OWNER_CONFLICT)
                updated = replace(current, status=ToolOperationStatus.CLAIMED, owner=owner, fence=current.fence + 1, lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now)
                values = {"status": updated.status.value, "owner": updated.owner, "fence": updated.fence, "lease_expires_at": updated.lease_expires_at, "updated_at": updated.updated_at}
            result = await session.execute(
                update(table)
                .where(*self._where(table, tool_operation_id=tool_operation_id, fence=current.fence, status=current.status.value))
                .values(**values)
            )
            if result.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._transaction.mark_changed()
        if effect_unknown:
            _logger.warning("tool effect became unknown: tenant=%s tool_operation=%s", tenant_id, tool_operation_id)
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        if updated is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return updated

    async def renew(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, lease_seconds: int) -> ToolOperationRecord:
        self._check_tenant(tenant_id)
        from sqlalchemy import update

        validate_lease_owner(owner)
        _validate_tool_lease(lease_seconds)
        async with self._mutation() as session:
            table = self._table("runtime_tool_operations")
            row = await self._one(session, table, *self._where(table, tool_operation_id=tool_operation_id))
            if row is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            current = _tool(row, self._tenant_id)
            now = datetime.now(timezone.utc)
            if current.owner != owner or current.fence != fence or current.status is not ToolOperationStatus.CLAIMED or current.lease_expires_at is None or current.lease_expires_at <= now:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            updated = replace(current, lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now)
            result = await session.execute(update(table).where(*self._where(table, tool_operation_id=tool_operation_id, owner=owner, fence=fence, status=ToolOperationStatus.CLAIMED.value)).values(lease_expires_at=updated.lease_expires_at, updated_at=updated.updated_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._transaction.mark_changed()
        return updated

    async def complete(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, result_object_ref: ObjectRef | None) -> ToolOperationRecord:
        self._check_tenant(tenant_id)
        from sqlalchemy import update

        validate_lease_owner(owner)
        async with self._mutation() as session:
            table = self._table("runtime_tool_operations")
            row = await self._one(session, table, *self._where(table, tool_operation_id=tool_operation_id))
            if row is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            current = _tool(row, self._tenant_id)
            now = datetime.now(timezone.utc)
            if current.status is ToolOperationStatus.COMPLETED:
                if current.owner == owner and current.fence == fence and current.result_object_ref == result_object_ref:
                    return current
                raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
            if current.owner != owner or current.fence != fence or current.status is not ToolOperationStatus.CLAIMED or current.lease_expires_at is None or current.lease_expires_at <= now:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            updated = replace(current, status=ToolOperationStatus.COMPLETED, lease_expires_at=None, result_object_ref=result_object_ref, updated_at=now)
            result = await session.execute(update(table).where(*self._where(table, tool_operation_id=tool_operation_id, owner=owner, fence=fence, status=ToolOperationStatus.CLAIMED.value)).values(status=updated.status.value, lease_expires_at=None, result_store_id=None if result_object_ref is None else result_object_ref.store_id, result_object_key=None if result_object_ref is None else result_object_ref.key, result_digest=None if result_object_ref is None else result_object_ref.digest, result_size=None if result_object_ref is None else result_object_ref.size, updated_at=updated.updated_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._transaction.mark_changed()
        return updated

    async def fail(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, error_code: str) -> ToolOperationRecord:
        self._check_tenant(tenant_id)
        from sqlalchemy import update

        validate_lease_owner(owner)
        async with self._mutation() as session:
            table = self._table("runtime_tool_operations")
            row = await self._one(session, table, *self._where(table, tool_operation_id=tool_operation_id))
            if row is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            current = _tool(row, self._tenant_id)
            now = datetime.now(timezone.utc)
            if current.owner != owner or current.fence != fence or current.status is not ToolOperationStatus.CLAIMED or current.lease_expires_at is None or current.lease_expires_at <= now:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            updated = replace(current, status=ToolOperationStatus.FAILED, error_code=error_code, lease_expires_at=None, updated_at=now)
            result = await session.execute(update(table).where(*self._where(table, tool_operation_id=tool_operation_id, owner=owner, fence=fence, status=ToolOperationStatus.CLAIMED.value)).values(status=updated.status.value, lease_expires_at=None, error_code=error_code, updated_at=updated.updated_at))
            if result.rowcount != 1:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            self._transaction.mark_changed()
        return updated


def _validate_tool_lease(lease_seconds: int) -> None:
    if not 1 <= lease_seconds <= 3600:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _optional_utc(value: object) -> datetime | None:
    return None if value is None else _utc(value)


def _object_ref(row: Mapping[str, object], prefix: str = "object_") -> ObjectRef | None:
    values = tuple(row.get(f"{prefix}{name}") for name in ("store_id", "key", "digest", "size"))
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ObjectRef(str(values[0]), str(values[1]), str(values[2]), int(values[3]))


def _lease_matches(row: Mapping[str, object], lease: "TaskLease") -> bool:
    expiry = _optional_utc(row["lease_expires_at"])
    return (
        TaskStatus(str(row["status"])) is TaskStatus.RUNNING
        and row["owner"] == lease.owner
        and int(row["fence"]) == lease.fence
        and expiry is not None
        and expiry > datetime.now(timezone.utc)
    )


def _task_graph_status(statuses: tuple[TaskStatus, ...]) -> TaskStatus:
    terminal = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
    if not statuses:
        return TaskStatus.SUCCEEDED
    if all(status in terminal for status in statuses):
        if TaskStatus.FAILED in statuses:
            return TaskStatus.FAILED
        if TaskStatus.BLOCKED in statuses:
            return TaskStatus.BLOCKED
        if TaskStatus.CANCELLED in statuses:
            return TaskStatus.CANCELLED
        return TaskStatus.SUCCEEDED
    if TaskStatus.RUNNING in statuses:
        return TaskStatus.RUNNING
    if TaskStatus.READY in statuses:
        return TaskStatus.READY
    return TaskStatus.PENDING


def _memory(row: Mapping[str, object], tenant_id: str) -> MemoryRecord:
    reference = _object_ref(row)
    if reference is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return MemoryRecord(str(row["memory_id"]), tenant_id, str(row["memory_scope_key"]), reference, str(row["object_digest"]), row["metadata_json"] or {}, int(row["revision"]), _utc(row["created_at"]), _utc(row["updated_at"]))


def _artifact(row: Mapping[str, object], tenant_id: str) -> ArtifactRecord:
    reference = _object_ref(row)
    if reference is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ArtifactRecord(str(row["artifact_id"]), str(row["execution_id"]), tenant_id, str(row["producer"]), str(row["media_type"]), int(row["object_size"]), str(row["object_digest"]), reference, _utc(row["created_at"]))


def _operation_matches(current: OperationLedgerRecord, candidate: OperationLedgerInput) -> bool:
    return (
        current.tenant_id == candidate.tenant_id
        and current.resource_kind is candidate.resource_kind
        and current.resource_id == candidate.resource_id
        and current.execution_id == candidate.execution_id
        and current.operation_kind is candidate.operation_kind
        and current.request_digest == candidate.request_digest
    )


def _result(row: Mapping[str, object], tenant_id: str) -> ResultRecord | None:
    output_values = tuple(row.get(name) for name in ("output_schema_id", "output_schema_revision", "output_schema_fingerprint", "result_store_id", "result_object_key", "result_digest", "result_size"))
    accounting_values = tuple(row.get(name) for name in ("stop_reason", "input_tokens", "output_tokens", "total_cost_micros", "result_created_at"))
    if all(value is None for value in output_values + accounting_values):
        return None
    if any(value is None for value in accounting_values) or (any(value is None for value in output_values) and not all(value is None for value in output_values)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    object_ref = None
    output_schema_id = None
    output_schema_revision = None
    output_schema_fingerprint = None
    if any(value is not None for value in output_values):
        output_schema_id = str(row["output_schema_id"])
        output_schema_revision = int(row["output_schema_revision"])
        output_schema_fingerprint = str(row["output_schema_fingerprint"])
        object_ref = ObjectRef(str(row["result_store_id"]), str(row["result_object_key"]), str(row["result_digest"]), int(row["result_size"]))
    return ResultRecord(
        str(row["execution_id"]),
        tenant_id,
        output_schema_id,
        output_schema_revision,
        output_schema_fingerprint,
        object_ref,
        StopReason(str(row["stop_reason"])),
        int(row["input_tokens"]),
        int(row["output_tokens"]),
        int(row["total_cost_micros"]),
        _utc(row["result_created_at"]),
    )


def _evaluation(row: Mapping[str, object], tenant_id: str) -> EvaluationRecord:
    return EvaluationRecord(str(row["evaluation_id"]), tenant_id, str(row["execution_id"]), str(row["dataset_id"]), int(row["dataset_revision"]), str(row["evaluator_id"]), int(row["evaluator_revision"]), str(row["binding_digest"]), str(row["output_schema_fingerprint"]), None if row["artifact_digest"] is None else str(row["artifact_digest"]), EvaluationStatus(str(row["status"])), int(row["revision"]), row["metrics_json"] or {}, _utc(row["created_at"]), _utc(row["updated_at"]))


def _approval(row: Mapping[str, object], tenant_id: str) -> ApprovalRecord:
    return ApprovalRecord(str(row["approval_id"]), str(row["execution_id"]), tenant_id, str(row["operation_id"]), ApprovalStatus(str(row["status"])), None if row["idempotency_key_hash"] is None else str(row["idempotency_key_hash"]), None if row["decision"] is None else ApprovalDecision(str(row["decision"])), None if row["decided_by"] is None else str(row["decided_by"]), None if row["decision_digest"] is None else str(row["decision_digest"]), _utc(row["created_at"]), _optional_utc(row["decided_at"]))


def _external(row: Mapping[str, object], tenant_id: str) -> ExternalCallRecord:
    return ExternalCallRecord(str(row["call_id"]), str(row["execution_id"]), tenant_id, str(row["operation_id"]), ExternalCallStatus(str(row["status"])), None if row["idempotency_key_hash"] is None else str(row["idempotency_key_hash"]), _object_ref(row), None if row["payload_digest"] is None else str(row["payload_digest"]), _utc(row["created_at"]), _optional_utc(row["supplied_at"]))


def _tool(row: Mapping[str, object], tenant_id: str) -> ToolOperationRecord:
    return ToolOperationRecord(str(row["tool_operation_id"]), tenant_id, str(row["step_run_id"]), str(row["tool_call_id"]), str(row["idempotency_key_hash"]), str(row["tool_name"]), str(row["arguments_hash"]), str(row["binding_fingerprint"]), bool(row["replay_safe"]), ToolOperationStatus(str(row["status"])), None if row["owner"] is None else str(row["owner"]), int(row["fence"]), _optional_utc(row["lease_expires_at"]), _object_ref(row, "result_"), None if row["error_code"] is None else str(row["error_code"]), _utc(row["created_at"]), _utc(row["updated_at"]))


def _input_json(value: RecoveryExecutionInput) -> dict[str, object]:
    return {
        "version": 2,
        "user_prompt": value.user_prompt,
        "principal_id": value.principal_id,
        "principal_kind": value.principal_kind,
        "session_id": value.session_id,
        "memory_scope": value.memory_scope,
        "agent_id": value.agent_id,
        "binding_digest": value.binding_digest,
        "lineage_kind": value.lineage_kind,
        "parent_execution_id": value.parent_execution_id,
        "root_execution_id": value.root_execution_id,
        "source_execution_id": value.source_execution_id,
        "base_execution_id": value.base_execution_id,
        "idempotency": (
            None
            if value.idempotency is None
            else {
                "scope": value.idempotency.scope,
                "key_hash": value.idempotency.key_hash,
                "request_digest": value.idempotency.request_digest,
            }
        ),
    }


def _handoff_json(value: RecoveryTerminalHandoff | None) -> object:
    return None if value is None else {"source_step_run_id": value.source_step_run_id, "outcome": {"terminal_status": value.outcome.terminal_status.value, "error_code": value.outcome.error_code, "safe_error_details": value.outcome.safe_error_details, "stop_reason": value.outcome.stop_reason.value, "output_schema_id": value.outcome.output_schema_id, "output_schema_revision": value.outcome.output_schema_revision, "output_schema_fingerprint": value.outcome.output_schema_fingerprint, "recovery_object_ref": None if value.outcome.recovery_object_ref is None else {"store_id": value.outcome.recovery_object_ref.store_id, "key": value.outcome.recovery_object_ref.key, "digest": value.outcome.recovery_object_ref.digest, "size": value.outcome.recovery_object_ref.size}, "input_tokens": value.outcome.input_tokens, "output_tokens": value.outcome.output_tokens, "total_cost_micros": value.outcome.total_cost_micros, "terminal_event_type": value.outcome.terminal_event_type.value, "terminal_event_payload": value.outcome.terminal_event_payload, "result_created_at": value.outcome.result_created_at.isoformat()}, "conversation": None if value.conversation is None else {"session_id": value.conversation.session_id, "expected_cursor": None if value.conversation.expected_cursor is None else value.conversation.expected_cursor.step_run_id, "next_cursor": value.conversation.next_cursor.step_run_id}}


def _checkpoint(row: Mapping[str, object], tenant_id: str) -> RecoveryCheckpoint:
    try:
        raw_input = row["input_json"]
        if not isinstance(raw_input, dict):
            raise ValueError("input")
        raw_identity = raw_input.get("idempotency")
        identity = None if raw_identity is None else RecoveryIdempotencyInput(str(raw_identity["scope"]), str(raw_identity["key_hash"]), str(raw_identity["request_digest"]))
        user_prompt = raw_input.get("user_prompt", raw_input.get("prompt"))
        if not isinstance(user_prompt, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        checkpoint_input = RecoveryExecutionInput(
            user_prompt,
            str(raw_input["principal_id"]),
            str(raw_input["principal_kind"]),
            raw_input.get("session_id"),
            raw_input.get("memory_scope"),
            str(raw_input["agent_id"]),
            str(raw_input["binding_digest"]),
            str(raw_input["lineage_kind"]),
            raw_input.get("parent_execution_id"),
            str(raw_input["root_execution_id"]),
            raw_input.get("source_execution_id"),
            raw_input.get("base_execution_id"),
            identity,
        )
        handoff = _handoff_from_json(row["terminal_handoff_json"])
        return RecoveryCheckpoint(str(row["execution_id"]), tenant_id, checkpoint_input, str(row["step_run_id"]), int(row["agent_run_sequence"]), RecoveryCheckpointState(str(row["state"])), RecoveryHandoffPhase(str(row["handoff_phase"])), handoff, None if row["handoff_contract_digest"] is None else str(row["handoff_contract_digest"]), None if row["pending_operation_id"] is None else str(row["pending_operation_id"]), int(row["revision"]), _utc(row["created_at"]), _utc(row["updated_at"]))
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _handoff_from_json(value: object) -> RecoveryTerminalHandoff | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("handoff")
    outcome = value["outcome"]
    if not isinstance(outcome, dict):
        raise ValueError("outcome")
    raw_ref = outcome.get("recovery_object_ref")
    reference = None if raw_ref is None else ObjectRef(str(raw_ref["store_id"]), str(raw_ref["key"]), str(raw_ref["digest"]), int(raw_ref["size"]))
    raw_conversation = value.get("conversation")
    conversation = None if raw_conversation is None else RecoveryConversationIntent(str(raw_conversation["session_id"]), None if raw_conversation.get("expected_cursor") is None else ConversationCursor(str(raw_conversation["expected_cursor"])), ConversationCursor(str(raw_conversation["next_cursor"])))
    terminal = RecoveryTerminalOutcome(ExecutionStatus(str(outcome["terminal_status"])), outcome.get("error_code"), outcome.get("safe_error_details") or {}, StopReason(str(outcome["stop_reason"])), outcome.get("output_schema_id"), outcome.get("output_schema_revision"), outcome.get("output_schema_fingerprint"), reference, int(outcome["input_tokens"]), int(outcome["output_tokens"]), int(outcome["total_cost_micros"]), ExecutionEventType(str(outcome["terminal_event_type"])), outcome.get("terminal_event_payload") or {}, _utc(datetime.fromisoformat(str(outcome["result_created_at"]))))
    return RecoveryTerminalHandoff(terminal, str(value["source_step_run_id"]), conversation)


__all__ = []
