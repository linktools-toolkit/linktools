#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL-backed managed tool-operation repository."""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from ..capability.tool import ToolOperationRecord, ToolStateStore
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.value import ToolOperationStatus
from ..storage.database import SqlSchemaRegistry
from ..storage.names import storage_name

if TYPE_CHECKING:
    from collections.abc import Mapping
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

try:
    from sqlalchemy import BigInteger, Boolean, Column, DateTime, Index, Integer, String, Table, UniqueConstraint, select, update
    from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
    from sqlalchemy.ext.asyncio import async_sessionmaker as _async_sessionmaker
except ModuleNotFoundError as error:
    if error.name != "sqlalchemy":
        raise
    BigInteger = Boolean = Column = DateTime = Index = Integer = String = Table = UniqueConstraint = select = update = None
    _AsyncSession = None
    _async_sessionmaker = None


class SqlToolOperationRepository(ToolStateStore):
    """Persist tool operations with tenant, lease, and fence ownership."""

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> "Table":
        _require_sqlalchemy()
        integer_id = BigInteger().with_variant(Integer, "sqlite")
        table = Table(
            storage_name("tool_operations"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("tenant_id", String(128), nullable=False),
            Column("operation_id", String(256), nullable=False),
            Column("run_id", String(256), nullable=False),
            Column("tool_call_id", String(256), nullable=False),
            Column("idempotency_key_hash", String(64), nullable=False),
            Column("tool_name", String(256), nullable=False),
            Column("arguments_hash", String(64), nullable=False),
            Column("binding_fingerprint", String(64), nullable=False),
            Column("replay_safe", Boolean, nullable=False),
            Column("status", String(32), nullable=False),
            Column("owner", String(256), nullable=True),
            Column("fence", Integer, nullable=False),
            Column("lease_expires_at", DateTime(timezone=True), nullable=True),
            Column("result_ref", String(512), nullable=True),
            Column("result_digest", String(64), nullable=True),
            Column("error_code", String(128), nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("tenant_id", "operation_id", name=storage_name("tool_operations_uk_tenant_operation")),
            UniqueConstraint("tenant_id", "run_id", "tool_call_id", name=storage_name("tool_operations_uk_run_call")),
        )
        Index(storage_name("tool_operations_ix_tenant_status"), table.c.tenant_id, table.c.status, table.c.updated_at)
        registry.add_table(table, owner="adapter.tool")
        return table

    def __init__(self, session_factory: "async_sessionmaker[AsyncSession]", *, table: "Table") -> None:
        _require_sqlalchemy()
        self._sessions = session_factory
        self._table = table

    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        async with self._sessions() as session:
            async with session.begin():
                row = (await session.execute(select(self._table).where(self._table.c.tenant_id == record.tenant_id, self._table.c.operation_id == record.operation_id))).mappings().first()
                if row is not None:
                    existing = _record_from_row(row)
                    if (
                        existing.run_id != record.run_id
                        or existing.tool_call_id != record.tool_call_id
                        or existing.idempotency_key_hash != record.idempotency_key_hash
                        or existing.tool_name != record.tool_name
                        or existing.arguments_hash != record.arguments_hash
                        or existing.binding_fingerprint != record.binding_fingerprint
                        or existing.replay_safe != record.replay_safe
                    ):
                        raise LinktoolsAIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                    return existing
                await session.execute(self._table.insert().values(_record_values(record)))
                return record

    async def get_operation(self, operation_id: str, *, tenant_id: str) -> "ToolOperationRecord | None":
        async with self._sessions() as session:
            row = (await session.execute(select(self._table).where(self._table.c.tenant_id == tenant_id, self._table.c.operation_id == operation_id))).mappings().first()
        return None if row is None else _record_from_row(row)

    async def claim(self, operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> ToolOperationRecord:
        if not owner.strip() or not 1 <= lease_seconds <= 3600:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        now = datetime.now(timezone.utc)
        async with self._sessions() as session:
            async with session.begin():
                current = await self._locked(session, operation_id, tenant_id)
                if current is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
                    raise LinktoolsAIError(ErrorCode.TASK_TERMINAL_CONFLICT)
                if current.status is ToolOperationStatus.CLAIMED and current.lease_expires_at is not None and current.lease_expires_at <= now and not current.replay_safe:
                    unknown = _replace_record(current, status=ToolOperationStatus.EFFECT_UNKNOWN, lease_expires_at=None)
                    await session.execute(update(self._table).where(self._table.c.tenant_id == tenant_id, self._table.c.operation_id == operation_id, self._table.c.fence == current.fence).values(_record_values(unknown)))
                    raise LinktoolsAIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                if current.owner is not None and current.lease_expires_at is not None and current.lease_expires_at > now:
                    raise LinktoolsAIError(ErrorCode.TASK_OWNER_CONFLICT)
                updated = _replace_record(current, status=ToolOperationStatus.CLAIMED, owner=owner, fence=current.fence + 1, lease_expires_at=now + timedelta(seconds=lease_seconds))
                await session.execute(update(self._table).where(self._table.c.tenant_id == tenant_id, self._table.c.operation_id == operation_id).values(_record_values(updated)))
                return updated

    async def renew(self, operation_id: str, *, tenant_id: str, owner: str, fence: int, lease_seconds: int) -> ToolOperationRecord:
        if not owner.strip() or not 1 <= lease_seconds <= 3600:
            raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = await self._owned(operation_id, tenant_id, owner, fence)
        return await self._update_owned(current, lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds))

    async def complete(self, operation_id: str, *, tenant_id: str, owner: str, fence: int, result_ref: "str | None", result_digest: str) -> ToolOperationRecord:
        current = await self._owned(operation_id, tenant_id, owner, fence)
        if current.status is ToolOperationStatus.COMPLETED:
            if current.result_digest == result_digest:
                return current
            raise LinktoolsAIError(ErrorCode.TOOL_RESULT_CONFLICT)
        return await self._update_owned(current, status=ToolOperationStatus.COMPLETED, result_ref=result_ref, result_digest=result_digest, lease_expires_at=None)

    async def fail(self, operation_id: str, *, tenant_id: str, owner: str, fence: int, error_code: str) -> ToolOperationRecord:
        current = await self._owned(operation_id, tenant_id, owner, fence)
        return await self._update_owned(current, status=ToolOperationStatus.FAILED, error_code=error_code, lease_expires_at=None)

    async def _owned(self, operation_id: str, tenant_id: str, owner: str, fence: int) -> ToolOperationRecord:
        current = await self.get_operation(operation_id, tenant_id=tenant_id)
        if current is None or current.owner != owner or current.fence != fence:
            raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        return current

    async def _update_owned(self, current: ToolOperationRecord, **changes: object) -> ToolOperationRecord:
        updated = _replace_record(current, **changes)
        async with self._sessions() as session:
            async with session.begin():
                outcome = await session.execute(update(self._table).where(self._table.c.tenant_id == current.tenant_id, self._table.c.operation_id == current.operation_id, self._table.c.owner == current.owner, self._table.c.fence == current.fence).values(_record_values(updated)))
                if outcome.rowcount != 1:
                    raise LinktoolsAIError(ErrorCode.TASK_FENCE_STALE)
        return updated

    async def _locked(self, session: "AsyncSession", operation_id: str, tenant_id: str) -> "ToolOperationRecord | None":
        row = (await session.execute(select(self._table).where(self._table.c.tenant_id == tenant_id, self._table.c.operation_id == operation_id).with_for_update())).mappings().first()
        return None if row is None else _record_from_row(row)


def _record_values(record: ToolOperationRecord) -> dict[str, object]:
    return {
        "tenant_id": record.tenant_id, "operation_id": record.operation_id, "run_id": record.run_id,
        "tool_call_id": record.tool_call_id, "idempotency_key_hash": record.idempotency_key_hash,
        "tool_name": record.tool_name, "arguments_hash": record.arguments_hash,
        "binding_fingerprint": record.binding_fingerprint, "replay_safe": record.replay_safe,
        "status": record.status.value, "owner": record.owner, "fence": record.fence,
        "lease_expires_at": record.lease_expires_at, "result_ref": record.result_ref,
        "result_digest": record.result_digest, "error_code": record.error_code,
        "created_at": record.created_at, "updated_at": record.updated_at,
    }


def _record_from_row(row: "Mapping[str, object]") -> ToolOperationRecord:
    return ToolOperationRecord(
        str(row["operation_id"]), str(row["tenant_id"]), str(row["run_id"]), str(row["tool_call_id"]),
        str(row["idempotency_key_hash"]), str(row["tool_name"]), str(row["arguments_hash"]),
        str(row["binding_fingerprint"]), bool(row["replay_safe"]), ToolOperationStatus(str(row["status"])),
        None if row["owner"] is None else str(row["owner"]), int(row["fence"]),
        row["lease_expires_at"] if isinstance(row["lease_expires_at"], datetime) else None,
        None if row["result_ref"] is None else str(row["result_ref"]),
        None if row["result_digest"] is None else str(row["result_digest"]),
        None if row["error_code"] is None else str(row["error_code"]),
        _datetime(row["created_at"]), _datetime(row["updated_at"]),
    )


def _replace_record(record: ToolOperationRecord, **changes: object) -> ToolOperationRecord:
    values = _record_values(record)
    values.update(changes)
    return _record_from_row(values)


def _datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _require_sqlalchemy() -> None:
    if _AsyncSession is None or _async_sessionmaker is None or Table is None or select is None or update is None:
        raise LinktoolsAIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING, "SQLAlchemy is required for SqlToolOperationRepository")


__all__ = ["SqlToolOperationRepository"]
