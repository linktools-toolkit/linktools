"""SQLAlchemy ToolStateStore with one fenced operation aggregate."""

from datetime import datetime, timedelta, timezone
from sqlalchemy import JSON, Boolean, DateTime, Integer, String, UniqueConstraint, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from ....errors import StorageConflictError
from ....json import JsonValue, normalize_json
from ....storage.coordination.lease import Lease, assert_active, claim, release, renew
from ....storage.database import CoordinationScope
from ....storage.sqlalchemy.base import Base
from ....storage.sqlalchemy.conventions import TABLE_PREFIX, as_utc
from ..models import ToolOperation, ToolOperationStatus


class OperationRow(Base):
    __tablename__ = f"{TABLE_PREFIX}tool_operations"
    __table_args__ = (UniqueConstraint("run_id", "tool_call_id", name="uq_tool_operation_call"),)
    operation_id: Mapped[str] = mapped_column(String(128), unique=True)
    tenant_id: Mapped[str | None] = mapped_column(String(128))
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    tool_call_id: Mapped[str] = mapped_column(String(128))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    tool_name: Mapped[str] = mapped_column(String(128))
    arguments_hash: Mapped[str] = mapped_column(String(128))
    binding_fingerprint: Mapped[str] = mapped_column(String(128))
    replay_safe: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(32), index=True)
    owner: Mapped[str | None] = mapped_column(String(128))
    fence: Mapped[int] = mapped_column(Integer, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[JsonValue | None] = mapped_column(JSON, nullable=True)
    error: Mapped[JsonValue | None] = mapped_column(JSON, nullable=True)


class SqlAlchemyToolStateBackend:
    coordination_scope = CoordinationScope.SHARED_DATABASE

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @staticmethod
    async def _row(session, operation_id: str, *, for_update: bool = False):
        query = select(OperationRow).where(
            OperationRow.operation_id == operation_id
        )
        if for_update:
            query = query.with_for_update()
        return await session.scalar(query)

    @staticmethod
    def _operation(row: OperationRow) -> ToolOperation:
        return ToolOperation(
            id=row.operation_id,
            tenant_id=row.tenant_id,
            execution_id=row.run_id,
            tool_call_id=row.tool_call_id,
            idempotency_key=row.idempotency_key,
            tool_name=row.tool_name,
            arguments_hash=row.arguments_hash,
            binding_fingerprint=row.binding_fingerprint,
            status=ToolOperationStatus(row.status),
            replay_safe=row.replay_safe,
            lease=Lease(row.owner, row.fence, row.lease_expires_at),
            result=row.result,
            error=row.error,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    async def _update_claimed(session, row: OperationRow, **values: object) -> OperationRow:
        result = await session.execute(
            update(OperationRow)
            .where(
                OperationRow.operation_id == row.operation_id,
                OperationRow.status == ToolOperationStatus.CLAIMED.value,
                OperationRow.owner == row.owner,
                OperationRow.fence == row.fence,
                OperationRow.lease_expires_at == row.lease_expires_at,
            )
            .values(**values)
        )
        if result.rowcount != 1:
            raise StorageConflictError("tool operation claim changed concurrently")
        updated = await SqlAlchemyToolStateBackend._row(session, row.operation_id)
        return updated

    async def prepare(self, operation: ToolOperation) -> ToolOperation:
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    row = await self._row(session, operation.id, for_update=True)
                    if row is not None:
                        return self._replay(row, operation)
                    session.add(OperationRow(
                        operation_id=operation.id,
                        tenant_id=operation.tenant_id,
                        run_id=operation.execution_id,
                        tool_call_id=operation.tool_call_id,
                        idempotency_key=operation.idempotency_key,
                        tool_name=operation.tool_name,
                        arguments_hash=operation.arguments_hash,
                        binding_fingerprint=operation.binding_fingerprint,
                        replay_safe=operation.replay_safe,
                        status=operation.status.value,
                        owner=operation.owner,
                        fence=operation.fence,
                        lease_expires_at=operation.lease.expires_at,
                        result=operation.result,
                        error=operation.error,
                        created_at=operation.created_at,
                        updated_at=operation.updated_at,
                    ))
                    await session.flush()
                    return operation
        except IntegrityError:
            async with self.session_factory() as session:
                row = await session.scalar(
                    select(OperationRow).where(
                        OperationRow.run_id == operation.execution_id,
                        OperationRow.tool_call_id == operation.tool_call_id,
                    )
                )
                if row is None:
                    raise StorageConflictError("tool operation prepare conflict")
                return self._replay(row, operation)

    def _replay(self, row: OperationRow, operation: ToolOperation) -> ToolOperation:
        if (
            row.operation_id != operation.id
            or row.idempotency_key != operation.idempotency_key
            or row.arguments_hash != operation.arguments_hash
            or row.binding_fingerprint != operation.binding_fingerprint
            or row.tenant_id != operation.tenant_id
            or row.run_id != operation.execution_id
            or row.tool_call_id != operation.tool_call_id
            or row.tool_name != operation.tool_name
        ):
            raise StorageConflictError("tool operation idempotency conflict")
        return self._operation(row)

    async def get(self, operation_id: str) -> ToolOperation | None:
        async with self.session_factory() as session:
            row = await self._row(session, operation_id)
            return None if row is None else self._operation(row)

    async def claim(self, operation_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._row(session, operation_id, for_update=True)
                if row is None:
                    raise KeyError(operation_id)
                if row.status == ToolOperationStatus.COMPLETED.value:
                    return self._operation(row)
                if row.status == ToolOperationStatus.FAILED.value:
                    raise StorageConflictError("failed tool operation cannot be claimed")
                if row.status == ToolOperationStatus.INDETERMINATE.value:
                    return self._operation(row)
                now = datetime.now(timezone.utc)
                if (
                    row.status == ToolOperationStatus.CLAIMED.value
                    and row.lease_expires_at is not None
                    and as_utc(row.lease_expires_at) <= now
                    and not row.replay_safe
                ):
                    row.status = ToolOperationStatus.INDETERMINATE.value
                    row.updated_at = now
                    await session.flush()
                    return self._operation(row)
                lease = claim(
                    Lease(row.owner, row.fence, row.lease_expires_at),
                    owner=owner,
                    now=now,
                    duration=duration,
                )
                result = await session.execute(
                    update(OperationRow)
                    .where(
                        OperationRow.operation_id == operation_id,
                        OperationRow.status == row.status,
                        OperationRow.fence == row.fence,
                    )
                    .values(
                        status=ToolOperationStatus.CLAIMED.value,
                        owner=lease.owner,
                        fence=lease.fence,
                        lease_expires_at=lease.expires_at,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    raise StorageConflictError("tool operation claim conflict")
                claimed = await self._row(session, operation_id)
                return self._operation(claimed)

    async def renew(self, operation_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._row(session, operation_id, for_update=True)
                if row is None:
                    raise KeyError(operation_id)
                now = datetime.now(timezone.utc)
                lease = renew(
                    Lease(row.owner, row.fence, row.lease_expires_at),
                    owner=owner,
                    fence=fence,
                    now=now,
                    duration=duration,
                )
                updated = await self._update_claimed(
                    session,
                    row,
                    lease_expires_at=lease.expires_at,
                    updated_at=now,
                )
                return self._operation(updated)

    async def complete(self, operation_id: str, *, owner: str, fence: int, result: JsonValue) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._row(session, operation_id, for_update=True)
                if row is None:
                    raise KeyError(operation_id)
                now = datetime.now(timezone.utc)
                assert_active(Lease(row.owner, row.fence, row.lease_expires_at), owner=owner, fence=fence, now=now)
                if row.status != ToolOperationStatus.CLAIMED.value:
                    raise StorageConflictError("tool operation is not claimed")
                lease = release(Lease(row.owner, row.fence, row.lease_expires_at))
                updated = await self._update_claimed(
                    session,
                    row,
                    status=ToolOperationStatus.COMPLETED.value,
                    result=normalize_json(result),
                    owner=lease.owner,
                    lease_expires_at=lease.expires_at,
                    updated_at=now,
                )
                return self._operation(updated)

    async def fail(self, operation_id: str, *, owner: str, fence: int, error: JsonValue) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._row(session, operation_id, for_update=True)
                if row is None:
                    raise KeyError(operation_id)
                now = datetime.now(timezone.utc)
                assert_active(Lease(row.owner, row.fence, row.lease_expires_at), owner=owner, fence=fence, now=now)
                if row.status != ToolOperationStatus.CLAIMED.value:
                    raise StorageConflictError("tool operation is not claimed")
                lease = release(Lease(row.owner, row.fence, row.lease_expires_at))
                updated = await self._update_claimed(
                    session,
                    row,
                    status=ToolOperationStatus.FAILED.value,
                    error=normalize_json(error),
                    owner=lease.owner,
                    lease_expires_at=lease.expires_at,
                    updated_at=now,
                )
                return self._operation(updated)

    async def mark_indeterminate(self, operation_id: str, *, owner: str, fence: int, error: JsonValue) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await self._row(session, operation_id, for_update=True)
                if row is None:
                    raise KeyError(operation_id)
                now = datetime.now(timezone.utc)
                assert_active(Lease(row.owner, row.fence, row.lease_expires_at), owner=owner, fence=fence, now=now)
                updated = await self._update_claimed(
                    session,
                    row,
                    status=ToolOperationStatus.INDETERMINATE.value,
                    error=normalize_json(error),
                    lease_expires_at=None,
                    updated_at=now,
                )
                return self._operation(updated)
