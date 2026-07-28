"""SQLAlchemy ToolStateStore with one fenced operation aggregate."""

from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import TABLE_PREFIX
from ..state import ToolOperation, ToolOperationStatus


class OperationRow(Base):
    __tablename__ = f"{TABLE_PREFIX}tool_operations"
    __table_args__ = (UniqueConstraint("run_id", "tool_call_id", name="uq_tool_operation_call"),)
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(String(255))
    run_id: Mapped[str] = mapped_column(String(255), index=True)
    tool_call_id: Mapped[str] = mapped_column(String(255))
    idempotency_key: Mapped[str] = mapped_column(String(255))
    tool_name: Mapped[str] = mapped_column(String(255))
    arguments_hash: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), index=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    fence: Mapped[int] = mapped_column(Integer, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[Any] = mapped_column(JSON, nullable=True)
    error: Mapped[Any] = mapped_column(JSON, nullable=True)


class SqlAlchemyToolStateStore:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @staticmethod
    def _operation(row: OperationRow) -> ToolOperation:
        return ToolOperation(row.id, row.tenant_id, row.run_id, row.tool_call_id, row.idempotency_key, row.tool_name, row.arguments_hash, ToolOperationStatus(row.status), row.owner, row.fence, row.lease_expires_at, row.result, row.error)

    @staticmethod
    def _check_fence(row: OperationRow | None, owner: str, fence: int) -> None:
        if row is None or row.owner != owner or row.fence != fence:
            raise ValueError("tool operation fence conflict")

    async def prepare(self, operation: ToolOperation) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(OperationRow, operation.id, with_for_update=True)
                if row is not None:
                    if row.idempotency_key != operation.idempotency_key or row.arguments_hash != operation.arguments_hash:
                        raise ValueError("tool operation idempotency conflict")
                    return self._operation(row)
                session.add(OperationRow(**asdict(operation)))
                return operation

    async def get(self, operation_id: str) -> ToolOperation | None:
        async with self.session_factory() as session:
            row = await session.get(OperationRow, operation_id)
            return None if row is None else self._operation(row)

    async def claim(self, operation_id: str, *, owner: str) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(OperationRow, operation_id, with_for_update=True)
                if row is None:
                    raise KeyError(operation_id)
                if row.status == ToolOperationStatus.COMPLETED.value:
                    return self._operation(row)
                row.status, row.owner, row.fence = ToolOperationStatus.CLAIMED.value, owner, row.fence + 1
                return self._operation(row)

    async def renew(self, operation_id: str, *, owner: str, fence: int) -> ToolOperation:
        async with self.session_factory() as session:
            row = await session.get(OperationRow, operation_id)
            self._check_fence(row, owner, fence)
            return self._operation(row)

    async def complete(self, operation_id: str, *, owner: str, fence: int, result: Any) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(OperationRow, operation_id, with_for_update=True)
                self._check_fence(row, owner, fence)
                row.status, row.result = ToolOperationStatus.COMPLETED.value, result
                return self._operation(row)

    async def fail(self, operation_id: str, *, owner: str, fence: int, error: Any) -> ToolOperation:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(OperationRow, operation_id, with_for_update=True)
                self._check_fence(row, owner, fence)
                row.status, row.error = ToolOperationStatus.FAILED.value, error
                return self._operation(row)
