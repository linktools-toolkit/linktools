#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL-backed production tool-operation state."""

import hashlib
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..capability.tool import ToolState, ToolStateStore
from ..core.errors import ErrorCode, LinktoolsAIError
from ..storage.database import SqlSchemaRegistry
from ..storage.dialects import resolve_dialect
from ..storage.names import storage_name

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

try:
    from sqlalchemy import BigInteger, Boolean, Column, DateTime, Index, Integer, JSON, String, Table, UniqueConstraint, select
    from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
    from sqlalchemy.ext.asyncio import async_sessionmaker as _async_sessionmaker
except ModuleNotFoundError as error:
    if error.name != "sqlalchemy":
        raise
    Column = None
    Boolean = None
    BigInteger = None
    DateTime = None
    Index = None
    Integer = None
    JSON = None
    String = None
    Table = None
    UniqueConstraint = None
    select = None
    _AsyncSession = None
    _async_sessionmaker = None


class SqlToolState(ToolStateStore):
    """Persist tool operation state in the caller-owned SQL database."""

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> 'Table':
        _require_sqlalchemy()
        integer_id = BigInteger().with_variant(Integer, "sqlite")
        table = Table(
            storage_name("tool_operations"),
            registry.metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("operation_id", String(128), nullable=False),
            Column("tenant_id", String(128), nullable=True),
            Column("run_id", String(128), nullable=False),
            Column("tool_call_id", String(128), nullable=False),
            Column("idempotency_key", String(128), nullable=False),
            Column("tool_name", String(128), nullable=False),
            Column("arguments_hash", String(128), nullable=False),
            Column("binding_fingerprint", String(128), nullable=False),
            Column("replay_safe", Boolean, nullable=False),
            Column("status", String(32), nullable=False),
            Column("owner", String(128), nullable=True),
            Column("fence", Integer, nullable=False),
            Column("lease_expires_at", DateTime(timezone=True), nullable=True),
            Column("result", JSON, nullable=True),
            Column("error", JSON, nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("operation_id", name=storage_name("tool_operations_uk_operation_id")),
        )
        Index(storage_name("tool_operations_ix_updated_at"), table.c.updated_at)
        Index(storage_name("tool_operations_ix_created_at"), table.c.created_at)
        Index(storage_name("tool_operations_uk_run_id_tool_call_id"), table.c.run_id, table.c.tool_call_id, unique=True)
        registry.add_table(table, owner="adapter.tool")
        return table

    def __init__(
        self,
        session_factory: 'async_sessionmaker[AsyncSession]',
        *,
        table: 'Table',
    ) -> None:
        _require_sqlalchemy()
        self._sessions = session_factory
        self._table = table

    async def get(self, operation_id: str) -> 'ToolState | None':
        if not operation_id.strip():
            raise ValueError("tool operation id is required")
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(self._table).where(self._table.c.operation_id == operation_id)
                )
            ).mappings().first()
        if row is None:
            return None
        return _state_from_row(row)

    async def put(self, state: ToolState) -> ToolState:
        async with self._sessions() as session:
            async with session.begin():
                timestamp = datetime.now(timezone.utc)
                outcome = await resolve_dialect(session).insert_ignore_conflict(
                    session,
                    table=self._table,
                    values=_new_row(state, timestamp),
                    index_elements=("operation_id",),
                )
                if outcome.inserted:
                    return state
                previous = (
                    await session.execute(
                        select(self._table).where(self._table.c.operation_id == state.operation_id)
                    )
                ).mappings().first()
                if previous is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT, "tool operation state race")
                existing = _state_from_row(previous)
                if existing != state:
                    raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT, "tool operation state conflict")
                return existing
        return state


def _new_row(state: ToolState, timestamp: datetime) -> 'dict[str, str | int | bool | datetime | dict[str, str] | None]':
    return {
        "operation_id": state.operation_id,
        "id": int.from_bytes(hashlib.sha256(state.operation_id.encode("utf-8")).digest()[:8], "big", signed=True),
        "tenant_id": None,
        "run_id": state.operation_id,
        "tool_call_id": state.operation_id,
        "idempotency_key": state.operation_id,
        "tool_name": "linktools-tool-state",
        "arguments_hash": "",
        "binding_fingerprint": "",
        "replay_safe": True,
        "status": state.state,
        "owner": None,
        "fence": 0,
        "lease_expires_at": None,
        "result": None if state.result_digest is None else {"digest": state.result_digest},
        "error": None,
        "updated_at": timestamp,
        "created_at": timestamp,
    }


def _state_from_row(row: 'Mapping[str, str | int | bool | datetime | dict[str, str] | None]') -> ToolState:
    raw_result = row["result"]
    result_digest: str | None = None
    if isinstance(raw_result, dict) and isinstance(raw_result.get("digest"), str):
        result_digest = raw_result["digest"]
    return ToolState(str(row["operation_id"]), str(row["status"]), result_digest)


def _require_sqlalchemy() -> None:
    if (
        _AsyncSession is None
        or _async_sessionmaker is None
        or Table is None
        or BigInteger is None
        or select is None
        or UniqueConstraint is None
    ):
        raise LinktoolsAIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING, "SQLAlchemy is required for SqlToolState")


__all__ = ["SqlToolState"]
