#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed occupancy on the existing strictly coordinated Conversation row."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from ..core import SessionStatus, validate_resource_id
from ..errors import AIError, ErrorCode
from .state import SqlStateRowReader, SqlStateTransaction

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from ._sql import SqlStateStore
    from .state import StateTransaction


@dataclass(frozen=True, slots=True)
class StrictSessionTurn:
    execution_id: str | None
    fence: int | None
    storage_revision: int


class StrictSessionOccupancy:
    """Use only inside the current Conversation/Execution transaction."""

    def __init__(self, store: SqlStateStore, session_record_key: bytes) -> None:
        self._context = store.context
        self._table = store.storage_group.metadata.tables["ai_state_records"]
        c = self._table.c
        self._owner = (c.store_key == store.store_key, c.kind == "session",
                       c.key_locator == sha256(session_record_key).hexdigest(), c.key == session_record_key)

    async def read(self, tx: StateTransaction) -> StrictSessionTurn | None:
        from sqlalchemy import select

        if not isinstance(tx, SqlStateRowReader):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        c = self._table.c
        rows = await tx.read_sql_rows(self._context, select(
            c.strict_turn_execution_id, c.strict_turn_fence, c.storage_revision,
        ).where(*self._owner))
        if not rows:
            return None
        row, = rows
        value = row["strict_turn_execution_id"]
        return StrictSessionTurn(None if value is None else value.decode("utf-8"),
                                 row["strict_turn_fence"], row["storage_revision"])

    async def claim(self, tx: StateTransaction, expected: StrictSessionTurn, *, execution_id: str) -> int | None:
        # Caller revokes the exact former Execution authority in this same
        # mutation boundary before assigning an already occupied session.
        if expected.fence is None:
            return None
        if not 0 <= expected.fence < (2**63 - 1):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        c = self._table.c
        previous = None if expected.execution_id is None else expected.execution_id.encode("utf-8")
        fence = expected.fence + 1
        changed = await self._write(tx, self._table.update().where(
            *self._owner, c.state == SessionStatus.OPEN.value, c.storage_revision == expected.storage_revision,
            c.strict_turn_fence == expected.fence, c.strict_turn_execution_id == previous,
        ).values(strict_turn_execution_id=validate_resource_id(execution_id).encode("utf-8"),
                 strict_turn_fence=fence))
        return fence if changed else None

    async def release(self, tx: StateTransaction, *, execution_id: str, fence: int) -> bool:
        c = self._table.c
        return await self._write(tx, self._table.update().where(
            *self._owner, c.strict_turn_fence == fence,
            c.strict_turn_execution_id == validate_resource_id(execution_id).encode("utf-8"),
        ).values(strict_turn_execution_id=None))

    async def _write(self, tx: StateTransaction, statement) -> bool:
        if not isinstance(tx, SqlStateTransaction):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

        async def write(connection: AsyncSession) -> bool:
            result = await connection.execute(statement)
            return result.rowcount == 1

        return await tx.run_sql(self._context, write)
