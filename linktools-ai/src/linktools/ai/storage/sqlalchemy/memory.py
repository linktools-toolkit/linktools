#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyMemoryStore: DB-backed MemoryStore (the Protocol in
memory/store.py). Mirrors SqlAlchemySwarmStore's structure:
`session_factory: Callable[[], AsyncSession]` constructor, `_as_utc` helper for
aiosqlite's naive-datetime round-trip, and read-check-mutate-commit transactions.

Search uses ``content LIKE`` with optional ``owner_id`` / ``category`` filters
(category is indexed for selectivity). The `_UNSET` sentinel distinguishes
"omit this field" from `category=None` meaning "explicitly clear" (same
semantics as FilesystemMemoryStore)."""

import json
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import MemoryRow
from ...errors import MemoryConflictError, MemoryNotFoundError
from ...memory.models import MemoryRecord
from ...memory.scope import LEGACY_TENANT_ID, MemoryScope, is_legacy_tenant
from ...memory.store import _UNSET


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes MemoryRecord is constructed with everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_record(row: MemoryRow) -> MemoryRecord:
    # A NULL tenant_id is a legacy row (pre-tenant). It is read back under the
    # reserved legacy tenant so it never matches a real tenant's search.
    return MemoryRecord(
        id=row.id,
        tenant_id=row.tenant_id or LEGACY_TENANT_ID,
        owner_id=row.owner_id,
        content=row.content,
        category=row.category,
        confidence=row.confidence,
        version=row.version,
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
        metadata=json.loads(row.metadata_json),
        user_id=row.user_id,
        workspace_id=row.workspace_id,
        session_id=row.session_id,
    )


class SqlAlchemyMemoryStore:
    """Multi-process MemoryStore backed by SQLAlchemy/AsyncSession.

    Optimistic concurrency on ``update`` / ``forget`` mirrors
    ``SqlAlchemySwarmStore.update_run`` (read-check-mutate-commit in one
    transaction). ``remember`` relies on the primary-key constraint: a duplicate
    id raises ``IntegrityError``, which is translated to ``MemoryConflictError``.
    """

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
    ) -> None:
        self._session_factory = session_factory
        # UoW mode: when set, every method uses this shared session directly and
        # does NOT open its own session or call session.begin() -- the UoW owns
        # the transaction. None means normal mode (own session + transaction).
        self._session = session

    async def _execute_in_session(self, fn):
        """Run ``fn(session)`` in own transaction (normal mode) or against the
        shared session (UoW mode). See SqlAlchemyRunStore._execute_in_session."""
        if self._session is not None:
            result = await fn(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            async with session.begin():
                return await fn(session)

    # -- read ----------------------------------------------------------

    async def get(self, memory_id: str) -> "MemoryRecord | None":
        async def _do(session):
            result = await session.execute(
                select(MemoryRow).where(MemoryRow.id == memory_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)

        return await self._execute_in_session(_do)

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 10,
        category: "str | None" = None,
    ) -> "tuple[MemoryRecord, ...]":
        async def _do(session):
            stmt = select(MemoryRow).where(MemoryRow.content.like(f"%{query}%"))
            # Hard tenant isolation. A real tenant matches only its own rows;
            # the legacy scope additionally sees pre-tenant NULL-tenant rows
            # (the migration quarantine). A NULL row tenant never matches a
            # real tenant, so legacy data is never exposed.
            if is_legacy_tenant(scope.tenant_id):
                stmt = stmt.where(
                    or_(
                        MemoryRow.tenant_id == LEGACY_TENANT_ID,
                        MemoryRow.tenant_id.is_(None),
                    )
                )
            else:
                stmt = stmt.where(MemoryRow.tenant_id == scope.tenant_id)
            # Sub-scopes: a NULL record field = "shared at tenant level", so the
            # filter is (record IS NULL OR record == scope value).
            if scope.user_id is not None:
                stmt = stmt.where(
                    or_(
                        MemoryRow.user_id.is_(None),
                        MemoryRow.user_id == scope.user_id,
                    )
                )
            if scope.workspace_id is not None:
                stmt = stmt.where(
                    or_(
                        MemoryRow.workspace_id.is_(None),
                        MemoryRow.workspace_id == scope.workspace_id,
                    )
                )
            if scope.session_id is not None:
                stmt = stmt.where(
                    or_(
                        MemoryRow.session_id.is_(None),
                        MemoryRow.session_id == scope.session_id,
                    )
                )
            if category is not None:
                stmt = stmt.where(MemoryRow.category == category)
            stmt = stmt.order_by(MemoryRow.created_at).limit(limit)
            result = await session.execute(stmt)
            return tuple(_row_to_record(row) for row in result.scalars())

        return await self._execute_in_session(_do)

    # -- write ---------------------------------------------------------

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        async def _do(session):
            session.add(
                MemoryRow(
                    id=record.id,
                    tenant_id=record.tenant_id,
                    owner_id=record.owner_id,
                    content=record.content,
                    category=record.category,
                    confidence=record.confidence,
                    version=record.version,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    metadata_json=json.dumps(dict(record.metadata)),
                    user_id=record.user_id,
                    workspace_id=record.workspace_id,
                    session_id=record.session_id,
                )
            )

        try:
            await self._execute_in_session(_do)
        except IntegrityError as exc:
            # Duplicate primary key -> conflict, matching FilesystemMemoryStore's
            # "memory already exists" semantics. In UoW mode the IntegrityError
            # has already poisoned the shared transaction (it will roll back);
            # we still translate so callers see the domain error type.
            raise MemoryConflictError(f"memory already exists: {record.id}") from exc
        return record

    async def update(
        self,
        memory_id: str,
        *,
        expected_version: int,
        content: object = _UNSET,
        category: object = _UNSET,
        confidence: object = _UNSET,
        metadata: object = _UNSET,
    ) -> MemoryRecord:
        async def _do(session):
            query_result = await session.execute(
                select(MemoryRow).where(MemoryRow.id == memory_id)
            )
            row = query_result.scalar_one_or_none()
            if row is None:
                raise MemoryNotFoundError(f"memory not found: {memory_id}")
            if row.version != expected_version:
                raise MemoryConflictError(
                    f"expected version {expected_version}, found {row.version}"
                )
            # Apply ONLY fields explicitly passed (i.e. `is not _UNSET`); a
            # None value means "clear this field" (e.g. category=None).
            if content is not _UNSET:
                row.content = content
            if category is not _UNSET:
                row.category = category
            if confidence is not _UNSET:
                row.confidence = confidence
            if metadata is not _UNSET:
                row.metadata_json = json.dumps(metadata)
            row.version = row.version + 1
            row.updated_at = datetime.now(timezone.utc)
            await session.flush()
            return _row_to_record(row)

        return await self._execute_in_session(_do)

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        async def _do(session):
            query_result = await session.execute(
                select(MemoryRow).where(MemoryRow.id == memory_id)
            )
            row = query_result.scalar_one_or_none()
            if row is None:
                raise MemoryNotFoundError(f"memory not found: {memory_id}")
            if row.version != expected_version:
                raise MemoryConflictError(
                    f"expected version {expected_version}, found {row.version}"
                )
            await session.delete(row)

        await self._execute_in_session(_do)
