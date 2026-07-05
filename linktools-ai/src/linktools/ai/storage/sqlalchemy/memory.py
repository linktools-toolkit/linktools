#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyMemoryStore: DB-backed MemoryStore (the Protocol in
memory_runtime/store.py). Mirrors SqlAlchemySwarmStore's structure:
`session_factory: Callable[[], AsyncSession]` constructor, `_as_utc` helper for
aiosqlite's naive-datetime round-trip, and read-check-mutate-commit transactions.

Search uses ``content LIKE`` with optional ``owner_id`` / ``category`` filters
(category is indexed for selectivity). The `_UNSET` sentinel distinguishes
"omit this field" from `category=None` meaning "explicitly clear" (same
semantics as FileMemoryStore)."""

import json
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import MemoryRow
from ...errors import MemoryConflictError, MemoryNotFoundError
from ...memory_runtime.models import MemoryRecord
from ...memory_runtime.store import _UNSET


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes MemoryRecord is constructed with everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_record(row: MemoryRow) -> MemoryRecord:
    return MemoryRecord(
        id=row.id,
        owner_id=row.owner_id,
        content=row.content,
        category=row.category,
        confidence=row.confidence,
        version=row.version,
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
        metadata=json.loads(row.metadata_json),
    )


class SqlAlchemyMemoryStore:
    """Multi-process MemoryStore backed by SQLAlchemy/AsyncSession.

    Optimistic concurrency on ``update`` / ``forget`` mirrors
    ``SqlAlchemySwarmStore.update_run`` (read-check-mutate-commit in one
    transaction). ``remember`` relies on the primary-key constraint: a duplicate
    id raises ``IntegrityError``, which is translated to ``MemoryConflictError``.
    """

    def __init__(self, *, session_factory: "Callable[[], AsyncSession]") -> None:
        self._session_factory = session_factory

    # -- read ----------------------------------------------------------

    async def get(self, memory_id: str) -> "MemoryRecord | None":
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryRow).where(MemoryRow.id == memory_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)

    async def search(
        self,
        query: str,
        *,
        owner_id: "str | None" = None,
        category: "str | None" = None,
        limit: int = 10,
    ) -> "tuple[MemoryRecord, ...]":
        async with self._session_factory() as session:
            stmt = select(MemoryRow).where(MemoryRow.content.like(f"%{query}%"))
            if owner_id is not None:
                stmt = stmt.where(MemoryRow.owner_id == owner_id)
            if category is not None:
                stmt = stmt.where(MemoryRow.category == category)
            stmt = stmt.order_by(MemoryRow.created_at).limit(limit)
            result = await session.execute(stmt)
            return tuple(_row_to_record(row) for row in result.scalars())

    # -- write ---------------------------------------------------------

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        async with self._session_factory() as session:
            try:
                async with session.begin():
                    session.add(MemoryRow(
                        id=record.id,
                        owner_id=record.owner_id,
                        content=record.content,
                        category=record.category,
                        confidence=record.confidence,
                        version=record.version,
                        created_at=record.created_at,
                        updated_at=record.updated_at,
                        metadata_json=json.dumps(dict(record.metadata)),
                    ))
            except IntegrityError as exc:
                # Duplicate primary key -> conflict, matching FileMemoryStore's
                # "memory already exists" semantics.
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
        async with self._session_factory() as session:
            async with session.begin():
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

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        async with self._session_factory() as session:
            async with session.begin():
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
