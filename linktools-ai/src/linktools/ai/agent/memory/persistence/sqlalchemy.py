#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyMemoryBackend: DB-backed MemoryStore (the Protocol in
memory/store.py). Mirrors ``SqlAlchemyTaskBackend``'s structure:
`session_factory: Callable[[], AsyncSession]` constructor, ``as_utc`` for
aiosqlite's naive-datetime round-trip, and read-check-mutate-commit transactions.

Search uses ``content LIKE`` with optional ``owner_id`` / ``category`` filters
(category is indexed for selectivity). The `UNSET` sentinel distinguishes
"omit this field" from `category=None` meaning "explicitly clear" (same
semantics as FilesystemMemoryBackend)."""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from sqlalchemy import (
    DECIMAL,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    or_,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from linktools.core import environ

from ....errors import MemoryConflictError, MemoryNotFoundError
from ....storage.sqlalchemy.base import Base
from ....storage.sqlalchemy.conventions import TABLE_PREFIX, as_utc, timestamp_indexes
from ....storage.sqlalchemy.dialects import resolve_dialect
from ..models import MemoryMatch, MemoryRecord
from ..scope import LEGACY_TENANT_ID, is_legacy_tenant
from ..store import UNSET

logger = environ.get_logger("ai.agent.memory.persistence.sqlalchemy")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ....storage.sqlalchemy.dialects import SqlAlchemyDialect
    from ..scope import MemoryScope


class MemoryRow(Base):
    __tablename__ = f"{TABLE_PREFIX}memories"
    __table_args__ = (
        UniqueConstraint("memory_id", name="uk_memory_id"),
        Index("ix_tenant_id", "tenant_id"),
        *timestamp_indexes(),
    )

    memory_id: "Mapped[str]" = mapped_column(String(128), comment="Memory id")
    tenant_id: "Mapped[str | None]" = mapped_column(
        String(128), nullable=True, comment="Tenant id"
    )
    owner_id: "Mapped[str]" = mapped_column(String(128), comment="Owner id")
    content: "Mapped[str]" = mapped_column(Text, comment="Memory content")
    category: "Mapped[str | None]" = mapped_column(
        String(64), nullable=True, comment="Category"
    )
    confidence: "Mapped[float | None]" = mapped_column(
        DECIMAL(5, 4), nullable=True, comment="Confidence [0,1]"
    )
    version: "Mapped[int]" = mapped_column(Integer, comment="Version (optimistic lock)")
    metadata_json: "Mapped[str]" = mapped_column(Text, comment="Metadata (JSON)")
    user_id: "Mapped[str | None]" = mapped_column(
        String(128), nullable=True, comment="User id"
    )
    workspace_id: "Mapped[str | None]" = mapped_column(
        String(128), nullable=True, comment="Workspace id"
    )
    session_id: "Mapped[str | None]" = mapped_column(
        String(128), nullable=True, comment="Session id"
    )


def _row_to_record(row: MemoryRow) -> MemoryRecord:
    # A NULL tenant_id is a legacy row (pre-tenant). It is read back under the
    # reserved legacy tenant so it never matches a real tenant's search.
    return MemoryRecord(
        id=row.memory_id,
        tenant_id=row.tenant_id or LEGACY_TENANT_ID,
        owner_id=row.owner_id,
        content=row.content,
        category=row.category,
        confidence=None if row.confidence is None else float(row.confidence),
        version=row.version,
        created_at=as_utc(row.created_at),
        updated_at=as_utc(row.updated_at),
        metadata=json.loads(row.metadata_json),
        user_id=row.user_id,
        workspace_id=row.workspace_id,
        session_id=row.session_id,
    )


class SqlAlchemyMemoryBackend:
    """Multi-process MemoryStore backed by SQLAlchemy/AsyncSession.

    Optimistic concurrency on ``update`` / ``forget`` uses the same
    read-check-mutate-commit-in-one-transaction pattern as the other
    SQLAlchemy stores. ``remember`` relies on the primary-key constraint: a duplicate
    id raises ``IntegrityError``, which is translated to ``MemoryConflictError``.
    """

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
        dialect: "SqlAlchemyDialect | None" = None,
    ) -> None:
        self._session_factory = session_factory
        # UoW mode: when set, every method uses this shared session directly and
        # does NOT open its own session or call session.begin() -- the UoW owns
        # the transaction. None means normal mode (own session + transaction).
        self._session = session
        self._dialect = dialect

    async def _dialect_for(self, session: "AsyncSession") -> "SqlAlchemyDialect":
        if self._dialect is None:
            self._dialect = resolve_dialect(session)
        return self._dialect

    async def _execute_in_session(self, fn):
        """Run ``fn(session)`` in its own transaction (normal mode) or against
        the shared session (UoW mode)."""
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
                select(MemoryRow).where(MemoryRow.memory_id == memory_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)

        return await self._execute_in_session(_do)

    async def search(
        self,
        query: str,
        *,
        scope: "MemoryScope",
        limit: int = 10,
        category: "str | None" = None,
    ) -> "tuple[MemoryMatch, ...]":
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
            # Keyword (content LIKE) search carries no ranking signal; every hit
            # is returned with score=None rather than a fabricated value.
            return tuple(
                MemoryMatch(record=_row_to_record(row), score=None)
                for row in result.scalars()
            )

        return await self._execute_in_session(_do)

    # -- write ---------------------------------------------------------

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        async def _do(session):
            session.add(
                MemoryRow(
                    memory_id=record.id,
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
            # Duplicate primary key -> conflict, matching FilesystemMemoryBackend's
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
        content: object = UNSET,
        category: object = UNSET,
        confidence: object = UNSET,
        metadata: object = UNSET,
    ) -> MemoryRecord:
        async def _do(session):
            values = {
                "version": expected_version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
            if content is not UNSET:
                values["content"] = content
            if category is not UNSET:
                values["category"] = category
            if confidence is not UNSET:
                values["confidence"] = confidence
            if metadata is not UNSET:
                values["metadata_json"] = json.dumps(metadata)
            # UPDATE ... RETURNING folds the write and the read-back into one
            # statement (UPDATE-then-SELECT on MySQL). Returning every column
            # the record needs avoids a second SELECT round trip.
            dialect = await self._dialect_for(session)
            columns = (
                "memory_id",
                "tenant_id",
                "owner_id",
                "content",
                "category",
                "confidence",
                "version",
                "metadata_json",
                "created_at",
                "updated_at",
                "user_id",
                "workspace_id",
                "session_id",
            )
            rows = await dialect.update_returning(
                session,
                model=MemoryRow,
                where=(
                    (MemoryRow.memory_id == memory_id)
                    & (MemoryRow.version == expected_version)
                ),
                values=values,
                returning=columns,
            )
            if not rows:
                await self._raise_write_conflict(session, memory_id, expected_version)
            # update_returning yields a Row with attribute-per-column access,
            # the same shape _row_to_record reads off an ORM MemoryRow.
            return _row_to_record(rows[0])

        return await self._execute_in_session(_do)

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        async def _do(session):
            result = await session.execute(
                delete(MemoryRow).where(
                    MemoryRow.memory_id == memory_id,
                    MemoryRow.version == expected_version,
                )
            )
            if result.rowcount != 1:
                await self._raise_write_conflict(session, memory_id, expected_version)

        await self._execute_in_session(_do)

    @staticmethod
    async def _raise_write_conflict(
        session: "AsyncSession",
        memory_id: str,
        expected_version: int,
    ) -> None:
        actual = await session.scalar(
            select(MemoryRow.version).where(MemoryRow.memory_id == memory_id)
        )
        if actual is None:
            raise MemoryNotFoundError(f"memory not found: {memory_id}")
        if environ.debug:
            logger.debug(
                "memory write conflict: id=%s expected=%s actual=%s",
                memory_id,
                expected_version,
                actual,
            )
        raise MemoryConflictError(
            f"expected version {expected_version}, found {actual}"
        )
