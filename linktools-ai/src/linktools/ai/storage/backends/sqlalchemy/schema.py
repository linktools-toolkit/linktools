#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schema provider Protocol + the SQLite reference implementation.

The storage kernel's object backend NEVER calls ``Base.metadata.create_all``
and NEVER modifies the shared declarative ``Base.metadata``. A schema provider
is the seam at which a downstream that owns the engine + driver also owns the
DDL: it must guarantee the schema the backend expects is present before the
backend's first write, or fail fast.

Core ships :class:`SqliteReferenceSchemaProvider` for tests and the local
single-process reference deployment -- it can ``create_for_tests_and_local_
reference`` (CREATE TABLE IF NOT EXISTS, no migration framework) and ``validate``
(check the expected tables exist; raise if not). A downstream with its own
migration framework injects a provider that runs the migration + checksum at
construction time.

The Protocol intentionally exposes only ``validate``: the backend does not
care HOW the schema got there, only that it IS there. ``create_for_tests_and_
local_reference`` is a SQLite-reference convenience, not part of the Protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

if False:  # TYPE_CHECKING
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@runtime_checkable
class SqlAlchemySchemaProvider(Protocol):
    """The contract the object backend relies on at construction or first use:
    the schema the backend's queries assume MUST be present, or this raises."""

    async def validate(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
    ) -> None:
        ...


# The table names the object backend queries. ``validate`` checks every one of
# them exists; a missing or extra table is a fail-fast condition. Kept here
# (not in the models module) so a downstream reading the Protocol + this list
# has the full contract without importing the backend's ORM.
_REQUIRED_TABLES = (
    "storage_objects",
    "storage_object_versions",
    "storage_object_revision",
    "storage_object_idempotency",
)


class SqliteReferenceSchemaProvider:
    """The SQLite reference schema provider. ``create_for_tests_and_local_
    reference`` runs CREATE TABLE IF NOT EXISTS for the object backend's
    metadata -- NO migration framework, NO versioning. ``validate`` checks the
    expected tables are present (raises if any are missing). Production use
    injects a provider that runs a real migration + checksum."""

    async def create_for_tests_and_local_reference(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
    ) -> None:
        from sqlalchemy import inspect

        from .models import Base

        async with session_factory() as session:
            conn = await session.connection()
            await conn.run_sync(Base.metadata.create_all)
            await session.commit()

    async def validate(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
    ) -> None:
        from sqlalchemy import inspect

        async with session_factory() as session:
            conn = await session.connection()
            existing = await conn.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )
        missing = [t for t in _REQUIRED_TABLES if t not in existing]
        if missing:
            raise StorageSchemaNotReadyError(
                f"SQLite reference schema is missing required tables: "
                f"{sorted(missing)}; call create_for_tests_and_local_reference() "
                f"first, or inject a schema provider that runs your migration"
            )


class StorageSchemaNotReadyError(Exception):
    """The schema the object backend expects is not present at validate time."""


__all__: "list[str]" = (
    "SqlAlchemySchemaProvider",
    "SqliteReferenceSchemaProvider",
    "StorageSchemaNotReadyError",
)
