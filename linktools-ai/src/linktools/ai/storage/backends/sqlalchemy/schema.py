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

from datetime import datetime, timezone
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
    "storage_schema_version",
)

_REQUIRED_COLUMNS = {
    "storage_objects": {"key", "key_hash", "version", "commit_revision", "content"},
    "storage_object_versions": {"key", "key_hash", "version", "commit_revision", "content"},
    "storage_object_idempotency": {"key", "key_hash", "operation", "request_hash", "result_key", "result_version", "commit_revision"},
    "storage_schema_version": {"component", "version", "checksum", "updated_at"},
    "storage_object_revision": {"id", "value"},
}

_REQUIRED_NON_NULL_COLUMNS = {
    "storage_objects": {"key", "key_hash", "version", "content", "commit_revision"},
    "storage_object_versions": {"key", "key_hash", "version", "commit_revision"},
    "storage_object_idempotency": {"key", "key_hash", "operation", "request_hash", "commit_revision"},
    "storage_object_revision": {"id", "value"},
    "storage_schema_version": {"component", "version", "checksum", "updated_at"},
}

_REQUIRED_INDEXES = {
    "storage_objects": "ix_storage_objects_key_hash",
    "storage_object_versions": "ix_storage_object_versions_key_hash",
    "storage_object_idempotency": "ix_storage_object_idempotency_key_hash",
}

_REQUIRED_UNIQUES = {
    "storage_objects": "uq_storage_objects_key_hash",
    "storage_object_versions": "uq_storage_object_versions_key_hash_version",
    "storage_object_idempotency": "uq_storage_object_idempotency_key_hash",
}
_SCHEMA_COMPONENT = "storage_object"
_SCHEMA_CHECKSUM = "storage-object-schema-v1"


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
        from ...sqlalchemy.models import Base as DomainBase
        # Register commit-log tables before the reference DDL is emitted.
        from ....run.persistence.sqlalchemy import commit_log as _run_commit_log
        from ....swarm.persistence import sqlalchemy_commit as _swarm_commit_log

        async with session_factory() as session:
            conn = await session.connection()
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(DomainBase.metadata.create_all)
            from .models import StorageSchemaVersionRow
            from sqlalchemy import select
            existing = await session.scalar(
                select(StorageSchemaVersionRow).where(
                    StorageSchemaVersionRow.component == _SCHEMA_COMPONENT
                )
            )
            if existing is None:
                session.add(StorageSchemaVersionRow(
                    component=_SCHEMA_COMPONENT,
                    version=1,
                    checksum=_SCHEMA_CHECKSUM,
                    updated_at=datetime.now(timezone.utc),
                ))
            await session.commit()

    async def validate(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
    ) -> None:
        from sqlalchemy import inspect

        async with session_factory() as session:
            conn = await session.connection()
            def inspect_schema(sync_conn):
                inspector = inspect(sync_conn)
                tables = set(inspector.get_table_names())
                present = tables.intersection(_REQUIRED_TABLES)
                return (
                    tables,
                    {
                        table: {
                            c["name"]: c for c in inspector.get_columns(table)
                        }
                        for table in present
                    },
                    {
                        table: {i["name"] for i in inspector.get_indexes(table)}
                        for table in present
                    },
                    {
                        table: inspector.get_unique_constraints(table)
                        for table in present
                    },
                    {
                        table: inspector.get_pk_constraint(table).get(
                            "constrained_columns", []
                        )
                        for table in present
                    },
                )

            existing, columns, indexes, uniques, primary_keys = await conn.run_sync(
                inspect_schema
            )
        missing = [t for t in _REQUIRED_TABLES if t not in existing]
        if missing:
            raise StorageSchemaNotReadyError(
                f"SQLite reference schema is missing required tables: "
                f"{sorted(missing)}; call create_for_tests_and_local_reference() "
                f"first, or inject a schema provider that runs your migration"
            )
        malformed = {
            table: sorted(required - set(columns.get(table, {})))
            for table, required in _REQUIRED_COLUMNS.items()
            if required - set(columns.get(table, {}))
        }
        if malformed:
            raise StorageSchemaNotReadyError(f"SQLite schema is missing required columns: {malformed}")
        missing_indexes = {
            table: name
            for table, name in _REQUIRED_INDEXES.items()
            if name not in indexes.get(table, set())
        }
        if missing_indexes:
            raise StorageSchemaNotReadyError(
                f"SQLite schema is missing required key-hash indexes: {missing_indexes}"
            )
        missing_non_null = {
            table: sorted(
                column
                for column in required
                if columns.get(table, {}).get(column, {}).get("nullable", True)
            )
            for table, required in _REQUIRED_NON_NULL_COLUMNS.items()
            if any(
                columns.get(table, {}).get(column, {}).get("nullable", True)
                for column in required
            )
        }
        if missing_non_null:
            raise StorageSchemaNotReadyError(
                f"SQLite schema has nullable required columns: {missing_non_null}"
            )
        missing_primary_keys = {
            table: sorted(required)
            for table, required in {
                "storage_objects": {"id"},
                "storage_object_versions": {"id"},
                "storage_object_idempotency": {"id"},
                "storage_object_revision": {"id"},
                "storage_schema_version": {"component"},
            }.items()
            if not required.intersection(primary_keys.get(table, []))
        }
        if missing_primary_keys:
            raise StorageSchemaNotReadyError(
                f"SQLite schema is missing required primary keys: {missing_primary_keys}"
            )
        for table, required_name in _REQUIRED_UNIQUES.items():
            names = {u.get("name") for u in uniques.get(table, ())}
            if required_name not in names:
                raise StorageSchemaNotReadyError(
                    f"SQLite schema is missing required unique constraint {required_name!r}"
                )
        async with session_factory() as session:
            from sqlalchemy import select
            from .models import StorageSchemaVersionRow
            row = await session.scalar(
                select(StorageSchemaVersionRow).where(
                    StorageSchemaVersionRow.component == _SCHEMA_COMPONENT
                )
            )
            if row is None or row.version != 1 or row.checksum != _SCHEMA_CHECKSUM:
                raise StorageSchemaNotReadyError(
                    "SQLite schema version/checksum is missing or invalid"
                )


class StorageSchemaNotReadyError(Exception):
    """The schema the object backend expects is not present at validate time."""


__all__: "list[str]" = (
    "SqlAlchemySchemaProvider",
    "SqliteReferenceSchemaProvider",
    "StorageSchemaNotReadyError",
)
