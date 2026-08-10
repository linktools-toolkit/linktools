#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit SQL schema provisioning and runtime validation."""

from typing import TYPE_CHECKING, Protocol

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._database import StorageDatabase, sql_constraint_signature

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


class _SqlTypeValue(Protocol):
    def __str__(self) -> str: ...


_logger = environ.get_logger("ai.storage.initialize")


async def initialize_storage(database: StorageDatabase) -> None:
    """Validate an existing storage schema before runtime access."""
    if not database.schema_manifest_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    await validate_schema(database.session_factory, database.metadata)


async def validate_schema(
    session_factory: "async_sessionmaker[AsyncSession]",
    metadata: "MetaData",
) -> None:
    """Validate owned tables without issuing schema-changing statements."""
    engine = await _resolve_engine(session_factory)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(_validate_schema, metadata)
    except AIError:
        _logger.exception("SQL schema validation failed: table_count=%s", len(metadata.tables))
        raise
    _logger.info("SQL schema validated: table_count=%s", len(metadata.tables))


async def _resolve_engine(session_factory: "async_sessionmaker[AsyncSession]") -> "AsyncEngine":
    from sqlalchemy.ext.asyncio import AsyncEngine

    async with session_factory() as session:
        bound = session.bind
    if not isinstance(bound, AsyncEngine):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return bound


def _validate_schema(connection: "Connection", metadata: "MetaData") -> None:
    from sqlalchemy import inspect
    inspector = inspect(connection)
    actual_tables = set(inspector.get_table_names())
    expected_tables = set(metadata.tables)
    if not expected_tables.issubset(actual_tables):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for table_name, table in metadata.tables.items():
        primary_key = set(inspector.get_pk_constraint(table_name).get("constrained_columns", ()))
        actual_columns = {
            f"{column['name']}:{_type_name(column['type'], column['name'])}:{int(bool(column['nullable']))}:{int(column['name'] in primary_key)}"
            for column in inspector.get_columns(table_name)
        }
        expected_columns = {
            f"{column.name}:{_type_name(column.type.dialect_impl(connection.dialect), column.name)}:{int(bool(column.nullable))}:{int(column.primary_key)}"
            for column in table.columns
        }
        if actual_columns != expected_columns:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_constraints = {
            sql_constraint_signature(constraint)
            for constraint in table.constraints
        }
        actual_constraints = {
            f"PrimaryKeyConstraint::{','.join(inspector.get_pk_constraint(table_name).get('constrained_columns', ())) }:"
        }
        actual_constraints.update(
            f"CheckConstraint:{item.get('name') or ''}::{item.get('sqltext') or ''}"
            for item in inspector.get_check_constraints(table_name)
        )
        actual_constraints.update(
            f"UniqueConstraint:{item.get('name') or ''}:{','.join(item.get('column_names', ())) }:"
            for item in inspector.get_unique_constraints(table_name)
        )
        actual_constraints.update(
            f"ForeignKeyConstraint:{item.get('name') or ''}:{','.join(item.get('constrained_columns', ())) }:"
            for item in inspector.get_foreign_keys(table_name)
        )
        if actual_constraints != expected_constraints:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        actual_indexes = {
            f"{item.get('name') or ''}:{','.join(item.get('column_names', ())) }"
            for item in inspector.get_indexes(table_name)
            if not item.get("unique")
        }
        expected_indexes = {
            index.name + ":" + ",".join(column.name for column in index.columns)
            for index in table.indexes
            if index.name is not None and index.info.get("ddl_dialect", connection.dialect.name) == connection.dialect.name
        }
        if actual_indexes != expected_indexes:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _type_name(value: _SqlTypeValue, column_name: "str | None" = None) -> str:
    name = str(value)
    normalized = name.upper()
    if "JSON" in normalized or (column_name == "payload" and normalized in {"LONGTEXT", "TEXT"}):
        return "JSON"
    if normalized in {"DATETIME", "TIMESTAMP"}:
        return "TIMESTAMP"
    return name


__all__ = ["initialize_storage", "validate_schema"]
