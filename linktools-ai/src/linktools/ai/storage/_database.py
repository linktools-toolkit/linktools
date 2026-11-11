#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain-independent SQL context, metadata primitives, and validation."""

import asyncio
import sqlite3
import sys
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from operator import index
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, ClassVar, NoReturn, TypeVar

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._sqlite_owner import _LOCAL_PROFILE, _SqliteFileOwner, _connection_file, _finish_owned_io, require_sqlite_write
from ._dialects import (
    _SQLITE_BUSY_TIMEOUT_MS,
    _is_sqlite_busy,
    SqlAlchemyDialect,
    SqlTransactionDisposition,
    SqlTransactionPhase,
    column_type_matches,
    dialect_for_name,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.storage.database")
ValueT = TypeVar("ValueT")


class SqlSnapshotConflict(AIError):
    """An optimistic SQL write proved this transaction's snapshot is stale."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.STORAGE_CONFLICT, "SQL publication snapshot changed")


@dataclass(slots=True)
class SqlStorageContext:
    """The borrowed-engine boundary used by SQL adapters."""

    engine: "AsyncEngine"
    sessions: "async_sessionmaker[AsyncSession]"
    dialect: SqlAlchemyDialect
    owns_engine: bool = False
    LOCAL_SQLITE_APPLICATION_ID: ClassVar[int] = _LOCAL_PROFILE
    _sqlite_owner: _SqliteFileOwner | None = field(default=None, init=False, repr=False)
    _local_mutations: dict["AsyncSession", bool] = field(default_factory=dict, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _initialize_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _engine_configured: bool = field(default=False, init=False, repr=False)
    _validated_metadata: "MetaData | None" = field(default=None, init=False, repr=False)

    async def initialize(self, *, metadata: "MetaData | None" = None) -> None:
        async with self._initialize_lock:
            if self._closed:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            if not self._engine_configured:
                await self.dialect.configure_engine(self.engine)
                self._engine_configured = True
            if metadata is not None and metadata is not self._validated_metadata:
                await _validate_sql_schema(self.engine, metadata)
                self._validated_metadata = metadata
        _logger.debug(
            "SQL context initialized: dialect=%s owns_engine=%s metadata=%s",
            self.dialect.name,
            self.owns_engine,
            0 if metadata is None else len(metadata.tables),
        )

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            if any(self._local_mutations.values()):
                raise AIError(ErrorCode.STORAGE_CONFLICT, "Local SQLite mutations must settle before close")
            self._closed = True
            if self._sqlite_owner is not None:
                self._sqlite_owner.stop_writes()
            try:
                for session in tuple(self._local_mutations):
                    await _finish_owned_io(asyncio.create_task(session.close()))
                    self._local_mutations.pop(session)
                if self.owns_engine:
                    await self.engine.dispose()
                if self._sqlite_owner is not None:
                    await self._sqlite_owner.close()
            except BaseException:
                self._closed = False
                raise

    async def acquire_local_sqlite(self, path: str | Path) -> None:
        """Hold the native writer lock before provisioning or opening a local source."""
        async with self._initialize_lock:
            if self._closed:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            if not self.owns_engine or self.engine.dialect.name != "sqlite":
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            if self._sqlite_owner is None:
                self._sqlite_owner = _SqliteFileOwner(path)
            elif self._sqlite_owner.path != Path(path).expanduser().resolve():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._sqlite_owner.acquire()

    async def is_local_sqlite_source(self, path: str | Path) -> bool:
        """Inspect an existing fixed profile without granting writer permission."""
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self.engine.dialect.name != "sqlite":
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        async with self.engine.connect() as connection:
            source = await _connection_file(connection)
        if source.path != Path(path).expanduser().resolve():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return source.profile == _LOCAL_PROFILE

    @property
    def exclusive_writer(self) -> bool:
        return not self._closed and self._sqlite_owner is not None and self._sqlite_owner.active

    async def require_write_access(self) -> None:
        """Check a writer on its actual connection, not its engine URL."""
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if self._sqlite_owner is not None:
            self._sqlite_owner.require_open()
        if self.engine.dialect.name == "sqlite" or self._sqlite_owner is not None:
            async with self.engine.connect() as connection:
                await require_sqlite_write(connection, self._sqlite_owner)

    async def run_statement(self, operation: Awaitable[ValueT]) -> ValueT:
        """Settle SQLite statement I/O before propagating caller cancellation.

        Use for buffered calls or owned stream acquisition, not transaction callbacks.
        """
        if not self.dialect.single_writer:
            return await operation
        # An interrupted SQLite fetch can retain its cursor and lock in the
        # cancellation traceback. Finish statement I/O before unwinding the session.
        task = asyncio.ensure_future(operation)
        try:
            await asyncio.wait((task,))
            return task.result()
        except asyncio.CancelledError as cancelled:
            while not task.done():
                try:
                    await asyncio.wait((task,))
                except asyncio.CancelledError:
                    pass
            try:
                task.result()
            except BaseException as error:
                _release_sqlite_error_frames(error)
                raise cancelled from error
            raise
        except BaseException as error:
            _release_sqlite_error_frames(error)
            raise

    async def run_mutation(
        self,
        callback: Callable[["AsyncSession"], Awaitable[ValueT]],
        *,
        retry_limit: int | None = None,
        domain: str = "sql",
    ) -> ValueT:
        """Retry known aborts; explicit limits cap attempts on every dialect.

        Default SQLite BUSY retries admit new attempts within its busy window.
        In-flight I/O is still settled, not cancelled when that window expires.
        """
        from sqlalchemy.exc import DBAPIError

        if retry_limit is not None:
            retry_limit = index(retry_limit)
            if retry_limit < 1:
                raise ValueError("retry_limit must be positive")
        started = monotonic()
        busy_deadline = started + _SQLITE_BUSY_TIMEOUT_MS / 1000
        busy_error: BaseException | None = None
        retry_delay = 0.005
        attempt = -1

        def can_retry(error: BaseException) -> bool:
            nonlocal busy_error
            if (retry_limit is None and self.engine.dialect.name == "sqlite"
                    and self.dialect.name == "sqlite" and self.dialect.single_writer and isinstance(error, DBAPIError)
                    and _is_sqlite_busy(error.orig)):
                busy_error = error
                return monotonic() < busy_deadline
            busy_error = None
            return attempt + 1 < (8 if retry_limit is None else retry_limit)

        while True:
            attempt += 1
            if attempt and self.dialect.single_writer:
                # Release the previous snapshot/session before giving another
                # SQLite writer time to finish instead of reacquiring read locks.
                delay = retry_delay
                if busy_error is not None:
                    remaining = busy_deadline - monotonic()
                    if remaining <= 0:
                        raise busy_error
                    delay = min(delay, remaining)
                await asyncio.sleep(delay)
                if busy_error is not None and monotonic() >= busy_deadline:
                    raise busy_error
                busy_error = None
                retry_delay = min(retry_delay * 2, 0.05)
            if self._closed:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            if self._sqlite_owner is not None:
                self._sqlite_owner.require_open()
            session = self.sessions()
            transaction = session.begin()
            if self._sqlite_owner is not None:
                self._local_mutations[session] = True
            entered = False
            try:
                await transaction.__aenter__()
                entered = True
                try:
                    if self.engine.dialect.name == "sqlite" or self._sqlite_owner is not None:
                        await require_sqlite_write(await session.connection(), self._sqlite_owner)
                    if self.dialect.single_writer:
                        # SQLite's legacy driver otherwise defers BEGIN until the
                        # first write, allowing prerequisite reads to disagree.
                        await self.dialect.begin_consistent_read(session)
                    result = await callback(session)
                except BaseException as error:
                    _release_sqlite_error_frames(error)
                    disposition = (SqlTransactionDisposition.RETRYABLE_ABORTED
                                   if isinstance(error, SqlSnapshotConflict) else
                                   self.dialect.classify_transaction_error(
                                       error, phase=SqlTransactionPhase.BODY,
                                       connection_invalidated=_connection_invalidated(error),
                                   ))
                    await transaction.__aexit__(*sys.exc_info())
                    if disposition is SqlTransactionDisposition.RETRYABLE_ABORTED and can_retry(error):
                        _logger.warning(
                            "retrying SQL mutation after aborted body: domain=%s dialect=%s attempt=%s",
                            domain,
                            self.dialect.name,
                            attempt + 1,
                        )
                        continue
                    raise
                try:
                    await transaction.__aexit__(None, None, None)
                except BaseException as error:
                    _release_sqlite_error_frames(error)
                    disposition = self.dialect.classify_transaction_error(
                        error,
                        phase=SqlTransactionPhase.COMMIT,
                        connection_invalidated=_connection_invalidated(error),
                    )
                    if disposition is SqlTransactionDisposition.RETRYABLE_ABORTED and can_retry(error):
                        _logger.warning(
                            "retrying SQL mutation after aborted commit: domain=%s dialect=%s attempt=%s",
                            domain,
                            self.dialect.name,
                            attempt + 1,
                        )
                        continue
                    if disposition is SqlTransactionDisposition.COMMIT_UNKNOWN:
                        _logger.error(
                            "SQL mutation commit outcome unknown: domain=%s dialect=%s "
                            "attempt=%s duration_ms=%.3f outcome=unknown",
                            domain,
                            self.dialect.name,
                            attempt + 1,
                            (monotonic() - started) * 1000,
                        )
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error
                    raise
                _logger.debug(
                    "SQL mutation committed: domain=%s dialect=%s attempt=%s duration_ms=%.3f outcome=committed",
                    domain,
                    self.dialect.name,
                    attempt + 1,
                    (monotonic() - started) * 1000,
                )
                return result
            finally:
                try:
                    if not entered:
                        await transaction.__aexit__(*sys.exc_info())
                finally:
                    if self._sqlite_owner is None:
                        await session.close()
                    else:
                        try:
                            await _finish_owned_io(asyncio.create_task(session.close()))
                        except BaseException:
                            self._local_mutations[session] = False
                            raise
                        else:
                            self._local_mutations.pop(session)

    @property
    def closed(self) -> bool:
        return self._closed


def _release_sqlite_error_frames(error: BaseException) -> None:
    from sqlalchemy.exc import DBAPIError

    if isinstance(error, DBAPIError) and isinstance(error.orig, sqlite3.Error):
        # A failed adapter call can retain its native cursor in exited frames.
        # Release it before this connection can be reused for a blocking COMMIT;
        # later GC on the event loop could otherwise prevent readers releasing.
        # Keep exception identity, database codes and the traceback stack intact.
        traceback.clear_frames(error.__traceback__)


def create_sqlite_storage_context(path: str | Path) -> SqlStorageContext:
    """Create an owned engine whose native failed cursors close on their worker."""
    from sqlalchemy import URL
    from sqlalchemy.ext.asyncio import create_async_engine

    from ._sqlite_driver import _OwnedConnection

    engine = create_async_engine(
        URL.create("sqlite+aiosqlite", database=str(path)),
        connect_args={"factory": _OwnedConnection},
    )
    return create_sql_storage_context(engine, owns_engine=True)


def create_sql_storage_context(engine: "AsyncEngine", *, owns_engine: bool = False) -> SqlStorageContext:
    dialect = dialect_for_name(engine.dialect.name)
    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

    if not isinstance(engine, AsyncEngine):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return SqlStorageContext(
        engine=engine,
        sessions=async_sessionmaker(engine, expire_on_commit=False),
        dialect=dialect,
        owns_engine=owns_engine,
    )


async def provision_sql(engine: "AsyncEngine", metadata: "MetaData") -> None:
    """Create the explicitly requested metadata without a global schema."""

    dialect = dialect_for_name(engine.dialect.name)
    await dialect.configure_engine(engine)
    if not metadata.tables:
        return
    async with engine.begin() as connection:
        await require_sqlite_write(connection)
        await connection.run_sync(metadata.create_all)
    await validate_sql(engine, metadata)
    _logger.info("SQL metadata provisioned: dialect=%s tables=%s", engine.dialect.name, len(metadata.tables))


async def validate_sql(engine: "AsyncEngine", metadata: "MetaData") -> None:
    """Validate that required metadata is a compatible subset of the database."""

    dialect = dialect_for_name(engine.dialect.name)
    await dialect.configure_engine(engine)
    await _validate_sql_schema(engine, metadata)
    _logger.debug("SQL metadata validated: dialect=%s tables=%s", engine.dialect.name, len(metadata.tables))


async def _validate_sql_schema(engine: "AsyncEngine", metadata: "MetaData") -> None:
    if not metadata.tables:
        return
    # Validation is an initialization/fail-fast path. Validate sequentially so
    # one schema mismatch cannot return while sibling validation coroutines
    # still hold checked-out connections that race engine disposal.
    for table in metadata.tables.values():
        await _validate_table_schema(engine, table)


async def _validate_table_schema(engine: "AsyncEngine", table: "Table") -> None:
    async with engine.connect() as connection:
        await connection.run_sync(_validate_connection_table, table)


def _validate_connection_table(connection: "Connection", table: "Table") -> None:
    from sqlalchemy import inspect

    inspector = inspect(connection)
    if not inspector.has_table(table.name, schema=table.schema):
        _schema_mismatch(table=table.name, category="table")

    actual_columns = {
        str(column["name"]): column
        for column in inspector.get_columns(table.name, schema=table.schema)
    }
    for expected_column in table.columns:
        actual_column = actual_columns.get(expected_column.name)
        if actual_column is None:
            _schema_mismatch(table=table.name, category="column", column=expected_column.name)
        if not column_type_matches(connection, expected=expected_column, actual=actual_column):
            _schema_mismatch(table=table.name, category="type", column=expected_column.name)

    expected_primary = tuple(column.name for column in table.primary_key.columns)
    actual_primary = tuple(
        str(column)
        for column in (
            inspector.get_pk_constraint(table.name, schema=table.schema).get(
                "constrained_columns"
            )
            or ()
        )
    )
    if expected_primary and actual_primary != expected_primary:
        _schema_mismatch(
            table=table.name,
            category="primary_key",
            columns=expected_primary,
        )

    expected_unique = _expected_unique_columns(connection, table)
    actual_unique = _reflected_unique_columns(inspector, table)
    for columns in expected_unique:
        if frozenset(columns) not in actual_unique:
            _schema_mismatch(
                table=table.name,
                category="unique",
                columns=columns,
            )


def _expected_unique_columns(
    connection: "Connection",
    table: "Table",
) -> tuple[tuple[str, ...], ...]:
    from sqlalchemy import UniqueConstraint

    dialect_name = connection.dialect.name
    values: set[tuple[str, ...]] = set()
    for constraint in table.constraints:
        if not isinstance(constraint, UniqueConstraint):
            continue
        if not _ddl_element_applies(constraint, dialect_name):
            continue
        columns = tuple(column.name for column in constraint.columns)
        if columns:
            values.add(columns)
    for index in table.indexes:
        if not index.unique or not _ddl_element_applies(index, dialect_name):
            continue
        columns = tuple(column.name for column in index.columns)
        if columns:
            values.add(columns)
    return tuple(sorted(values))


def _ddl_element_applies(element: object, dialect_name: str) -> bool:
    info = getattr(element, "info", {})
    target = info.get("ddl_dialect") if isinstance(info, dict) else None
    if target is None:
        return True
    if target == "portable":
        return dialect_name in {"sqlite", "postgresql"}
    return target == dialect_name


def _reflected_unique_columns(inspector: object, table: "Table") -> set[frozenset[str]]:
    values: set[frozenset[str]] = set()
    for constraint in inspector.get_unique_constraints(table.name, schema=table.schema):
        columns = constraint.get("column_names") or ()
        if columns and all(isinstance(column, str) for column in columns):
            values.add(frozenset(columns))
    for index in inspector.get_indexes(table.name, schema=table.schema):
        if index.get("unique") is not True or not _is_full_unique_index(index):
            continue
        columns = index.get("column_names") or ()
        if columns and all(isinstance(column, str) for column in columns):
            values.add(frozenset(columns))
    return values


def _is_full_unique_index(index: object) -> bool:
    if not isinstance(index, dict):
        return False
    options = index.get("dialect_options")
    if not isinstance(options, dict):
        return True
    for key, value in options.items():
        if value in (None, False, {}, ()):
            continue
        if "where" in str(key).lower() or "length" in str(key).lower():
            return False
    return True


def _schema_mismatch(
    *,
    table: str,
    category: str,
    column: str | None = None,
    columns: tuple[str, ...] | None = None,
) -> NoReturn:
    details: dict[str, object] = {"table": table, "category": category}
    if column is not None:
        details["column"] = column
    if columns is not None:
        details["columns"] = list(columns)
    raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, safe_details=details)


def _connection_invalidated(error: BaseException) -> bool:
    from sqlalchemy.exc import DBAPIError

    return isinstance(error, DBAPIError) and error.connection_invalidated


__all__ = [
    "SqlStorageContext",
    "SqlSnapshotConflict",
    "create_sql_storage_context",
    "create_sqlite_storage_context",
    "provision_sql",
    "validate_sql",
]
