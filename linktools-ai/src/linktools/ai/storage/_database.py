"""Domain-independent SQL context, metadata primitives, and validation."""

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, NoReturn, TypeVar

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._dialects import (
    SqlAlchemyDialect,
    SqlTransactionDisposition,
    SqlTransactionPhase,
    column_type_matches,
    configure_sqlite_engine,
    dialect_for_name,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.storage.database")
ValueT = TypeVar("ValueT")


@dataclass(slots=True)
class SqlStorageContext:
    """The borrowed-engine boundary used by SQL adapters."""

    engine: "AsyncEngine"
    sessions: "async_sessionmaker[AsyncSession]"
    dialect: SqlAlchemyDialect
    owns_engine: bool = False
    _closed: bool = field(default=False, init=False, repr=False)
    _initialize_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _sqlite_configured: bool = field(default=False, init=False, repr=False)
    _validated_metadata: "MetaData | None" = field(default=None, init=False, repr=False)

    async def initialize(self, *, metadata: "MetaData | None" = None) -> None:
        async with self._initialize_lock:
            if self._closed:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            if self.engine.dialect.name == "sqlite" and not self._sqlite_configured:
                await configure_sqlite_engine(self.engine)
                self._sqlite_configured = True
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
        if self._closed:
            return
        self._closed = True
        if self.owns_engine:
            await self.engine.dispose()

    async def run_mutation(
        self,
        callback: Callable[["AsyncSession"], Awaitable[ValueT]],
        *,
        retry_limit: int = 8,
        domain: str = "sql",
    ) -> ValueT:
        if retry_limit < 1:
            raise ValueError("retry_limit must be positive")
        started = monotonic()
        for attempt in range(retry_limit):
            session = self.sessions()
            transaction = session.begin()
            entered = False
            try:
                await transaction.__aenter__()
                entered = True
                try:
                    result = await callback(session)
                except BaseException as error:
                    disposition = self.dialect.classify_transaction_error(
                        error,
                        phase=SqlTransactionPhase.BODY,
                        connection_invalidated=_connection_invalidated(error),
                    )
                    await transaction.__aexit__(*sys.exc_info())
                    if disposition is SqlTransactionDisposition.RETRYABLE_ABORTED and attempt + 1 < retry_limit:
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
                    disposition = self.dialect.classify_transaction_error(
                        error,
                        phase=SqlTransactionPhase.COMMIT,
                        connection_invalidated=_connection_invalidated(error),
                    )
                    if disposition is SqlTransactionDisposition.RETRYABLE_ABORTED and attempt + 1 < retry_limit:
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
                if not entered:
                    await transaction.__aexit__(*sys.exc_info())
                await session.close()
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    @property
    def closed(self) -> bool:
        return self._closed


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

    dialect_for_name(engine.dialect.name)
    if engine.dialect.name == "sqlite":
        await configure_sqlite_engine(engine)
    if not metadata.tables:
        return
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    await validate_sql(engine, metadata)
    _logger.info("SQL metadata provisioned: dialect=%s tables=%s", engine.dialect.name, len(metadata.tables))


async def validate_sql(engine: "AsyncEngine", metadata: "MetaData") -> None:
    """Validate that required metadata is a compatible subset of the database."""

    dialect_for_name(engine.dialect.name)
    if engine.dialect.name == "sqlite":
        await configure_sqlite_engine(engine)
    await _validate_sql_schema(engine, metadata)
    _logger.debug("SQL metadata validated: dialect=%s tables=%s", engine.dialect.name, len(metadata.tables))


async def _validate_sql_schema(engine: "AsyncEngine", metadata: "MetaData") -> None:
    if not metadata.tables:
        return
    await asyncio.gather(*(_validate_table_schema(engine, table) for table in metadata.tables.values()))


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


def _schema_mismatch(*, table: str, category: str, column: str | None = None) -> NoReturn:
    details: dict[str, object] = {"table": table, "category": category}
    if column is not None:
        details["column"] = column
    raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, safe_details=details)


def _connection_invalidated(error: BaseException) -> bool:
    from sqlalchemy.exc import DBAPIError

    return isinstance(error, DBAPIError) and error.connection_invalidated


__all__ = [
    "SqlStorageContext",
    "create_sql_storage_context",
    "provision_sql",
    "validate_sql",
]
