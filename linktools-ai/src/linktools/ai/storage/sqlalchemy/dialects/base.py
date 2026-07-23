#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dialect-strategy base: the portable contract every SqlAlchemy dialect
strategy implements, plus the integrity-violation taxonomy.

A dialect strategy owns the dialect-specific pieces of the insert/conflict
algorithm: it classifies a caught :class:`~sqlalchemy.exc.IntegrityError`, and
it executes a unique-key-conflict-detecting INSERT in the way that dialect
supports. It does NOT replicate Asset CRUD -- the generic concurrency algorithm
lives in the backend and calls :meth:`SqlAlchemyDialectStrategy.execute_conflict_insert`
plus :meth:`SqlAlchemyDialectStrategy.classify_integrity_error`, so the
algorithm itself stays dialect-neutral.

The split exists because SQLite cannot use a SAVEPOINT to absorb a conflicting
INSERT (aiosqlite commits the savepoint immediately, breaking UoW rollback), so
the SQLite and PostgreSQL strategies use ``ON CONFLICT DO NOTHING`` -- which
avoids the IntegrityError entirely -- while the MySQL strategy uses a savepoint
plus classification (MySQL savepoints are transactionally sound)."""

from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from ....errors import StorageError


class IntegrityViolationKind(str, Enum):
    ASSET_KEY = "asset_key"
    IDEMPOTENCY_KEY = "idempotency_key"
    OTHER = "other"


@runtime_checkable
class SqlAlchemyDialectStrategy(Protocol):
    """Per-dialect conflict-classification + conflict-detecting insert.

    ``name`` is the SQLAlchemy dialect name (``sqlite`` / ``mysql`` /
    ``postgresql``); :func:`storage.sqlalchemy.dialects.resolve_dialect_strategy`
    selects the strategy by this name at adapter construction time."""

    name: str

    def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
        """Map a caught ``IntegrityError`` to the asset domain's violation kind.

        Implementations must return :attr:`IntegrityViolationKind.OTHER` for any
        violation that is not one of the two known asset unique-key conflicts
        so the caller can re-raise the original error unchanged."""
        ...

    async def execute_conflict_insert(
        self,
        session: Any,
        table: Any,
        values: "Mapping[str, Any]",
        *,
        index_elements: "list[str]",
    ) -> bool:
        """INSERT ``values`` into ``table``; return ``True`` when a unique-key
        conflict on ``index_elements`` meant the row was NOT inserted (the
        caller reconciles/retries), ``False`` when it was inserted.

        A non-unique IntegrityError must propagate unchanged (never swallowed).
        The mechanism is dialect-specific: SQLite/PostgreSQL use
        ``ON CONFLICT DO NOTHING`` (rowcount signals the conflict, no exception
        -- safe under a surrounding UoW transaction); MySQL uses a SAVEPOINT
        plus :meth:`classify_integrity_error`."""
        ...


class UnsupportedSqlAlchemyDialectError(StorageError):
    """Raised when a session_factory binds an engine whose dialect is not one
    of the explicitly supported SQLite / MySQL / PostgreSQL dialects.

    Constructed eagerly at :class:`SqlAlchemyStorageAdapter` construction so an
    unsupported dialect surfaces immediately rather than on first write."""

    def __init__(self, dialect_name: "str | None", *, reason: str = "") -> None:
        self.dialect_name = dialect_name
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"unsupported SQLAlchemy dialect {dialect_name!r}; "
            f"only sqlite, mysql, and postgresql are supported{detail}"
        )


__all__: "list[str]" = (
    "IntegrityViolationKind",
    "SqlAlchemyDialectStrategy",
    "UnsupportedSqlAlchemyDialectError",
)
