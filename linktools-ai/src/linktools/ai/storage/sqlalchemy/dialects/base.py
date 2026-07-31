#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-first SQLAlchemy dialect.

The dialect is the ONLY seam at which a store or backend adapts to the
concrete SQL vendor it runs against. It is generic over every SQL store with
a create-only conflict-detecting insert (the storage kernel's object backend,
the artifact-record store, ...) -- none of them hardcode vendor-specific SQL
of their own; they all share this one seam for:

- ``insert_ignore_conflict`` -- INSERT ``values`` into any mapped model's
  table if no row conflicts on the given unique columns; return whether the
  insert landed.
- ``upsert_increment`` -- self-seed a singleton counter row keyed by ``pk``
  and atomically add ``step`` to its ``column`` in one statement, returning the
  new value (SQLite/PostgreSQL via ``INSERT ... ON CONFLICT ... RETURNING``;
  MySQL via ``INSERT ... ON DUPLICATE KEY UPDATE col = LAST_INSERT_ID(col +
  step)`` read back from the driver's ``lastrowid``).
- ``classify_integrity_error`` -- map a caught SQLAlchemy IntegrityError to
  one of the kernel-level integrity kinds so the caller can re-raise the
  original error for non-unique-constraint violations.

Core ships three concrete implementations -- :class:`SqliteDialect`,
:class:`MySQLDialect`, :class:`PostgreSQLDialect` -- for SQLite/aiosqlite,
MySQL, and PostgreSQL respectively. A caller normally never constructs one
directly: every store/backend accepts an optional ``dialect`` and, when it is
not given, lazily resolves the right built-in on first use via
:func:`~linktools.ai.storage.sqlalchemy.dialects.resolve_dialect` (keyed off
an open session's bound engine's ``dialect.name``) -- this happens once,
inside the dialects package, not by the store/backend branching on the name
itself. A downstream wanting a vendor with no built-in ships its own dialect
implementation and passes it explicitly."""


from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class InsertResult:
    """Outcome of an ``insert_ignore_conflict`` attempt. ``inserted`` is True
    when a fresh row was committed; False when a row already existed with the
    same unique-column values (so the caller reconciles against that row).
    ``row_id`` carries the auto-generated primary key when the dialect's
    RETURNING clause supplied it (SQLite/PostgreSQL); None when the dialect
    has no RETURNING support (MySQL) — the caller then does a fallback SELECT
    if it needs the id."""

    inserted: bool
    row_id: "int | None" = None


class IntegrityViolationKind(str, Enum):
    """The vendor-agnostic taxonomy for a caught IntegrityError. The kernel's
    object backend only cares whether a write lost a unique-key race; the
    finer-grained FOREIGN_KEY / CHECK / UNKNOWN kinds exist so a downstream
    dialect can surface structural integrity errors distinctly."""

    UNIQUE_CONFLICT = "unique_conflict"
    FOREIGN_KEY = "foreign_key"
    CHECK = "check"
    UNKNOWN = "unknown"


@runtime_checkable
class SqlAlchemyDialect(Protocol):
    """Per-vendor dialect contract shared by every SQL store/backend.

    A caller may inject one explicitly (a test double, or a vendor with no
    built-in); otherwise :func:`resolve_dialect` supplies the matching
    built-in. ``name`` is informational (a downstream tags its own dialect;
    core does not branch on it)."""

    @property
    def name(self) -> str:
        ...

    async def insert_ignore_conflict(
        self,
        session: "AsyncSession",
        *,
        model: type,
        values: "Mapping[str, Any]",
        index_elements: "Sequence[str]",
    ) -> InsertResult:
        """INSERT ``values`` into ``model``'s table iff no row conflicts on
        the unique constraint covering ``index_elements``; return
        ``InsertResult(inserted=True)`` when the row landed,
        ``InsertResult(inserted=False)`` when a conflicting row beat this
        call. A non-unique IntegrityError must propagate unchanged; only the
        targeted unique-key case is absorbed."""
        ...

    async def upsert_increment(
        self,
        session: "AsyncSession",
        *,
        model: type,
        pk: Any,
        column: str,
        step: int = 1,
    ) -> int:
        """Self-seed ``model``'s singleton row keyed by ``pk`` and add ``step``
        to ``column`` in one statement, returning the resulting value. The row
        is created on the first call with ``column = step`` (the first-increment
        value); a unique-key conflict advances ``column = column + step``
        instead. The singleton can therefore never be absent, so this never
        returns ``None``.

        This is the vendor seam for an atomic read-modify-write on a
        self-seeding counter: SQLite/PostgreSQL fold it into one
        ``INSERT ... ON CONFLICT (pk) DO UPDATE ... RETURNING`` round trip;
        MySQL (no portable RETURNING) wraps the upsert in a
        ``LAST_INSERT_ID(expr)`` idiom so the incremented value is observable
        via the driver's ``lastrowid`` in a single statement. The atomic
        increment itself (``column = column + step``) is enforced by the
        database under row locking, so concurrent callers never lose an update."""
        ...

    def classify_integrity_error(
        self,
        error: BaseException,
    ) -> IntegrityViolationKind:
        """Map a caught IntegrityError to the kernel-level kind. Implementations
        should classify the original driver-level error (``error.orig`` for a
        SQLAlchemy IntegrityError) and return ``UNKNOWN`` for anything they do
        not specifically recognize so the caller re-raises the original."""
        ...


def classify_integrity_error_by_message(
    error: BaseException,
    *,
    unique_markers: "Sequence[str]",
    foreign_key_markers: "Sequence[str]" = ("foreign key constraint",),
    check_markers: "Sequence[str]" = ("check constraint",),
) -> IntegrityViolationKind:
    """Shared message-sniffing fallback for dialects whose structured
    error-code path missed. Every vendor-specific dialect's
    ``classify_integrity_error`` ends with the same lowercased-substring
    check; this helper owns that shape once so the vendor files only supply
    their own marker lists. ``unique_markers`` is vendor-specific (the
    wording differs between SQLite's "unique constraint", MySQL's "duplicate
    entry", and PostgreSQL's "duplicate key"); the FK/check markers happen to
    match across the three built-ins and default to the common substrings."""
    orig = getattr(error, "orig", None)
    message = str(orig or error).lower()
    if any(m in message for m in unique_markers):
        return IntegrityViolationKind.UNIQUE_CONFLICT
    if any(m in message for m in foreign_key_markers):
        return IntegrityViolationKind.FOREIGN_KEY
    if any(m in message for m in check_markers):
        return IntegrityViolationKind.CHECK
    return IntegrityViolationKind.UNKNOWN


def primary_key_column(model: type) -> Any:
    """Return the single primary-key :class:`InstrumentedAttribute` of a mapped
    ``model``. Shared by the ``upsert_increment`` implementations (every
    built-in keys the singleton counter row on its PK). A model with a
    composite primary key has no single keying column and this raises
    ``ValueError`` -- ``upsert_increment`` targets singleton counter rows,
    none of which have composite keys."""
    table = model.__table__
    pk_columns = list(table.primary_key.columns)
    if len(pk_columns) != 1:
        raise ValueError(
            f"{model.__name__} has {len(pk_columns)} primary-key columns; "
            f"upsert_increment keys a singleton row on a single PK"
        )
    return getattr(model, pk_columns[0].name)


__all__: "list[str]" = (
    "InsertResult",
    "IntegrityViolationKind",
    "SqlAlchemyDialect",
    "classify_integrity_error_by_message",
    "primary_key_column",
)
