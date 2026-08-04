#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared SQLAlchemy conventions for enterprise-DB conformance.

- ``TABLE_PREFIX``: the unified table-name prefix (every ``__tablename__``
  routes through it).
- ``OnUpdateDateTime``: the marker used by the shared declarative ``Base`` for
  ``updated_at``.
- ``sha256_hash``: 32-byte digest used as the uniqueness carrier for wide
  natural keys (commit_id, scope+key, job_id+key) -- mirrors the storage
  kernel's ``key_hash`` pattern.
"""

import hashlib
from datetime import datetime, timezone
from sqlalchemy import BigInteger, DateTime, Index, Integer
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateIndex

TABLE_PREFIX = "ai_"

#: BIGINT AUTO_INCREMENT surrogate-PK type. Uses BigInteger on MySQL/Postgres
#: (the lint-required BIGINT) but Integer on SQLite, because SQLite only treats
#: the exact type ``INTEGER PRIMARY KEY`` as the rowid alias that autoincrements
#: -- ``BIGINT PRIMARY KEY`` would reject null-id inserts. The variant keeps
#: one declaration correct on every dialect.
BIGSERIAL = BigInteger().with_variant(Integer(), "sqlite")


@compiles(CreateIndex, "sqlite")
def _sqlite_unique_index_name(element, compiler, **kw):
    """SQLite's index namespace is database-global, so short per-table index
    names collide under SQLite. Auto-named indexes already carry their table name;
    this rewrite prefixes only deliberately short names in emitted SQLite DDL."""
    index = element.element
    table = index.table
    short = index.name
    unique_name = (
        short
        if short is None or short.startswith(f"{table.name}_")
        else f"{table.name}_{short}"
    )
    cols = ", ".join(compiler.preparer.quote(c.name) for c in index.columns)
    unique_kw = "UNIQUE " if index.unique else ""
    return (
        f"CREATE {unique_kw}INDEX {compiler.preparer.quote(unique_name)} "
        f"ON {compiler.preparer.quote(table.name)} ({cols})"
    )


def sha256_hash(value: str) -> bytes:
    """32-byte SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).digest()


def as_utc(value: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of
    # the tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes every domain constructs elsewhere.
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def timestamp_indexes() -> tuple:
    """A fresh auto-named timestamp-index pair for a table's ``__table_args__``.
    SQLAlchemy includes the table name in each generated name, keeping indexes
    unique in PostgreSQL's schema-wide namespace."""
    return (
        Index(None, "updated_at"),
        Index(None, "created_at"),
    )


class OnUpdateDateTime(DateTime):
    """DateTime that appends ``ON UPDATE CURRENT_TIMESTAMP`` under MySQL and
    renders as a plain DateTime everywhere else. ``inherit_cache`` is set so
    SQLAlchemy's compilation cache keys on the type correctly."""
    inherit_cache = True


@compiles(OnUpdateDateTime)
def _onupdate_datetime_default(element, compiler, **kw):  # noqa: D401
    # SQLite / other non-MySQL dialects: plain DATETIME -- no ON UPDATE clause.
    return compiler.visit_DATETIME(element, **kw)


@compiles(OnUpdateDateTime, "postgresql")
def _onupdate_datetime_postgresql(element, compiler, **kw):  # noqa: D401
    return compiler.visit_TIMESTAMP(element, **kw)


@compiles(OnUpdateDateTime, "mysql")
def _onupdate_datetime_mysql(element, compiler, **kw):  # noqa: D401
    # MySQL: the ON UPDATE clause is valid in the type-modifier position
    # (``col DATETIME ON UPDATE CURRENT_TIMESTAMP DEFAULT CURRENT_TIMESTAMP``).
    return compiler.visit_DATETIME(element, **kw) + " ON UPDATE CURRENT_TIMESTAMP"


__all__ = [
    "BIGSERIAL",
    "OnUpdateDateTime",
    "TABLE_PREFIX",
    "as_utc",
    "sha256_hash",
    "timestamp_indexes",
]
