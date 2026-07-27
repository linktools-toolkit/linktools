#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared SQLAlchemy conventions for enterprise-DB conformance.

- ``TABLE_PREFIX``: the unified table-name prefix (every ``__tablename__``
  routes through it).
- ``TimestampMixin``: the built-in ``created_at`` / ``updated_at`` columns with
  a ``CURRENT_TIMESTAMP`` server default (valid on SQLite and MySQL 5.6.5+).
  ``updated_at`` additionally renders ``ON UPDATE CURRENT_TIMESTAMP`` under the
  MySQL dialect only, via the marker type below; SQLite/Postgres render a plain
  ``DATETIME`` so ``create_all`` works there unchanged.
- ``sha256_hash``: 32-byte digest used as the uniqueness carrier for wide
  natural keys (commit_id, scope+key, job_id+key) -- mirrors the storage
  kernel's ``key_hash`` pattern.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, text
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Mapped, mapped_column
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
    """SQLite's index namespace is database-global, so the lint's short per-table
    index names (``ix_key``, ``ix_updated_at`` repeated across every table) collide
    under SQLite. MySQL/Postgres scope index names per-table and keep the short
    names (matching init_schema.sql). This rewrite prefixes the index name with
    its table ONLY in the emitted SQLite DDL; the in-memory Index.name stays the
    short lint-conformant name."""
    index = element.element
    table = index.table
    short = index.name
    unique_name = f"{table.name}_{short}" if short else short
    cols = ", ".join(compiler.preparer.quote(c.name) for c in index.columns)
    unique_kw = "UNIQUE " if index.unique else ""
    return (
        f"CREATE {unique_kw}INDEX {compiler.preparer.quote(unique_name)} "
        f"ON {compiler.preparer.quote(table.name)} ({cols})"
    )


def sha256_hash(value: str) -> bytes:
    """32-byte SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).digest()


def timestamp_indexes() -> tuple:
    """A fresh ``(ix_updated_at, ix_created_at)`` pair for a table's
    ``__table_args__``. Returns new Index instances on each call so they bind
    to the calling table (an Index object cannot be shared across tables)."""
    return (
        Index("ix_updated_at", "updated_at"),
        Index("ix_created_at", "created_at"),
    )


class _OnUpdateDateTime(DateTime):
    """DateTime that appends ``ON UPDATE CURRENT_TIMESTAMP`` under MySQL and
    renders as a plain DateTime everywhere else. ``inherit_cache`` is set so
    SQLAlchemy's compilation cache keys on the type correctly."""
    inherit_cache = True


@compiles(_OnUpdateDateTime)
def _onupdate_datetime_default(element, compiler, **kw):  # noqa: D401
    # SQLite / Postgres / others: plain DATETIME -- no ON UPDATE clause.
    return compiler.visit_DATETIME(element, **kw)


@compiles(_OnUpdateDateTime, "mysql")
def _onupdate_datetime_mysql(element, compiler, **kw):  # noqa: D401
    # MySQL: the ON UPDATE clause is valid in the type-modifier position
    # (``col DATETIME ON UPDATE CURRENT_TIMESTAMP DEFAULT CURRENT_TIMESTAMP``).
    return compiler.visit_DATETIME(element, **kw) + " ON UPDATE CURRENT_TIMESTAMP"


class TimestampMixin:
    """Built-in created_at + updated_at. Both NOT NULL with a
    CURRENT_TIMESTAMP server default; updated_at auto-refreshes on MySQL."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        comment="Created at",
    )
    updated_at: Mapped[datetime] = mapped_column(
        _OnUpdateDateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        comment="Updated at",
    )


__all__: "list[str]" = ["TABLE_PREFIX", "BIGSERIAL", "TimestampMixin", "sha256_hash", "timestamp_indexes"]
