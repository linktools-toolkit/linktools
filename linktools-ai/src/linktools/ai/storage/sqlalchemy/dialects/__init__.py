#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-first dialect package.

The dialect is the seam every SQL store/backend uses to adapt to its concrete
SQL vendor. A caller may inject one explicitly (a test double, or a vendor
with no built-in); otherwise it is left unset and :func:`resolve_dialect`
supplies the matching built-in, keyed off an open session's bound engine's
``dialect.name`` (``session.bind.dialect.name`` -- documented SQLAlchemy API,
no sessionmaker-internals digging) -- the store/backend itself never branches
on the name; that happens once, here, inside the dialects package.

Core ships three concrete implementations -- :class:`SqliteDialect`,
:class:`MySQLDialect`, :class:`PostgreSQLDialect` -- covering the common SQL
vendors' conflict-detecting insert + integrity-error classification. None of
them bundle an actual DBAPI driver: they use SQLAlchemy's own per-dialect
SQL-construction helpers (``sqlalchemy.dialects.mysql``/``postgresql``), so
the caller still supplies its own driver via the engine URL (e.g.
``mysql+asyncmy://`` or ``postgresql+asyncpg://``). A downstream with a
different vendor implements ``SqlAlchemyDialect`` itself and passes it
explicitly."""

from typing import Any

from .base import IntegrityViolationKind, InsertResult, SqlAlchemyDialect
from .mysql import MySQLDialect
from .postgresql import PostgreSQLDialect
from .sqlite import SqliteDialect

_BUILTIN_DIALECTS: "dict[str, SqlAlchemyDialect]" = {
    "sqlite": SqliteDialect(),
    "mysql": MySQLDialect(),
    "postgresql": PostgreSQLDialect(),
}


def resolve_dialect(
    session: Any, dialect: "SqlAlchemyDialect | None" = None
) -> SqlAlchemyDialect:
    """Return ``dialect`` if given; otherwise auto-detect one from an open
    session's bound engine (``session.bind.dialect.name``) and return the
    matching built-in :class:`SqlAlchemyDialect`. This is the one seam every
    store/backend's "explicit override, else auto-resolve" fallback should
    call -- callers that already have a resolved instance may pass it back in
    as ``dialect`` to memoize it (this function is then a no-op).

    Raises ``ValueError`` when ``dialect`` is not given and the session has
    no bound engine (a per-call bind rather than a sessionmaker-level one) or
    the resolved name has no built-in -- in either case the caller passes an
    explicit ``dialect`` instead."""
    if dialect is not None:
        return dialect
    bind = session.bind
    if bind is None:
        raise ValueError(
            "cannot auto-resolve a dialect: session has no bound engine -- "
            "pass dialect= explicitly"
        )
    name = bind.dialect.name
    try:
        return _BUILTIN_DIALECTS[name]
    except KeyError:
        raise ValueError(
            f"no built-in dialect for {name!r} -- pass dialect= explicitly "
            f"with your own SqlAlchemyDialect implementation"
        ) from None


__all__: "list[str]" = (
    "InsertResult",
    "IntegrityViolationKind",
    "MySQLDialect",
    "PostgreSQLDialect",
    "SqlAlchemyDialect",
    "SqliteDialect",
    "resolve_dialect",
)
