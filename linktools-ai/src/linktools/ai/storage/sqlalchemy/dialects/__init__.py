#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dialect-strategy package: select a :class:`SqlAlchemyDialectStrategy` by the
bound engine's dialect name at adapter construction, so an unsupported dialect
fails immediately rather than on first write.

The dialect layer is the ONLY place core branches on a dialect name -- the
adapter facade and the asset backend stay dialect-neutral and delegate conflict
classification here."""

from typing import TYPE_CHECKING

from .base import (
    IntegrityViolationKind,
    SqlAlchemyDialectStrategy,
    UnsupportedSqlAlchemyDialectError,
)
from .mysql import MySqlDialectStrategy
from .postgresql import PostgreSqlDialectStrategy
from .sqlite import SqliteDialectStrategy

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker


def resolve_dialect_strategy(
    session_factory: "async_sessionmaker",
) -> SqlAlchemyDialectStrategy:
    """Return the dialect strategy for the engine bound to ``session_factory``.

    Raises :class:`UnsupportedSqlAlchemyDialectError` for any dialect that is
    not SQLite, MySQL, or PostgreSQL. Called eagerly at adapter construction."""
    name = _bound_dialect_name(session_factory)
    if name == "sqlite":
        return SqliteDialectStrategy()
    if name == "mysql":
        return MySqlDialectStrategy()
    if name == "postgresql":
        return PostgreSqlDialectStrategy()
    raise UnsupportedSqlAlchemyDialectError(name)


def _bound_dialect_name(session_factory: "async_sessionmaker") -> str:
    engine = _bound_engine(session_factory)
    return engine.dialect.name


def _bound_engine(session_factory: "async_sessionmaker"):
    # async_sessionmaker stores the bind in its `kw` mapping. Fall back to a
    # constructed session's bind for factories that do not expose `kw`.
    kw = getattr(session_factory, "kw", None)
    bind = kw.get("bind") if isinstance(kw, dict) else None
    if bind is not None:
        return bind
    try:
        return session_factory().bind
    except Exception as exc:  # noqa: BLE001 - surface as a clear config error
        raise UnsupportedSqlAlchemyDialectError(
            None, reason="session_factory does not expose a bound engine"
        ) from exc


__all__: "list[str]" = (
    "IntegrityViolationKind",
    "SqlAlchemyDialectStrategy",
    "UnsupportedSqlAlchemyDialectError",
    "SqliteDialectStrategy",
    "MySqlDialectStrategy",
    "PostgreSqlDialectStrategy",
    "resolve_dialect_strategy",
)
