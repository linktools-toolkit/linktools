#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""StorageDatabase: the shared SQLAlchemy database every SQL store binds to.

A ``StorageDatabase`` carries the async engine, session factory, the shared
``Base.metadata``, the table prefix, and a ``CoordinationScope``. Construction
does NO I/O: ``create_async_engine`` is lazy, so a SQLite database file (and any
server connection) is opened only when ``initialize_storage`` runs or a session
is first used. SQLite is single-process (``CoordinationScope.PROCESS``); a
server database (MySQL/PostgreSQL) is ``CoordinationScope.SHARED_DATABASE`` and
may be shared across workers.
"""


from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from .sql.base import Base
from .sql.conventions import TABLE_PREFIX

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

class CoordinationScope(StrEnum):
    # Single-process: the local SQLite database is owned by one OS process.
    # Concurrent multi-worker claim against SQLite is not supported.
    PROCESS = "process"
    # A server database (MySQL/PostgreSQL) shared across workers; lease fencing
    # and row-level locking coordinate concurrent claim/commit.
    SHARED_DATABASE = "shared_database"


@dataclass(frozen=True, slots=True)
class StorageDatabase:
    engine: "AsyncEngine"
    session_factory: async_sessionmaker
    coordination_scope: CoordinationScope
    metadata: "type[DeclarativeBase]" = Base
    table_prefix: str = TABLE_PREFIX


def scope_for_url(url: str) -> CoordinationScope:
    """Single-process for SQLite, shared-database for server databases. Pure
    string inspection -- does not import a driver or connect."""
    return CoordinationScope.PROCESS if url.startswith("sqlite") else CoordinationScope.SHARED_DATABASE


def build_sqlite_storage(path: "str | Path") -> StorageDatabase:
    """Build a single-process SQLite StorageDatabase. No file is created until
    ``initialize_storage`` runs -- ``create_async_engine`` opens lazily."""
    from sqlalchemy.ext.asyncio import create_async_engine

    url = f"sqlite+aiosqlite:///{Path(path)}"
    engine = create_async_engine(url)
    return StorageDatabase(
        engine=engine,
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        coordination_scope=CoordinationScope.PROCESS,
    )


def build_storage(url: str, *, coordination_scope: "CoordinationScope | None" = None) -> StorageDatabase:
    """Build a StorageDatabase for an arbitrary async URL. The coordination scope
    defaults to PROCESS for SQLite and SHARED_DATABASE for server databases."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url)
    return StorageDatabase(
        engine=engine,
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        coordination_scope=coordination_scope or scope_for_url(url),
    )


__all__ = [
    "CoordinationScope",
    "StorageDatabase",
    "build_sqlite_storage",
    "build_storage",
    "scope_for_url",
]
