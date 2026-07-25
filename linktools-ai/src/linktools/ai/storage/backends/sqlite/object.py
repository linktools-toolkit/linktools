#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqliteObjectStore: a thin constructor over SqlAlchemyObjectStore.

Per the storage-kernel spec, SQLite must not duplicate the SQLAlchemy
backend's CAS/idempotency/history/transaction logic -- it only builds the
``sqlite+aiosqlite:///<path>`` engine + session_factory a filesystem-path
caller wants, then delegates everything else to SqlAlchemyObjectStore. This
is the one core site allowed to construct an engine from a path (the SQLite
reference-helper exemption, mirroring ``storage/sqlite/facade.py``); the
generic SQLAlchemy adapter itself takes only a caller-built session_factory
and parses no DSN."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ..sqlalchemy.object import SqlAlchemyObjectStore


class SqliteObjectStore(SqlAlchemyObjectStore):
    def __init__(self, *, path: "str | Path") -> None:
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        super().__init__(session_factory=session_factory)

    async def dispose(self) -> None:
        await self._engine.dispose()


__all__: "list[str]" = ["SqliteObjectStore"]
