#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqliteStorage: the SQLite reference Storage composition (plan §4.7 / §3.2).

Constructs a ``sqlite+aiosqlite`` async engine + ``async_sessionmaker`` (the
single core site allowed to construct an engine), Filesystem artifact blobs, a
ProcessLocalLeaseCoordinator, DATABASE-scope features, and delegates to
:class:`~linktools.ai.storage.sqlalchemy.facade.SqlAlchemyStorageAdapter`. Use
this for a single-process or single-active-worker reference profile; a
multi-worker deployment injects a distributed coordinator + brings its own
session_factory via the adapter directly.

The engine-construction call (``create_async_engine``) lives here, NOT in the
generic sqlalchemy adapter module, so the §6.5 boundary ('the generic adapter
constructs no engine') is preserved mechanically -- the architecture boundary
test exempts this module from its create_async_engine scan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..coordination.process_local import ProcessLocalLeaseCoordinator
from ..features import SQLALCHEMY_STORAGE_FEATURES
from ..filesystem.artifact import FilesystemArtifactBlobStore
from ..sqlalchemy.facade import SqlAlchemyStorageAdapter


def configure_wal_pragmas(engine: AsyncEngine) -> None:
    """Register a ``connect`` listener that sets write-throughput PRAGMAs on
    every raw SQLite connection the engine opens:

    - ``journal_mode=WAL`` -- write-ahead logging: readers do not block the
      writer and the writer does not fsync the main db on every commit, the
      single biggest lever for a write-heavy event store.
    - ``synchronous=NORMAL`` -- safe under WAL (no torn transactions on power
      loss in WAL mode) and far faster than the default FULL.

    These are the production-grade defaults a write-heavy SQLite deployment
    sets; without them the per-event commit path runs at rollback-journal +
    synchronous=FULL speeds (well below the plan §7.5 500 events/s gate). The
    listener attaches to the engine's SYNC engine (aiosqlite hands the raw
    sqlite3 connection to the connect event)."""

    def _set_pragmas(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    event.listen(engine.sync_engine, "connect", _set_pragmas)


class SqliteStorage(SqlAlchemyStorageAdapter):
    """SQLite reference Storage. Builds the engine + sessionmaker + FS blobs +
    process-local coordination + DATABASE features, then delegates the store
    wiring to the generic :class:`SqlAlchemyStorageAdapter`. The engine is held
    for an explicit ``dispose()`` on shutdown and is tuned with WAL +
    synchronous=NORMAL (via :func:`configure_wal_pragmas`) so the write-heavy
    event store meets plan §7.5 throughput out of the box."""

    def __init__(self, *, database: "str | Path") -> None:
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
        configure_wal_pragmas(self._engine)
        session_factory: "async_sessionmaker[AsyncSession]" = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        blobs_root = Path(database).parent / "blobs"
        super().__init__(
            session_factory=session_factory,
            artifact_blobs=FilesystemArtifactBlobStore(blobs_root=blobs_root),
            coordination=ProcessLocalLeaseCoordinator(),
            features=SQLALCHEMY_STORAGE_FEATURES,
        )

    async def dispose(self) -> None:
        """Release the engine's connection pool. Call on shutdown."""
        await self._engine.dispose()


__all__: "list[str]" = ["SqliteStorage", "configure_wal_pragmas"]
