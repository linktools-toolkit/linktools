#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqliteStorage: the SQLite reference Storage composition.

Constructs a ``sqlite+aiosqlite`` async engine + ``async_sessionmaker`` (the
single core site allowed to construct an engine), Filesystem artifact blobs, a
ProcessLocalLeaseCoordinator, DATABASE-scope features, and delegates to
:class:`~linktools.ai.storage.sqlalchemy.facade.SqlAlchemyStorageAdapter`. Use
this for a single-process or single-active-worker reference profile; a
multi-worker deployment injects a distributed coordinator + brings its own
session_factory via the adapter directly.

The engine-construction call (``create_async_engine``) lives here, NOT in the
generic sqlalchemy adapter module, so the boundary ('the generic adapter
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
from ..filesystem.artifact_coordination import FilesystemArtifactDigestCoordinator
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
    synchronous=FULL speeds (well below the  500 events/s gate). The
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
    event store meets throughput out of the box."""

    def __init__(
        self,
        *,
        database: "str | Path",
        artifact_root: "Path | None" = None,
    ) -> None:
        database_str = str(database)
        # An in-memory or URI database has no filesystem path to derive a
        # private artifact root from, and a shared ``parent / "blobs"`` would
        # collide across databases -- so the caller MUST name one explicitly.
        if artifact_root is None and (
            database_str == ":memory:" or database_str.startswith("file:")
        ):
            raise ValueError(
                "SqliteStorage with an in-memory or URI database requires an "
                "explicit artifact_root (the blob directory cannot be derived "
                "and must not be shared across databases)"
            )
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
        configure_wal_pragmas(self._engine)
        session_factory: "async_sessionmaker[AsyncSession]" = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        # Each database owns a private artifact root (``<db>.artifacts``), so two
        # SQLite databases in the same directory never share blob storage and a
        # sweep over one cannot touch the other's blobs.
        resolved_root = (
            Path(artifact_root)
            if artifact_root is not None
            else Path(f"{database_str}.artifacts")
        )
        self._artifact_root = resolved_root
        super().__init__(
            session_factory=session_factory,
            artifact_blobs=FilesystemArtifactBlobStore(blobs_root=resolved_root / "blobs"),
            coordination=ProcessLocalLeaseCoordinator(),
            features=SQLALCHEMY_STORAGE_FEATURES,
            # Blobs live on the shared filesystem, so the per-digest lock must
            # span processes (a separate sweeper worker) -- flock the blobs root.
            artifact_coordinator=FilesystemArtifactDigestCoordinator(
                root=resolved_root / "blobs"
            ),
        )

    async def dispose(self) -> None:
        """Release the engine's connection pool. Call on shutdown. The artifact
        root on disk is left in place -- dispose is a connection-pool release,
        not a data wipe."""
        await self._engine.dispose()


__all__: "list[str]" = ["SqliteStorage", "configure_wal_pragmas"]
