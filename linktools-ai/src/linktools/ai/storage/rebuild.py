#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dev-storage rebuild helpers: wipe + reconstruct the
Filesystem data dir and re-initialize the SQLite dev DB schema from scratch,
so a developer can reset the local environment to a clean baseline.

FilesystemStorage auto-mkdirs on construction (the data dir rebuilds by
deleting the root and re-instantiating). The SQLite rebuild runs
``Base.metadata.create_all`` on a caller-provided engine -- the ENGINE itself
is constructed by the caller (the one-click script or a test), never by this
module, honoring the adapter-boundary invariant that the core does not
parse connection strings or construct engines. Each rebuild is verified by a
RunStore round-trip.

The one-click CLI that constructs the engine + calls these helpers lives in
``linktools-ai/scripts/rebuild_dev_storage.py`` (a dev tool, outside the
installable core)."""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .facade import FilesystemStorage


def _smoke_run_record() -> Any:
    from ..run.models import RunInput, RunnableType, RunRecord, RunStatus

    now = datetime.now(timezone.utc)
    return RunRecord(
        id="rebuild-smoke",
        root_run_id="rebuild-smoke",
        parent_run_id=None,
        session_id="rebuild-smoke-session",
        runnable_id="rebuild-smoke-agent",
        runnable_type=RunnableType.AGENT,
        status=RunStatus.PENDING,
        input=RunInput(prompt="rebuild-smoke"),
        result=None,
        error=None,
        version=1,
        created_at=now,
        started_at=None,
        finished_at=None,
    )


async def _smoke_round_trip(storage: Any) -> None:
    record = _smoke_run_record()
    await storage.runs.create(record)
    fetched = await storage.runs.get("rebuild-smoke")
    assert fetched is not None, "rebuild smoke: RunStore.create then .get failed"
    assert fetched.id == "rebuild-smoke"


def rebuild_filesystem_storage(*, root: Path) -> FilesystemStorage:
    """Wipe ``root`` (if present) and reconstruct a fresh FilesystemStorage.

    The constructor recreates the per-store subdirs, so deleting the root and
    re-instantiating is a complete rebuild."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    return FilesystemStorage(root=root)


async def _init_sql_schema(engine: Any) -> None:
    from .sqlalchemy.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def rebuild_sqlite_storage(*, engine: Any, blobs_root: Path) -> Any:
    """Initialize a fresh SqlAlchemyStorage on a caller-constructed ``engine``:
    run ``Base.metadata.create_all`` and wrap a session_factory.

    The caller owns the engine lifecycle (constructs it, disposes it). This
    module does NOT construct engines -- the core never parses a connection
    string (the SQLAlchemy adapter-boundary invariant). The caller is
    expected to point the engine at a fresh (deleted-then-recreated) db file.
    ``blobs_root`` is the filesystem path the SqlAlchemyStorage uses for its
    FilesystemArtifactBlobStore (artifact content lives outside the DB)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from .sqlalchemy.facade import SqlAlchemyStorage

    asyncio.run(_init_sql_schema(engine))
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyStorage(
        session_factory=session_factory, blobs_root=blobs_root
    )


def rebuild_dev_storage(
    *, data_root: Path, sqlite_engine: "Any | None" = None
) -> dict:
    """One-click verify: wipe + rebuild the Filesystem data dir, smoke-test it,
    then (if a ``sqlite_engine`` is given) initialize the SQLite schema on that
    engine and smoke-test it too. Returns a summary dict. The caller constructs
    and disposes the sqlite_engine; this helper only initializes + verifies.
    The SQLite dev DB shares the same ``data_root`` for its artifact blobs
    (under ``{data_root}/artifacts/blobs``) so a single wipe resets both."""
    fs_storage = rebuild_filesystem_storage(root=data_root)
    asyncio.run(_smoke_round_trip(fs_storage))
    summary: "dict[str, str]" = {
        "filesystem_root": str(data_root),
        "filesystem_smoke": "ok",
    }
    if sqlite_engine is not None:
        sql_storage = rebuild_sqlite_storage(
            engine=sqlite_engine, blobs_root=data_root / "artifacts" / "blobs"
        )
        asyncio.run(_smoke_round_trip(sql_storage))
        summary["sqlite_smoke"] = "ok"
    return summary


__all__: "list[str]" = [
    "rebuild_dev_storage",
    "rebuild_filesystem_storage",
    "rebuild_sqlite_storage",
]
