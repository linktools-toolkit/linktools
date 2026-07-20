#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 9 op 11: the one-click dev-storage rebuild. Verifies BOTH the
Filesystem data dir and the SQLite dev DB can be wiped and reconstructed from
scratch (a RunStore round-trip succeeds on the fresh stores). The SQLite
engine is constructed by the CALLER (this test / the one-click script), never
by the core rebuild module -- honoring the §6.4 adapter-boundary invariant."""

import asyncio
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.run.models import (
    RunInput,
    RunnableType,
    RunRecord,
    RunStatus,
)
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.rebuild import (
    rebuild_dev_storage,
    rebuild_filesystem_storage,
    rebuild_sqlite_storage,
)


def _make_run(run_id: str, prompt: str = "x") -> RunRecord:
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id="s",
        runnable_id="a",
        runnable_type=RunnableType.AGENT,
        status=RunStatus.PENDING,
        input=RunInput(prompt=prompt),
        result=None,
        error=None,
        version=1,
        created_at=now,
        started_at=None,
        finished_at=None,
    )


def _populate_filesystem(root: Path, run_id: str = "pre-existing") -> None:
    async def _write() -> None:
        storage = FilesystemStorage(root=root)
        await storage.runs.create(_make_run(run_id))

    asyncio.run(_write())


def test_filesystem_rebuild_wipes_and_reconstructs(tmp_path):
    root = tmp_path / "data"
    _populate_filesystem(root, run_id="pre-existing")
    assert (root / "runs" / "pre-existing.json").exists()

    storage = rebuild_filesystem_storage(root=root)

    async def _check():
        assert await storage.runs.get("pre-existing") is None, (
            "rebuild did not wipe the pre-existing Filesystem data"
        )

    asyncio.run(_check())

    # A fresh round-trip works on the reconstructed storage.
    storage2 = FilesystemStorage(root=root)

    async def _rt():
        await storage2.runs.create(_make_run("post-rebuild"))
        fetched = await storage2.runs.get("post-rebuild")
        assert fetched is not None and fetched.id == "post-rebuild"

    asyncio.run(_rt())


def test_sqlite_rebuild_wipes_and_reconstructs(tmp_path):
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from linktools.ai.storage.sqlalchemy.facade import SqlAlchemyStorage

    db_path = tmp_path / "dev.db"

    async def _populate():
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        async with engine.begin() as conn:
            from linktools.ai.storage.sqlalchemy.models import Base

            await conn.run_sync(Base.metadata.create_all)
        storage = SqlAlchemyStorage(
            session_factory=async_sessionmaker(engine, expire_on_commit=False)
        )
        await storage.runs.create(_make_run("pre-existing"))
        await engine.dispose()

    asyncio.run(_populate())
    assert db_path.exists()

    # Rebuild: caller wipes the file + constructs a fresh engine, then the core
    # rebuild helper initializes the schema on it.
    db_path.unlink()
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    storage = rebuild_sqlite_storage(engine=engine)

    async def _check():
        assert await storage.runs.get("pre-existing") is None, (
            "rebuild did not wipe the pre-existing SQLite dev DB"
        )
        await storage.runs.create(_make_run("post-rebuild"))
        fetched = await storage.runs.get("post-rebuild")
        assert fetched is not None and fetched.id == "post-rebuild"

    asyncio.run(_check())
    asyncio.run(engine.dispose())


def test_rebuild_dev_storage_summary_reports_both_backends(tmp_path):
    summary = rebuild_dev_storage(data_root=tmp_path / "data")
    assert summary["filesystem_smoke"] == "ok"
    assert "sqlite_smoke" not in summary  # no engine -> SQLite skipped

    try:
        import sqlalchemy  # noqa: F401
        import aiosqlite  # noqa: F401
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        return  # SQLite extra absent -> only Filesystem rebuilt

    db_path = tmp_path / "dev.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        summary = rebuild_dev_storage(
            data_root=tmp_path / "data2", sqlite_engine=engine
        )
        assert summary["filesystem_smoke"] == "ok"
        assert summary["sqlite_smoke"] == "ok"
    finally:
        asyncio.run(engine.dispose())


def test_one_click_script_runs_wheel_installable_cli(tmp_path):
    # The one-click CLI (linktools-ai/scripts/rebuild_dev_storage.py) is a dev
    # tool that constructs the engine + calls the core helpers. Run it as a
    # subprocess and assert it exits 0 with a filesystem-smoke=ok summary.
    script = (
        Path(__file__).resolve().parents[3]
        / "linktools-ai"
        / "scripts"
        / "rebuild_dev_storage.py"
    )
    if not script.exists():
        pytest.skip(f"one-click script not present at {script}")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(tmp_path / "data"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"one-click script failed: rc={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "filesystem_smoke" in result.stdout
