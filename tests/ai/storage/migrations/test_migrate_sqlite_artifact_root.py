#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the one-shot SQLite artifact-blob-root migration."""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.storage.migrations import migrate_sqlite_artifact_root


async def _seed_artifact_records(conn, digests):
    await conn.exec_driver_sql(
        "CREATE TABLE ai_artifact_records (artifact_id TEXT PRIMARY KEY, sha256 TEXT)"
    )
    for i, d in enumerate(digests):
        await conn.exec_driver_sql(
            f"INSERT INTO ai_artifact_records (artifact_id, sha256) VALUES ('a{i}', '{d}')"
        )


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _write_blob(root, digest, data):
    shard = root / digest[:2]
    shard.mkdir(parents=True, exist_ok=True)
    (shard / digest).write_bytes(data)


def test_migrate_sqlite_artifact_root_copies_referenced_digests_with_sha_verify(tmp_path):
    """copy each referenced digest old -> new, verify SHA256, do NOT
    delete old. Unreferenced old blobs are reported, not moved."""
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/ar.db")
        d_ref = _sha("referenced-bytes")
        d_orphan = _sha("orphan-bytes")  # in old blobs, not referenced
        async with engine.begin() as conn:
            await _seed_artifact_records(conn, [d_ref])
        old = tmp_path / "oldblobs"
        new = tmp_path / "new" / "blobs"
        _write_blob(old, d_ref, b"referenced-bytes")
        _write_blob(old, d_orphan, b"orphan-bytes")

        report = await migrate_sqlite_artifact_root(
            engine=engine, old_blobs_root=old, new_blobs_root=new
        )
        assert report.copied == [d_ref]
        assert report.unreferenced == [d_orphan]
        assert report.missing == []
        # Copied into the new root at <xx>/<digest>, verified.
        assert (new / d_ref[:2] / d_ref).read_bytes() == b"referenced-bytes"
        # Old directory is left intact (may be shared with another db).
        assert (old / d_ref[:2] / d_ref).is_file()
        assert (old / d_orphan[:2] / d_orphan).is_file()
        await engine.dispose()

    asyncio.run(_run())


def test_migrate_sqlite_artifact_root_reports_missing_digests(tmp_path):
    """a digest referenced by a record but absent in the old dir is
    reported missing (not silently skipped)."""
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/ar2.db")
        d_present = _sha("present")
        d_absent = _sha("absent-but-referenced")
        async with engine.begin() as conn:
            await _seed_artifact_records(conn, [d_present, d_absent])
        old = tmp_path / "oldblobs"
        new = tmp_path / "new"
        _write_blob(old, d_present, b"present")

        report = await migrate_sqlite_artifact_root(
            engine=engine, old_blobs_root=old, new_blobs_root=new
        )
        assert report.copied == [d_present]
        assert report.missing == [d_absent]
        await engine.dispose()

    asyncio.run(_run())


def test_migrate_sqlite_artifact_root_fails_closed_on_corrupt_blob(tmp_path):
    """+ fail-closed: a blob whose bytes do not hash to its recorded
    digest is corrupt; the tool refuses to copy it (raises) rather than
    propagate bad bytes."""
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/ar3.db")
        d = _sha("correct-bytes")
        async with engine.begin() as conn:
            await _seed_artifact_records(conn, [d])
        old = tmp_path / "oldblobs"
        new = tmp_path / "new"
        _write_blob(old, d, b"WRONG-bytes")  # name says d, content does not hash to d

        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            await migrate_sqlite_artifact_root(
                engine=engine, old_blobs_root=old, new_blobs_root=new
            )
        # Nothing copied on the fail-closed path.
        assert not (new / d[:2] / d).is_file()
        await engine.dispose()

    asyncio.run(_run())


def test_migrate_sqlite_artifact_root_is_idempotent_and_shared_dir_safe(tmp_path):
    """multi-database-shared: two databases sharing one old blobs dir are
    migrated independently -- each copies only ITS referenced digests, and the
    old dir (with both dbs' blobs) survives both migrations for the other db.
    Re-running the same db is a no-op (already-copied blobs are left as-is)."""
    async def _run():
        d_a = _sha("db-a-blob")
        d_b = _sha("db-b-blob")
        old = tmp_path / "sharedblobs"
        _write_blob(old, d_a, b"db-a-blob")
        _write_blob(old, d_b, b"db-b-blob")

        eng_a = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/a.db")
        eng_b = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/b.db")
        async with eng_a.begin() as conn:
            await _seed_artifact_records(conn, [d_a])
        async with eng_b.begin() as conn:
            await _seed_artifact_records(conn, [d_b])

        new_a = tmp_path / "a.artifacts" / "blobs"
        new_b = tmp_path / "b.artifacts" / "blobs"
        rep_a = await migrate_sqlite_artifact_root(
            engine=eng_a, old_blobs_root=old, new_blobs_root=new_a
        )
        rep_b = await migrate_sqlite_artifact_root(
            engine=eng_b, old_blobs_root=old, new_blobs_root=new_b
        )
        # db a got only its blob; db b got only its blob. Each saw the OTHER's
        # blob as unreferenced (correct -- it has no record for it).
        assert rep_a.copied == [d_a] and d_b in rep_a.unreferenced
        assert rep_b.copied == [d_b] and d_a in rep_b.unreferenced
        # The shared old dir still holds both blobs (not deleted).
        assert (old / d_a[:2] / d_a).is_file()
        assert (old / d_b[:2] / d_b).is_file()
        # Re-running db a is a no-op (its blob is already in new_a).
        rep_a2 = await migrate_sqlite_artifact_root(
            engine=eng_a, old_blobs_root=old, new_blobs_root=new_a
        )
        assert rep_a2.copied == []
        await eng_a.dispose()
        await eng_b.dispose()

    asyncio.run(_run())
