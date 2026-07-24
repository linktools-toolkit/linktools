#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the one-shot asset layout migrations."""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.storage.migrations import (
    migrate_asset_layout,
    migrate_asset_path_hash_index,
    migrate_asset_path_hash_index_downgrade,
    migrate_sql_asset_tables,
    migrate_sql_asset_tables_downgrade,
    migrate_sqlite_artifact_root,
)


def test_migrate_renames_resources_to_assets(tmp_path):
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "f.txt").write_text("x")
    migrate_asset_layout(tmp_path)
    assert (tmp_path / "assets").is_dir()
    assert (tmp_path / "assets" / "f.txt").read_text() == "x"
    assert not (tmp_path / "resources").exists()


def test_migrate_noop_when_only_assets_exists(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "f.txt").write_text("y")
    migrate_asset_layout(tmp_path)
    assert (tmp_path / "assets" / "f.txt").read_text() == "y"


def test_migrate_creates_assets_when_neither_exists(tmp_path):
    migrate_asset_layout(tmp_path)
    assert (tmp_path / "assets").is_dir()


def test_migrate_fails_closed_when_both_exist(tmp_path):
    (tmp_path / "resources").mkdir()
    (tmp_path / "assets").mkdir()
    with pytest.raises(RuntimeError):
        migrate_asset_layout(tmp_path)
    # Neither directory touched on the fail-closed path.
    assert (tmp_path / "resources").is_dir()
    assert (tmp_path / "assets").is_dir()


def test_migrate_sql_renames_old_asset_tables(tmp_path):
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/m.db")
        async with engine.begin() as conn:
            await conn.exec_driver_sql(
                "CREATE TABLE ai_resources (id INTEGER PRIMARY KEY, path TEXT)"
            )
            await conn.exec_driver_sql(
                "CREATE TABLE ai_resource_idempotency (id INTEGER PRIMARY KEY, k TEXT)"
            )
            await conn.exec_driver_sql(
                "CREATE TABLE ai_resource_revision (id INTEGER PRIMARY KEY, v INTEGER)"
            )
            await conn.exec_driver_sql("INSERT INTO ai_resources (id, path) VALUES (1, 'p')")
        await migrate_sql_asset_tables(engine)
        async with engine.begin() as conn:
            res = await conn.exec_driver_sql("SELECT path FROM ai_assets WHERE id=1")
            assert res.fetchone()[0] == "p"
            for new in ("ai_assets", "ai_asset_idempotency", "ai_asset_revision"):
                row = (
                    await conn.exec_driver_sql(
                        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{new}'"
                    )
                ).fetchone()
                assert row is not None
        await engine.dispose()

    asyncio.run(_run())


def test_migrate_sql_idempotent_when_already_migrated(tmp_path):
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/m2.db")
        async with engine.begin() as conn:
            await conn.exec_driver_sql("CREATE TABLE ai_assets (id INTEGER PRIMARY KEY)")
        # No old tables -> no-op, no error.
        await migrate_sql_asset_tables(engine)
        await engine.dispose()

    asyncio.run(_run())


def test_migrate_sql_downgrade_renames_new_tables_back_to_old(tmp_path):
    """requires both upgrade AND downgrade. The downgrade reverses the
    rename (new -> old); runtime code never calls it."""
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/d.db")
        async with engine.begin() as conn:
            await conn.exec_driver_sql(
                "CREATE TABLE ai_assets (id INTEGER PRIMARY KEY, path TEXT)"
            )
            await conn.exec_driver_sql(
                "CREATE TABLE ai_asset_idempotency (id INTEGER PRIMARY KEY, k TEXT)"
            )
            await conn.exec_driver_sql(
                "CREATE TABLE ai_asset_revision (id INTEGER PRIMARY KEY, v INTEGER)"
            )
            await conn.exec_driver_sql("INSERT INTO ai_assets (id, path) VALUES (1, 'p')")
        await migrate_sql_asset_tables_downgrade(engine)
        async with engine.begin() as conn:
            res = await conn.exec_driver_sql("SELECT path FROM ai_resources WHERE id=1")
            assert res.fetchone()[0] == "p"
            for old in ("ai_resources", "ai_resource_idempotency", "ai_resource_revision"):
                row = (
                    await conn.exec_driver_sql(
                        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{old}'"
                    )
                ).fetchone()
                assert row is not None
            # The new names are gone.
            for gone in ("ai_assets", "ai_asset_idempotency", "ai_asset_revision"):
                row = (
                    await conn.exec_driver_sql(
                        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{gone}'"
                    )
                ).fetchone()
                assert row is None
        await engine.dispose()

    asyncio.run(_run())


def test_migrate_sql_downgrade_idempotent_when_no_new_tables(tmp_path):
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/d2.db")
        async with engine.begin() as conn:
            await conn.exec_driver_sql(
                "CREATE TABLE ai_resources (id INTEGER PRIMARY KEY)"
            )
        # No new tables -> no-op, no error.
        await migrate_sql_asset_tables_downgrade(engine)
        await engine.dispose()

    asyncio.run(_run())


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


async def _seed_pre_hash_schema(conn):
    """The pre-WP4 ai_assets/ai_asset_idempotency schema: full-column unique
    constraints, no path_hash/key_hash columns -- what an existing deployment
    looked like before spec section 7."""
    await conn.exec_driver_sql(
        """
        CREATE TABLE ai_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path VARCHAR(1024) NOT NULL UNIQUE,
            kind VARCHAR(32) NOT NULL,
            etag VARCHAR(64) NOT NULL,
            version INTEGER NOT NULL,
            content_type VARCHAR(255),
            size INTEGER NOT NULL,
            content BLOB NOT NULL,
            modified_at DATETIME NOT NULL,
            metadata_json TEXT NOT NULL,
            deleted_at DATETIME,
            whiteout_version INTEGER
        )
        """
    )
    await conn.exec_driver_sql(
        """
        CREATE TABLE ai_asset_idempotency (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key VARCHAR(512) NOT NULL UNIQUE,
            request_hash VARCHAR(64) NOT NULL,
            result_json TEXT
        )
        """
    )
    await conn.exec_driver_sql(
        "INSERT INTO ai_assets (path, kind, etag, version, content_type, size, "
        "content, modified_at, metadata_json, deleted_at, whiteout_version) "
        "VALUES ('/a.txt', 'file', 'e1', 1, NULL, 1, X'61', "
        "'2024-01-01T00:00:00', '{}', NULL, NULL)"
    )
    await conn.exec_driver_sql(
        "INSERT INTO ai_asset_idempotency (key, request_hash, result_json) "
        "VALUES ('put:k1', 'h1', NULL)"
    )


def test_migrate_asset_path_hash_index_backfills_and_enforces_hash_uniqueness(tmp_path):
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/hash.db")
        async with engine.begin() as conn:
            await _seed_pre_hash_schema(conn)

        await migrate_asset_path_hash_index(engine)

        async with engine.begin() as conn:
            row = (
                await conn.exec_driver_sql(
                    "SELECT path, path_hash FROM ai_assets WHERE path='/a.txt'"
                )
            ).fetchone()
            assert row is not None
            import hashlib

            assert row[1] == hashlib.sha256(b"/a.txt").digest()

            idem_row = (
                await conn.exec_driver_sql(
                    "SELECT key, key_hash FROM ai_asset_idempotency WHERE key='put:k1'"
                )
            ).fetchone()
            assert idem_row is not None
            assert idem_row[1] == hashlib.sha256(b"put:k1").digest()

            # The new unique constraint is on path_hash, not path: inserting a
            # second row with the SAME path_hash (spoofed) must now violate
            # the unique index.
            from sqlalchemy.exc import IntegrityError

            with pytest.raises(IntegrityError):
                await conn.exec_driver_sql(
                    "INSERT INTO ai_assets (path, path_hash, kind, etag, version, "
                    "content_type, size, content, modified_at, metadata_json, "
                    "deleted_at, whiteout_version) VALUES ('/different.txt', "
                    f"X'{row[1].hex()}', 'file', 'e2', 1, NULL, 1, X'62', "
                    "'2024-01-01T00:00:00', '{}', NULL, NULL)"
                )
        await engine.dispose()

    asyncio.run(_run())


def test_migrate_asset_path_hash_index_is_idempotent(tmp_path):
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/hash2.db")
        async with engine.begin() as conn:
            await _seed_pre_hash_schema(conn)
        await migrate_asset_path_hash_index(engine)
        # Re-running against an already-migrated schema must not error.
        await migrate_asset_path_hash_index(engine)
        async with engine.begin() as conn:
            row = (
                await conn.exec_driver_sql("SELECT COUNT(*) FROM ai_assets")
            ).fetchone()
            assert row[0] == 1
        await engine.dispose()

    asyncio.run(_run())


def test_migrate_asset_path_hash_index_downgrade_round_trips(tmp_path):
    async def _run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/hash3.db")
        async with engine.begin() as conn:
            await _seed_pre_hash_schema(conn)
        await migrate_asset_path_hash_index(engine)
        await migrate_asset_path_hash_index_downgrade(engine)
        async with engine.begin() as conn:
            cols = (
                await conn.exec_driver_sql("SELECT * FROM ai_assets LIMIT 0")
            ).keys()
            assert "path_hash" not in cols
            row = (
                await conn.exec_driver_sql(
                    "SELECT path FROM ai_assets WHERE path='/a.txt'"
                )
            ).fetchone()
            assert row is not None
        await engine.dispose()

    asyncio.run(_run())
