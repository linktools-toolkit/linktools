#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqliteStorage reference composition: the single core site
that constructs a SQLite engine. Verifies it composes Filesystem blobs +
process-local coordination + DATABASE-scope features around the generic
SqlAlchemyStorageAdapter, and that a cross-store transaction yields a UoW
whose artifact_records is session-bound. (Atomic commit/rollback of the shared
adapter is proven in test_facade.py; SqliteStorage delegates to that same
adapter, so it is not re-proven here.)"""

import asyncio

import pytest

from linktools.ai.storage.features import (
    CoordinationScope,
    StorageFeatures,
    TransactionScope,
)
from linktools.ai.storage.filesystem.artifact import FilesystemArtifactBlobStore


def test_sqlite_storage_builds_engine_sessionmaker_and_database_scope(tmp_path) -> None:
    from linktools.ai.storage.sqlite import SqliteStorage

    storage = SqliteStorage(database=tmp_path / "ref.db")
    try:
        assert isinstance(storage.features, StorageFeatures)
        assert storage.features.transaction_scope is TransactionScope.DATABASE
        assert storage.features.coordination_scope is CoordinationScope.PROCESS_LOCAL
        # The artifact facade is wired over Filesystem blobs (the default)
        # + a SQLAlchemy record store.
        assert isinstance(storage.artifacts._blob, FilesystemArtifactBlobStore)
        assert storage.artifacts._records is not None
    finally:
        asyncio.run(storage.dispose())


def test_sqlite_storage_transaction_yields_session_bound_uow(tmp_path) -> None:
    from linktools.ai.storage.sqlite import SqliteStorage

    storage = SqliteStorage(database=tmp_path / "tx.db")
    try:
        async def _run() -> None:
            # A DATABASE-scope transaction yields a UoW whose artifact_records
            # is a real session-bound store (the cross-store atomic surface).
            async with storage.transaction() as tx:
                assert tx.artifact_records is not None
                assert tx.runs is not None

        asyncio.run(_run())
    finally:
        asyncio.run(storage.dispose())


def test_sqlite_blob_root_is_private_per_database(tmp_path) -> None:
    # Two databases in the SAME directory must resolve to DIFFERENT artifact
    # roots, so a shared ``parent / "blobs"`` can never cross-contaminate them.
    from linktools.ai.storage.sqlite import SqliteStorage

    storage_a = SqliteStorage(database=tmp_path / "a.db")
    storage_b = SqliteStorage(database=tmp_path / "b.db")
    try:
        root_a = storage_a._artifact_root
        root_b = storage_b._artifact_root
        assert root_a != root_b
        # Default root is ``<database>.artifacts`` (NOT ``<parent>/blobs``).
        assert root_a == (tmp_path / "a.db.artifacts")
        assert root_b == (tmp_path / "b.db.artifacts")
        # The blob backends point at each root's private ``blobs`` subdir.
        assert storage_a.artifacts._blob._root == root_a / "blobs"
        assert storage_b.artifacts._blob._root == root_b / "blobs"
    finally:
        asyncio.run(storage_a.dispose())
        asyncio.run(storage_b.dispose())


def test_sqlite_sweep_over_one_db_does_not_touch_another(tmp_path) -> None:
    from linktools.ai.storage.sqlite import SqliteStorage

    storage_a = SqliteStorage(database=tmp_path / "a.db")
    storage_b = SqliteStorage(database=tmp_path / "b.db")
    try:
        # Plant a blob directly in b's blob root. a's blob backend walks ONLY its
        # own ``a.db.artifacts/blobs`` tree, so a sweep over a can never enumerate
        # (let alone delete) b's blobs, even though both live in the same dir.
        b_blobs_root = storage_b.artifacts._blob._root
        b_blobs_root.mkdir(parents=True, exist_ok=True)
        (b_blobs_root / "ab").mkdir(parents=True, exist_ok=True)
        planted = b_blobs_root / "ab" / ("b" * 64)
        planted.write_bytes(b"belongs-to-b")

        async def _enumerate_a() -> "list":
            return [d async for d, _ in storage_a.artifacts._blob.iter_digests_with_mtime()]

        a_digests = asyncio.run(_enumerate_a())
        # a sees none of b's blobs.
        assert "b" * 64 not in a_digests
        assert planted.exists()
    finally:
        asyncio.run(storage_a.dispose())
        asyncio.run(storage_b.dispose())


def test_sqlite_custom_artifact_root_is_honored(tmp_path) -> None:
    from linktools.ai.storage.sqlite import SqliteStorage

    custom = tmp_path / "elsewhere"
    storage = SqliteStorage(database=tmp_path / "c.db", artifact_root=custom)
    try:
        assert storage._artifact_root == custom
        assert storage.artifacts._blob._root == custom / "blobs"
    finally:
        asyncio.run(storage.dispose())


def test_sqlite_memory_without_artifact_root_fails(tmp_path) -> None:
    # An in-memory database has no filesystem path to derive a private root
    # from, so the caller MUST name one explicitly.
    from linktools.ai.storage.sqlite import SqliteStorage

    with pytest.raises(ValueError):
        SqliteStorage(database=":memory:")


def test_sqlite_uri_without_artifact_root_fails(tmp_path) -> None:
    from linktools.ai.storage.sqlite import SqliteStorage

    with pytest.raises(ValueError):
        SqliteStorage(database=f"file:{tmp_path}/u.db")


def test_sqlite_dispose_does_not_delete_artifact_root(tmp_path) -> None:
    from linktools.ai.storage.sqlite import SqliteStorage

    storage = SqliteStorage(database=tmp_path / "d.db")
    root = storage._artifact_root
    asyncio.run(storage.dispose())
    # dispose releases the connection pool only; data on disk is preserved.
    assert root.exists()
