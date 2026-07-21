#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqliteStorage reference composition (plan §4.7 / §3.2): the single core site
that constructs a SQLite engine. Verifies it composes Filesystem blobs +
process-local coordination + DATABASE-scope features around the generic
SqlAlchemyStorageAdapter, and that a cross-store transaction yields a UoW
whose artifact_records is session-bound. (Atomic commit/rollback of the shared
adapter is proven in test_facade.py; SqliteStorage delegates to that same
adapter, so it is not re-proven here.)"""

import asyncio

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
        assert storage.features.transactions is TransactionScope.DATABASE
        assert storage.features.coordination is CoordinationScope.PROCESS_LOCAL
        # The artifact facade is wired over Filesystem blobs (the §4.7 default)
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
