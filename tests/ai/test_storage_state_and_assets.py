#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State deletion and directory asset behavior."""

import hashlib
from pathlib import Path

import pytest
from linktools.ai.asset import (
    AssetKey,
    AssetRoot,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.core import PersistenceDomain
from linktools.ai.errors import AIError
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.storage import InMemoryCacheBackend, StorageEntryRevision


def _record(value: str) -> StoredRecord:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return StoredRecord(
        digest,
        None,
        None,
        "session",
        (value).encode("ascii"),
        "ACTIVE",
        0,
        None,
        0,
        None,
        {"value": value},
    )


@pytest.mark.asyncio
async def test_sql_deleted_record_does_not_resurface_from_cache(
    tmp_path: Path,
) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**20)
    owner = Persistence(PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache))
    await owner.initialize(namespace="test", tenant_id="tenant")
    store = owner.snapshot_resources().stores[PersistenceDomain.CONVERSATION]
    record = _record("record")
    try:
        await store.mutate(lambda transaction: transaction.insert_record(record))
        assert (
            await store.read(
                lambda transaction: transaction.get_record(record.key)
            )
            == record
        )
        assert await store.mutate(
            lambda transaction: transaction.delete_record(record.key)
        )
        assert (
            await store.read(
                lambda transaction: transaction.get_record(record.key)
            )
            is None
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_directory_assets_are_limited_to_registered_kinds(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "runtime").mkdir()
    (tmp_path / "agents" / "default.json").write_bytes(b"agent")
    (tmp_path / "runtime" / "state.json").write_bytes(b"runtime")
    backend = DirectoryAssetBackend(
        AssetRoot("file", str(tmp_path)),
        path_adapter=PrefixAssetPathAdapter({"agent": "agents"}),
        kinds=("agent",),
    )
    await backend.initialize()

    loaded = await backend.load_metadata(None)
    assert tuple(change.key for change in loaded.changes) == (
        AssetKey("agent", "default.json"),
    )
    assert await backend.get_many((AssetKey("agent", "default.json"),)) == {
        AssetKey("agent", "default.json"): b"agent"
    }
    assert await backend.get_at_revision(AssetKey("agent", "default.json"), StorageEntryRevision(99)) == b"agent"
    with pytest.raises(AIError):
        await backend.get(AssetKey("runtime", "state.json"))
