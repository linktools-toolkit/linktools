#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused evidence for lazy filesystem state and Asset boundaries."""

import asyncio
import hashlib
from pathlib import Path

import pytest

from linktools.ai.asset import AssetKey, AssetRoot, DirectoryAssetBackend, PrefixAssetPathAdapter
from linktools.ai.errors import AIError
from linktools.ai.runtime.state import FilesystemStateStorageGroup, FilesystemStateStore, StoredRecord


def _record(value: str) -> StoredRecord:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    partition = hashlib.sha256(b"partition").digest()
    return StoredRecord(
        digest,
        partition,
        None,
        None,
        "session",
        value,
        "ACTIVE",
        0,
        None,
        0,
        None,
        {"value": value},
    )


@pytest.mark.asyncio
async def test_group_initialize_is_lazy_and_member_reads_are_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    group = FilesystemStateStorageGroup(
        root,
        namespace="test",
        tenant_id="tenant",
        scope_digest="scope",
    )
    stores = tuple(
        FilesystemStateStore(
            root / domain,
            namespace="test",
            tenant_id="tenant",
            runtime_domain=domain,
            group=group,
        )
        for domain in ("conversation", "execution")
    )

    def fail_load(_store: FilesystemStateStore) -> None:
        raise AssertionError("initialize loaded the complete filesystem index")

    monkeypatch.setattr(FilesystemStateStore, "_load_index", fail_load)
    try:
        for store in stores:
            await store.initialize()
        entered = 0
        both_entered = asyncio.Event()

        async def read(_transaction: object) -> None:
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=0.5)

        await asyncio.gather(*(store.read(read) for store in stores))
    finally:
        await asyncio.gather(*(store.close() for store in stores))


@pytest.mark.asyncio
async def test_filesystem_deleted_record_does_not_resurface_from_cache(tmp_path: Path) -> None:
    store = FilesystemStateStore(
        tmp_path / "state",
        namespace="test",
        tenant_id="tenant",
        runtime_domain="conversation",
    )
    record = _record("record")
    await store.initialize()
    try:
        await store.mutate(lambda transaction: transaction.insert_record(record))
        assert await store.read(lambda transaction: transaction.get_record(record.key_digest)) == record
        assert await store.mutate(lambda transaction: transaction.delete_record(record.key_digest))
        assert await store.read(lambda transaction: transaction.get_record(record.key_digest)) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_directory_assets_are_limited_to_registered_kinds(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "runtime").mkdir()
    (tmp_path / "agents" / "default.json").write_bytes(b"agent")
    (tmp_path / "runtime" / "state.json").write_bytes(b"runtime")
    backend = DirectoryAssetBackend(
        AssetRoot("file:assets", "file", str(tmp_path), "assets"),
        path_adapter=PrefixAssetPathAdapter({"agent": "agents"}),
        kinds=("agent",),
    )
    await backend.initialize()

    loaded = await backend.load_metadata(None)
    assert tuple(change.key for change in loaded.changes) == (AssetKey("agent", "default.json"),)
    assert await backend.get_many((AssetKey("agent", "default.json"),)) == {
        AssetKey("agent", "default.json"): b"agent"
    }
    with pytest.raises(AIError):
        await backend.get(AssetKey("runtime", "state.json"))
