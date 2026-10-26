#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group-owned logical partitions preserve native keys and atomic settlement."""

import asyncio
import hashlib
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.state import StateGroupTransaction, StateStore, StoredRecord
from linktools.ai.persistence.state import StoredAlias, StateTransactionNestingError


def _key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _record(value: str) -> StoredRecord:
    return StoredRecord(_key("record"), None, None, "probe", "record", None, 0, None, 0, None, {"value": value})


def _storage(root: Path, backend: str) -> PersistenceStorage:
    if backend == "memory":
        return PersistenceStorage.in_memory()
    if backend == "filesystem":
        return PersistenceStorage.filesystem(root / "state")
    return PersistenceStorage.sqlite(root / "state.db")


async def _partition_contract(owner: StateStore) -> None:
    group = owner.storage_group
    assert await group.open_partition(_key("absent")) is None
    first, again = await asyncio.gather(
        group.open_partition(_key("first"), create=True),
        group.open_partition(_key("first"), create=True),
    )
    second = await group.open_partition(_key("second"), create=True)
    assert first is not None and first is again and second is not None and first is not second
    assert first.storage_group is group and second.storage_group is group

    async def write(transaction: StateGroupTransaction) -> None:
        await transaction.transaction(owner).insert_record(_record("owner"))
        for index, store in enumerate((first, second)):
            tx = transaction.transaction(store)
            await tx.insert_record(_record(str(index)))
            await tx.insert_alias(StoredAlias(_key("alias"), _key("record")))
            assert await tx.next_sequence(_key("sequence")) == 1

    async def abort(transaction: StateGroupTransaction) -> None:
        await write(transaction)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    with pytest.raises(AIError) as failure:
        await group.mutate((owner, first, second), abort)
    assert failure.value.code is ErrorCode.STORAGE_CONFLICT
    for store in (owner, first, second):
        assert await store.read(lambda tx: tx.get_record(_key("record"))) is None
    await group.mutate((owner, first, second), write)
    assert (await owner.read(lambda tx: tx.get_record(_key("record")))).data == {"value": "owner"}
    for index, store in enumerate((first, second)):
        values = await store.read(lambda tx: tx.scan_records_page(after=None, limit=1))
        assert values == (_record(str(index)),)
        assert await store.read(lambda tx: tx.resolve_alias(_key("alias"))) == _key("record")
        assert await store.read(lambda tx: tx.get_sequence(_key("sequence"))) == 1
    assert first in await group.scan_stores() and second in await group.scan_stores()
    with pytest.raises(StateTransactionNestingError):
        await owner.read(lambda _: group.open_partition(_key("nested"), create=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "filesystem", "sqlite"])
async def test_state_group_partition_keys_and_atomic_writes(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("scope", storage=_storage(tmp_path, backend)) as persistence:
        await _partition_contract(persistence.execution.executions.state_store)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
async def test_state_group_partitions_reopen_readonly_without_creating_data(tmp_path: Path, backend: str) -> None:
    storage = _storage(tmp_path, backend)
    async with Persistence.open("scope", storage=storage) as persistence:
        store = await persistence.execution.executions.state_store.storage_group.open_partition(_key("first"), create=True)
        assert store is not None
        await store.mutate(lambda tx: tx.insert_record(_record("saved")))
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    async with Persistence.open("scope", storage=storage, read_only=True) as persistence:
        group = persistence.execution.executions.state_store.storage_group
        store = await group.open_partition(_key("first"))
        assert store is not None
        assert await store.read(lambda tx: tx.get_record(_key("record"))) == _record("saved")
        assert await group.open_partition(_key("missing")) is None
        with pytest.raises(AIError) as caught:
            await group.open_partition(_key("missing"), create=True)
        assert caught.value.code is ErrorCode.STORAGE_READ_ONLY
    after = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.asyncio
async def test_state_group_partition_filesystem_publication_readback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("scope", storage=_storage(tmp_path, "filesystem")) as persistence:
        group = persistence.execution.executions.state_store.storage_group
        original = group._provision_group_sync

        def published_then_failed() -> None:
            original()
            raise OSError("publication acknowledgement lost")

        monkeypatch.setattr(group, "_provision_group_sync", published_then_failed)
        store = await group.open_partition(_key("published"), create=True)
        assert store is not None
        assert await group.open_partition(_key("published")) is store
        await store.mutate(lambda tx: tx.insert_record(_record("saved")))
    async with Persistence.open("scope", storage=_storage(tmp_path, "filesystem")) as persistence:
        store = await persistence.execution.executions.state_store.storage_group.open_partition(_key("published"))
        assert store is not None
        assert await store.read(lambda tx: tx.get_record(_key("record"))) == _record("saved")


@pytest.mark.asyncio
async def test_state_group_partition_filesystem_cancelled_open_finishes_owned_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("scope", storage=_storage(tmp_path, "filesystem")) as persistence:
        group = persistence.execution.executions.state_store.storage_group
        original = group._publication_call
        entered, release = asyncio.Event(), asyncio.Event()

        async def paused(callback):
            if getattr(callback, "__name__", "") == "_provision_group_sync":
                entered.set()
                await release.wait()
            return await original(callback)

        monkeypatch.setattr(group, "_publication_call", paused)
        opening = asyncio.create_task(group.open_partition(_key("cancelled"), create=True))
        await entered.wait()
        opening.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert await group.open_partition(_key("cancelled")) is not None


@pytest.mark.asyncio
async def test_state_group_partition_standalone_filesystem_reopens(tmp_path: Path) -> None:
    from .test_state_group_archive import _filesystem

    owner = _filesystem(tmp_path / "standalone", standalone=True)
    await owner.initialize()
    try:
        await _partition_contract(owner)
    finally:
        await owner.close()
    reopened = _filesystem(tmp_path / "standalone", standalone=True, read_only=True)
    await reopened.initialize()
    try:
        store = await reopened.storage_group.open_partition(_key("first"))
        assert store is not None
        assert await store.read(lambda tx: tx.get_record(_key("record"))) == _record("0")
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_state_group_partition_unknown_publication_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence import _filesystem

    async with Persistence.open("scope", storage=_storage(tmp_path, "filesystem")) as persistence:
        group = persistence.execution.executions.state_store.storage_group
        original_publish = group._provision_group_sync
        original_read = _filesystem._read_json
        published = False

        def fail_after_publish() -> None:
            nonlocal published
            original_publish()
            published = True
            raise OSError("acknowledgement lost")

        def fail_readback(path):
            if published and path == group._metadata_root / "manifest.json":
                raise OSError("readback unavailable")
            return original_read(path)

        monkeypatch.setattr(group, "_provision_group_sync", fail_after_publish)
        monkeypatch.setattr(_filesystem, "_read_json", fail_readback)
        with pytest.raises(AIError) as caught:
            await group.open_partition(_key("unknown"), create=True)
        assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        with pytest.raises(AIError):
            await group.open_partition(_key("other"), create=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "filesystem", "sqlite"])
async def test_state_group_partition_closing_one_handle_does_not_close_peers(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("scope", storage=_storage(tmp_path, backend)) as persistence:
        owner = persistence.execution.executions.state_store
        first = await owner.storage_group.open_partition(_key("first"), create=True)
        second = await owner.storage_group.open_partition(_key("second"), create=True)
        assert first is not None and second is not None
        await first.close()
        await second.mutate(lambda tx: tx.insert_record(_record("peer")))
        assert await owner.read(lambda tx: tx.get_record(_key("record"))) is None
        assert await second.read(lambda tx: tx.get_record(_key("record"))) == _record("peer")
