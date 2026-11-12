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
from linktools.ai.persistence.state import RECORD_KEY_MAX_BYTES, RecordScanCursor, StateTransactionNestingError, state_store_key


def _key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _record(value: str) -> StoredRecord:
    return StoredRecord(_key("record"), None, None, "probe", b'record', None, 0, None, 0, None, {"value": value})


def _storage(root: Path, backend: str) -> PersistenceStorage:
    if backend == "memory":
        return PersistenceStorage.in_memory()
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
        assert await store.read(lambda tx: tx.get_sequence(_key("sequence"))) == 1
    assert first in await group.scan_stores() and second in await group.scan_stores()
    with pytest.raises(StateTransactionNestingError):
        await owner.read(lambda _: group.open_partition(_key("nested"), create=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_state_group_partition_keys_and_atomic_writes(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("scope", storage=_storage(tmp_path, backend)) as persistence:
        await _partition_contract(persistence.execution.executions.state_store)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite"])
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
@pytest.mark.parametrize("committed", (False, True))
async def test_state_group_partition_unknown_write_preserves_source_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, committed: bool,
) -> None:
    storage = _storage(tmp_path, "sqlite")
    async with Persistence.open("scope", storage=storage) as persistence:
        group = persistence.execution.executions.state_store.storage_group
        store = await group.open_partition(_key("unknown"), create=True)
        original = type(group.context).run_mutation

        async def unknown(context, callback, **kwargs):
            if context is group.context:
                if committed:
                    await original(context, callback, **kwargs)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return await original(context, callback, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(type(group.context), "run_mutation", unknown)
            with pytest.raises(AIError) as caught:
                await store.mutate(lambda tx: tx.insert_record(_record("saved")))
            assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert await store.read_record(_key("record")) == (_record("saved") if committed else None)
    async with Persistence.open("scope", storage=storage, read_only=True) as persistence:
        store = await persistence.execution.executions.state_store.storage_group.open_partition(_key("unknown"))
        if committed:
            assert store is not None
            assert await store.read_record(_key("record")) == _record("saved")
        else:
            assert store is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_binary_record_boundaries_remain_partition_local(tmp_path: Path, backend: str) -> None:
    scopes = (("namespace", "tenant", "checkpoint"), ("other", "tenant", "checkpoint"),
              ("namespace", "other", "checkpoint"), ("namespace", "tenant", "execution"))
    keys = (b"a", b"z" * RECORD_KEY_MAX_BYTES)
    async with Persistence.open("scope", storage=_storage(tmp_path, backend)) as persistence:
        owner = persistence.execution.executions.state_store
        stores = [await owner.storage_group.open_partition(state_store_key(*scope), create=True) for scope in scopes]
        for index, store in enumerate(stores):
            assert store is not None
            from dataclasses import replace
            rows = tuple(replace(_record(str(index)), key=key) for key in keys)
            await store.mutate(lambda tx: tx.insert_records(rows))
            page = await store.read(lambda tx: tx.scan_records_page(after=RecordScanCursor(rows[0].kind, keys[0]), limit=1))
            assert page == (rows[1],)
            assert await store.read(lambda tx: tx.get_records(keys)) == {row.key: row for row in rows}
            with pytest.raises(ValueError):
                await store.read(lambda tx: tx.get_record(b"x" * (RECORD_KEY_MAX_BYTES + 1)))
            with pytest.raises(ValueError):
                RecordScanCursor(rows[0].kind, b"x" * (RECORD_KEY_MAX_BYTES + 1))
        await stores[0].mutate(lambda tx: tx.delete_records(keys))
        assert await stores[0].read(lambda tx: tx.scan_records()) == ()
        for index, store in enumerate(stores[1:], 1):
            values = await store.read(lambda tx: tx.get_records(keys))
            assert set(values) == set(keys)
            assert all(row.data == {"value": str(index)} for row in values.values())
            await store.validate_integrity()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", (b"", b"x" * (RECORD_KEY_MAX_BYTES + 1), "text-in-binary-column"))
async def test_sql_record_reads_reject_invalid_persisted_addresses(tmp_path: Path, raw: object) -> None:
    async with Persistence.open("scope", storage=_storage(tmp_path, "sqlite")) as persistence:
        store = persistence.execution.executions.state_store
        async def seed(tx):
            await tx.insert_record(_record("owner"))
        await store.mutate(seed)
        async with store.storage_group.context.engine.begin() as connection:
            await connection.exec_driver_sql("UPDATE ai_state_records SET key = ? WHERE store_key = ?", (raw, store.store_key))
        for read in (lambda: store.read(lambda tx: tx.scan_records()), store.validate_integrity):
            with pytest.raises(AIError) as rejected:
                await read()
            assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
