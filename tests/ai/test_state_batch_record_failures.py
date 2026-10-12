#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch preconditions fail without staging a partial record mutation."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._store import RecordReplacement
from tests.ai.test_state_batch_delete_backends import _open_store, _record

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sql"))
async def test_caught_insert_conflict_preserves_batch_atomicity(
    backend: str, tmp_path: Path,
) -> None:
    first, existing = sorted((_record(1), _record(2)), key=lambda r: r.key_digest)
    other = _record(3)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(existing))

        async def callback(transaction):
            with pytest.raises((AIError, IntegrityError)):
                await transaction.insert_records((first, existing))
            await transaction.insert_record(other)

        await store.mutate(callback)
        assert await store.read(lambda tx: tx.get_record(first.key_digest)) is None
        assert await store.read(lambda tx: tx.get_record(existing.key_digest)) == existing
        assert await store.read(lambda tx: tx.get_record(other.key_digest)) == other


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sql"))
async def test_caught_replace_precondition_allows_unrelated_write(
    backend: str, tmp_path: Path,
) -> None:
    records = tuple(sorted((_record(1), _record(2)), key=lambda r: r.key_digest))
    other = _record(3)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_records(records))

        async def callback(transaction):
            with pytest.raises(AIError) as raised:
                await transaction.replace_records(tuple(
                    RecordReplacement(replace(record, storage_version=index + 1), index)
                    for index, record in enumerate(records)
                ))
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert not await transaction.replace_record(
                replace(records[-1], storage_version=2), expected_storage_version=1,
            )
            await transaction.insert_record(other)

        await store.mutate(callback)
        assert await store.read(lambda tx: tx.get_records(tuple(r.key_digest for r in records))) == {
            record.key_digest: record for record in records
        }
        assert await store.read(lambda tx: tx.get_record(other.key_digest)) == other


@pytest.mark.parametrize("bulk,cancel_read", ((False, "list_aliases"), (True, "list_aliases"),
                                             (True, "load_fact_subjects")))
async def test_caught_filesystem_delete_cancel_preserves_all_relationships(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bulk: bool, cancel_read: str,
) -> None:
    import asyncio

    from tests.ai.test_state_batch_delete_backends import _alias, _fact, _seed

    root = tmp_path / "filesystem"
    records = (_record(1), _record(2))
    async with _open_store("filesystem", root) as store:
        await _seed(store, records)
    async with _open_store("filesystem", root) as store:
        entered, release = asyncio.Event(), asyncio.Event()
        to_thread = asyncio.to_thread
        reads = 0

        async def pause(function, *args, **kwargs):
            nonlocal reads
            if function.__name__ == cancel_read:
                reads += 1
                if cancel_read == "list_aliases" or reads == 2:
                    entered.set()
                    await release.wait()
            return await to_thread(function, *args, **kwargs)

        async def callback(transaction):
            try:
                if bulk:
                    await transaction.delete_records(tuple(r.key_digest for r in records))
                else:
                    await transaction.delete_record(records[0].key_digest)
            except asyncio.CancelledError:
                await transaction.insert_record(_record(3))

        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "to_thread", pause)
            task = asyncio.create_task(store.mutate(callback))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel()
                await task
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        assert len(await store.read(lambda tx: tx.scan_records())) == 3
        assert set(await store.read(lambda tx: tx.scan_aliases())) == {_alias(r) for r in records}
        assert {fact.stream_digest: fact for fact in await store.read(lambda tx: tx.scan_facts())} == {
            _fact(r).stream_digest: _fact(r) for r in records
        }
        await store.validate_integrity()
