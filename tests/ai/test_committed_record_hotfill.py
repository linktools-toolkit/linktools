#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Confirmed singleton fills share the read-through deadline and durable boundary."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import _sql
from linktools.ai.storage import InMemoryCacheBackend, StorageReadCache, _write_cache

from .test_runtime_record_cache import _record, _store

pytestmark = pytest.mark.asyncio


class ObservedCache(InMemoryCacheBackend):
    def __init__(self):
        super().__init__(max_bytes=100000)
        self.published = asyncio.Event()
        self.puts = 0

    async def put(self, key, content):
        await super().put(key, content)
        self.puts += 1
        self.published.set()


def _execution(value: int):
    return replace(_record(value), kind="execution")


async def test_confirmed_singleton_fill_skips_source_but_batch_and_control_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = ObservedCache()
    async with _store("sql", tmp_path, cache) as store:
        record = _execution(1)
        await store.mutate(lambda tx: tx.insert_record(record))
        await asyncio.wait_for(cache.published.wait(), 1)
        reads = 0
        read = store.read

        async def source(fn):
            nonlocal reads
            reads += 1
            return await read(fn)

        monkeypatch.setattr(store, "read", source)
        assert await store.read_record(record.key_digest) == record
        assert reads == 0
        other = _record(1, "head")
        assert await store.read_records((record.key_digest, other.key_digest)) == {record.key_digest: record}
        assert reads == 1
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == record
        assert reads == 2


async def test_unknown_commit_never_publishes_even_if_source_contains_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = ObservedCache()
    async with _store("sql", tmp_path, cache) as store:
        context_type = type(store.storage_group.context)
        run = context_type.run_mutation

        async def unknown(context, *args, **kwargs):
            await run(context, *args, **kwargs)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(context_type, "run_mutation", unknown)
        record = _execution(1)
        with pytest.raises(AIError) as raised:
            await store.mutate(lambda tx: tx.insert_record(record))
        assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        await asyncio.sleep(0)
        assert cache.puts == 0 and store._reads._publisher is None
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == record


@pytest.mark.parametrize("failure", ("error", "cancel"))
async def test_disposable_publish_failure_cannot_change_confirmed_source_result(
    tmp_path: Path, failure: str,
) -> None:
    entered = asyncio.Event()

    class FailingCache(InMemoryCacheBackend):
        async def put(self, key, content):
            entered.set()
            if failure == "cancel":
                raise asyncio.CancelledError
            raise OSError("disposable cache failed")

    async with _store("sql", tmp_path, FailingCache(max_bytes=100000)) as store:
        record = _execution(1)
        assert await store.mutate(lambda tx: tx.insert_record(record)) is None
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == record


async def test_slow_publisher_is_single_flight_and_late_old_fill_is_locally_invalidated(
    tmp_path: Path,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowCache(InMemoryCacheBackend):
        async def put(self, key, content):
            entered.set()
            await release.wait()
            await super().put(key, content)

    async with _store("sql", tmp_path, SlowCache(max_bytes=100000)) as store:
        first = _execution(1)
        await asyncio.wait_for(store.mutate(lambda tx: tx.insert_record(first)), .5)
        await asyncio.wait_for(entered.wait(), 1)
        publisher = store._reads._publisher
        second = _execution(2)
        assert await asyncio.wait_for(store.mutate(
            lambda tx: tx.replace_record(second, expected_storage_revision=0),
        ), .5)
        assert store._reads._publisher is publisher
        release.set()
        await publisher
        assert await store.read_record(first.key_digest) == second


async def test_close_cancels_owned_publication_and_releases_its_payload(tmp_path: Path) -> None:
    entered, stopped = asyncio.Event(), asyncio.Event()

    class SlowCache(InMemoryCacheBackend):
        async def put(self, key, content):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    cache = SlowCache(max_bytes=100000)
    async with _store("sql", tmp_path, cache) as store:
        await store.mutate(lambda tx: tx.insert_record(_execution(1)))
        await asyncio.wait_for(entered.wait(), 1)
        publisher = store._reads._publisher
        await asyncio.wait_for(store.close(), 1)
        assert stopped.is_set() and publisher.done()
        assert store._reads._publisher is None
        assert not cache._items


async def test_transaction_capture_deadline_is_not_renewed_by_commit_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])
    monkeypatch.setattr(_sql, "time", lambda: clock[0])
    cache = ObservedCache()
    async with _store("sql", tmp_path, cache) as store:
        record = _execution(1)
        async def transaction(tx):
            await tx.insert_record(record)
            clock[0] = 1002.0
        await store.mutate(transaction)
        await asyncio.wait_for(cache.published.wait(), 1)
        reads = 0
        read = store.read
        async def source(fn):
            nonlocal reads
            reads += 1
            return await read(fn)
        monkeypatch.setattr(store, "read", source)
        assert await store.read_record(record.key_digest) == record and reads == 0
        clock[0] = 1003.01
        assert await store.read_record(record.key_digest) == record and reads == 1


async def test_cross_owner_late_old_publish_keeps_original_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])
    cache = InMemoryCacheBackend(max_bytes=100000)
    old, new, observer = (StorageReadCache(cache) for _ in range(3))
    clock[0] = 1001.0
    await new.put("same", b"new", captured_at=1001.0)
    async def source():
        return b"new"
    assert await observer.get("same", source) == b"new"
    await old.put("same", b"old", captured_at=1000.0)
    assert await observer.get("same", source) == b"old"
    clock[0] = 1003.01
    assert await observer.get("same", source) == b"new"


async def test_cancelled_close_waiter_cannot_interrupt_publisher_cancellation_cleanup(
    tmp_path: Path,
) -> None:
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancellations = 0

    class SlowCache(InMemoryCacheBackend):
        async def put(self, key, content):
            nonlocal cancellations
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellations += 1
                cleaning.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellations += 1
                    raise

    async with _store("sql", tmp_path, SlowCache(max_bytes=100000)) as store:
        await store.mutate(lambda tx: tx.insert_record(_execution(1)))
        await asyncio.wait_for(entered.wait(), 1)
        close = asyncio.create_task(store.close())
        await asyncio.wait_for(cleaning.wait(), 1)
        publisher = store._reads._publisher
        try:
            close.cancel()
            with pytest.raises(asyncio.CancelledError):
                await close
            assert cancellations == 1 and not publisher.done()
        finally:
            release.set()
            await store.close()
        assert publisher.done() and store._reads._publisher is None


async def test_uncommitted_records_are_never_offered_to_the_cache(tmp_path: Path) -> None:
    cache = ObservedCache()
    entered, release = asyncio.Event(), asyncio.Event()
    async with _store("sql", tmp_path, cache) as store:
        async def pending(tx):
            await tx.insert_record(_execution(1))
            entered.set()
            await release.wait()
        mutation = asyncio.create_task(store.mutate(pending))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert cache.puts == 0 and store._reads._publisher is None
        finally:
            release.set()
            await mutation
        await asyncio.wait_for(cache.published.wait(), 1)
        assert cache.puts == 1


@pytest.mark.parametrize("writes", (False, True))
async def test_delayed_confirmed_mutation_cannot_republish_older_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writes: bool,
) -> None:
    cache = ObservedCache()
    async with _store("sql", tmp_path, cache) as store:
        initial, older, newer = (_execution(n) for n in (1, 2, 3))
        newer = replace(newer, storage_revision=1 + int(writes))
        await store.mutate(lambda tx: tx.insert_record(initial))
        await cache.published.wait()
        store.storage_group._mutation_lock = None
        captured, release = asyncio.Event(), asyncio.Event()
        context_type = type(store.storage_group.context)
        run = context_type.run_mutation

        async def delayed(context, *args, **kwargs):
            result = await run(context, *args, **kwargs)
            if asyncio.current_task().get_name() == "delayed-confirmed":
                captured.set()
                await release.wait()
            return result

        monkeypatch.setattr(context_type, "run_mutation", delayed)
        callback = (
            (lambda tx: tx.replace_record(older, expected_storage_revision=0))
            if writes else (lambda tx: tx.get_record(initial.key_digest))
        )
        early = asyncio.create_task(store.mutate(callback), name="delayed-confirmed")
        await captured.wait()
        await store.mutate(lambda tx: tx.replace_record(newer, expected_storage_revision=int(writes)))
        publisher = store._reads._publisher
        if publisher is not None:
            await publisher
        assert await store.read_record(initial.key_digest) == newer
        puts = cache.puts
        release.set()
        await early
        publisher = store._reads._publisher
        if publisher is not None:
            await publisher
        assert cache.puts == puts
        assert await store.read_record(initial.key_digest) == newer


async def test_confirmed_noop_preserves_cache_and_deletion_invalidates(tmp_path: Path) -> None:
    cache = ObservedCache()
    async with _store("sql", tmp_path, cache) as store:
        record = _execution(1)
        await store.mutate(lambda tx: tx.insert_record(record))
        await cache.published.wait()
        epoch = store._reads._epoch
        await store.mutate(lambda tx: tx.get_record(record.key_digest))
        assert store._reads._epoch is epoch
        assert cache.puts == 1
        await store.mutate(lambda tx: tx.delete_record(record.key_digest))
        assert store._reads._epoch is not epoch
        assert await store.read_record(record.key_digest) is None
