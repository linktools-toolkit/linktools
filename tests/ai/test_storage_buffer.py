#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Readable final-value admission, persistence and borrowed-store lifetimes."""

import asyncio
from pathlib import Path
from collections.abc import Hashable, Sequence

import pytest

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend, SqlAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import (
    StorageBatchFailure,
    StorageBatchPartialError,
    StorageBatchResult,
    StorageBuffer,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageOperation,
    StorageOverlay,
    StoragePutResult,
    StorageRevision,
    InMemoryCacheBackend,
)
from linktools.ai.storage import _write_cache as write_cache_module


class _Store:
    def __init__(self) -> None:
        self.values: dict[Hashable, bytes] = {}
        self.batches: list[tuple[StorageChange[Hashable, bytes], ...]] = []
        self.write_started = asyncio.Event()
        self.write_release = asyncio.Event()
        self.write_release.set()
        self.read_started = asyncio.Event()
        self.read_release = asyncio.Event()
        self.read_release.set()
        self.failure: Exception | None = None
        self.fail_after_apply = False
        self.closed = False

    async def get(self, key: Hashable) -> bytes | None:
        value = self.values.get(key)
        self.read_started.set()
        await self.read_release.wait()
        assert not self.closed
        return value

    async def apply_batch(
        self, changes: Sequence[StorageChange[Hashable, bytes]],
    ) -> StorageBatchResult[bytes, Hashable]:
        batch = tuple(changes)
        self.batches.append(batch)
        self.write_started.set()
        await self.write_release.wait()
        assert not self.closed
        if self.failure is not None and not self.fail_after_apply:
            raise self.failure
        revision = StorageRevision(str(len(self.batches)))
        results: list[StoragePutResult[bytes] | StorageDeleteResult[Hashable]] = []
        for change in batch:
            if change.operation is StorageOperation.DELETE:
                deleted = self.values.pop(change.key, None) is not None
                results.append(StorageDeleteResult(change.key, deleted, None, revision))
            else:
                assert change.value is not None
                self.values[change.key] = change.value
                results.append(StoragePutResult(change.value, StorageEntryRevision(1), revision, True))
        if self.failure is not None:
            raise self.failure
        return StorageBatchResult(revision, True, tuple(results))

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("read_pending", [False, True])
async def test_storage_buffer_fixed_read_view_and_final_value_batch(read_pending: bool) -> None:
    store = _Store()
    store.values["key"] = b"durable"
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024), read_pending=read_pending)
    assert await buffer.enqueue("key", b"first") is None
    assert await buffer.enqueue("other", b"other") is None
    await buffer.enqueue_delete("key")
    assert await buffer.get("key") == (None if read_pending else b"durable")
    await buffer.enqueue("key", b"final")
    assert await buffer.get("key") == (b"final" if read_pending else b"durable")
    await buffer.flush()
    assert [(change.key, change.value) for change in store.batches[0]] == [
        ("other", b"other"), ("key", b"final"),
    ]
    assert await buffer.get("key") == b"final"
    await buffer.enqueue_delete("key")
    await buffer.flush()
    assert await buffer.get("key") is None
    await buffer.close()
    assert not store.closed
    assert await store.get("other") == b"other"


@pytest.mark.asyncio
async def test_storage_buffer_accepts_hashable_keys_and_only_immutable_bytes() -> None:
    store = _Store()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024), read_pending=True)
    key = ("resource", 42)
    await buffer.enqueue(key, b"value")
    assert await buffer.get(key) == b"value"
    with pytest.raises(TypeError, match="immutable bytes"):
        await buffer.enqueue(key, bytearray(b"mutable"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        await buffer.enqueue([], b"value")  # type: ignore[arg-type]
    await buffer.flush()
    assert store.values == {key: b"value"}
    await buffer.close()


@pytest.mark.asyncio
async def test_storage_buffer_full_queue_waits_and_cancelled_admission_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(write_cache_module, "_BATCH_SIZE", 1)
    store = _Store()
    store.write_release.clear()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024, max_size=2))
    await buffer.enqueue("first", b"1")
    await asyncio.wait_for(store.write_started.wait(), 1)
    await buffer.enqueue("second", b"2")
    pending = asyncio.create_task(buffer.enqueue("cancelled", b"3"))
    await asyncio.sleep(0)
    assert not pending.done()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    admitted = asyncio.create_task(buffer.enqueue("third", b"3"))
    await asyncio.sleep(0)
    assert not admitted.done()
    store.write_release.set()
    await asyncio.wait_for(admitted, 1)
    await buffer.flush()
    assert store.values == {"first": b"1", "second": b"2", "third": b"3"}
    await buffer.close()


@pytest.mark.asyncio
async def test_storage_buffer_close_wakes_admission_waiter_and_drains_admitted_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(write_cache_module, "_BATCH_SIZE", 1)
    store = _Store()
    store.write_release.clear()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024, max_size=2))
    await buffer.enqueue("first", b"1")
    await asyncio.wait_for(store.write_started.wait(), 1)
    await buffer.enqueue("second", b"2")
    pending = asyncio.create_task(buffer.enqueue("rejected", b"3"))
    close = asyncio.create_task(buffer.close())
    with pytest.raises(AIError, match=ErrorCode.STORAGE_CLOSED.value):
        await asyncio.wait_for(pending, 1)
    assert not close.done()
    store.write_release.set()
    await asyncio.wait_for(close, 1)
    assert store.values == {"first": b"1", "second": b"2"}
    assert not store.closed
    with pytest.raises(AIError, match=ErrorCode.STORAGE_CLOSED.value):
        await buffer.get("first")


@pytest.mark.asyncio
@pytest.mark.parametrize("read_pending", [False, True])
async def test_storage_buffer_pending_view_revalidates_while_committed_view_keeps_source_snapshot(read_pending: bool) -> None:
    store = _Store()
    store.values["key"] = b"old"
    store.read_release.clear()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024), read_pending=read_pending)
    read = asyncio.create_task(buffer.get("key"))
    await asyncio.wait_for(store.read_started.wait(), 1)
    await buffer.enqueue("key", b"new")
    await buffer.flush()
    store.read_release.set()
    assert await asyncio.wait_for(read, 1) == (b"new" if read_pending else b"old")
    await buffer.close()


@pytest.mark.asyncio
async def test_storage_buffer_older_batch_settlement_keeps_newer_pending_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(write_cache_module, "_BATCH_SIZE", 1)
    store = _Store()
    store.write_release.clear()
    buffer = StorageBuffer(store, read_pending=True, cache=InMemoryCacheBackend(max_bytes=1024))
    await buffer.enqueue("key", b"first")
    await asyncio.wait_for(store.write_started.wait(), 1)
    await buffer.enqueue("key", b"latest")
    assert await buffer.get("key") == b"latest"
    store.write_release.set()
    await buffer.flush()
    assert await buffer.get("key") == b"latest"
    assert [batch[0].value for batch in store.batches] == [b"first", b"latest"]
    await buffer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    RuntimeError("write failed"),
    AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN),
    StorageBatchPartialError(StorageBatchFailure(1, "write failed", StorageRevision("partial"), ())),
])
async def test_storage_buffer_failed_or_unknown_batch_fails_closed_and_reads_source(failure: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(write_cache_module, "_BATCH_SIZE", 1)
    store = _Store()
    store.values["key"] = b"old"
    store.write_release.clear()
    store.failure = failure
    store.fail_after_apply = True
    buffer = StorageBuffer(store, read_pending=True, cache=InMemoryCacheBackend(max_bytes=1024, max_size=2))
    await buffer.enqueue("key", b"possibly committed")
    await asyncio.wait_for(store.write_started.wait(), 1)
    await buffer.enqueue("tail", b"never dispatched")
    pending = asyncio.create_task(buffer.enqueue("rejected", b"never admitted"))
    await asyncio.sleep(0)
    store.write_release.set()
    with pytest.raises(type(failure)):
        await asyncio.wait_for(pending, 1)
    with pytest.raises(type(failure)):
        await buffer.flush()
    assert await buffer.get("key") == b"possibly committed"
    assert await buffer.get("tail") is None
    assert len(store.batches) == 1
    with pytest.raises(type(failure)):
        await buffer.enqueue_delete("key")
    with pytest.raises(type(failure)):
        await buffer.close()
    assert not store.closed


@pytest.mark.asyncio
async def test_storage_buffer_cancelled_flush_does_not_cancel_owned_sink() -> None:
    store = _Store()
    store.write_release.clear()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024))
    await buffer.enqueue("key", b"value")
    flush = asyncio.create_task(buffer.flush())
    await asyncio.wait_for(store.write_started.wait(), 1)
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    store.write_release.set()
    await buffer.flush()
    assert store.values == {"key": b"value"}
    await buffer.close()


@pytest.mark.asyncio
async def test_storage_buffer_cancelled_close_keeps_sink_owned_for_later_join() -> None:
    store = _Store()
    store.write_release.clear()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024))
    await buffer.enqueue("key", b"value")
    close = asyncio.create_task(buffer.close())
    await asyncio.wait_for(store.write_started.wait(), 1)
    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(close, 1)
    joined = asyncio.create_task(buffer.close())
    await asyncio.sleep(0)
    assert not joined.done()
    store.write_release.set()
    await asyncio.wait_for(joined, 1)
    assert store.values == {"key": b"value"}
    assert not store.closed


@pytest.mark.asyncio
async def test_storage_buffer_close_drains_cancelled_admitted_read() -> None:
    class UncooperativeStore(_Store):
        async def get(self, key: Hashable) -> bytes | None:
            try:
                return await super().get(key)
            except asyncio.CancelledError:
                await self.read_release.wait()
                return self.values.get(key)

    store = UncooperativeStore()
    store.read_release.clear()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024))
    read = asyncio.create_task(buffer.get("key"))
    await asyncio.wait_for(store.read_started.wait(), 1)
    read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(read, 1)
    close = asyncio.create_task(buffer.close())
    await asyncio.sleep(0)
    assert not close.done()
    store.read_release.set()
    await asyncio.wait_for(close, 1)
    assert not store.closed


@pytest.mark.asyncio
async def test_storage_buffer_timeout_retains_uncooperative_sink_until_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(write_cache_module, "_WRITE_TIMEOUT_SECONDS", 0.01)
    cancelled = asyncio.Event()
    release = asyncio.Event()

    class UncooperativeStore(_Store):
        async def apply_batch(
            self, changes: Sequence[StorageChange[Hashable, bytes]],
        ) -> StorageBatchResult[bytes, Hashable]:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return await super().apply_batch(changes)

    store = UncooperativeStore()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024))
    await buffer.enqueue("key", b"value")
    flush = asyncio.create_task(buffer.flush())
    await asyncio.wait_for(cancelled.wait(), 1)
    assert not flush.done()
    close = asyncio.create_task(buffer.close())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    await asyncio.wait_for(flush, 1)
    await asyncio.wait_for(close, 1)
    assert store.values == {"key": b"value"}
    assert not store.closed


@pytest.mark.asyncio
async def test_storage_buffer_failed_cancelled_source_does_not_acknowledge_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(write_cache_module, "_WRITE_TIMEOUT_SECONDS", 0.01)
    store = _Store()
    store.write_release.clear()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024))
    await buffer.enqueue("key", b"value")
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await asyncio.wait_for(buffer.flush(), 1)
    assert await buffer.get("key") is None
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.close()
    assert not store.closed


@pytest.mark.asyncio
async def test_storage_buffer_reclaims_settled_keys_after_readers_exit() -> None:
    store = _Store()
    buffer = StorageBuffer(store, read_pending=True, cache=InMemoryCacheBackend(max_bytes=1024))
    for number in range(12):
        await buffer.enqueue(number, b"value")
    await buffer.flush()
    for number in range(12):
        assert await buffer.get(number) == b"value"
    # Payload/token retention is an internal memory invariant, not a public index.
    assert buffer._keys == {}
    await buffer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "close"])
async def test_storage_buffer_preserves_caller_cancel_after_owned_failure(operation: str) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class FailingStore(_Store):
        async def get(self, key: Hashable) -> bytes | None:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            raise RuntimeError("source failed")

        async def apply_batch(
            self, changes: Sequence[StorageChange[Hashable, bytes]],
        ) -> StorageBatchResult[bytes, Hashable]:
            entered.set()
            await release.wait()
            raise RuntimeError("source failed")

    buffer = StorageBuffer(FailingStore(), cache=InMemoryCacheBackend(max_bytes=1024))
    if operation == "close":
        await buffer.enqueue("key", b"value")
    task = asyncio.create_task(buffer.get("key") if operation == "read" else buffer.close())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    release.set()
    if operation == "close":
        with pytest.raises(RuntimeError, match="source failed"):
            await buffer.close()
    else:
        await buffer.close()


@pytest.mark.asyncio
async def test_storage_buffer_revalidates_pending_read_after_unknown_commit() -> None:
    store = _Store()
    store.values["key"] = b"old"
    store.read_release.clear()
    store.failure = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    store.fail_after_apply = True
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=1024), read_pending=True)
    read = asyncio.create_task(buffer.get("key"))
    await asyncio.wait_for(store.read_started.wait(), 1)
    await buffer.enqueue("key", b"committed")
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.flush()
    store.read_release.set()
    assert await asyncio.wait_for(read, 1) == b"committed"
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.close()


@pytest.mark.asyncio
async def test_storage_buffer_partial_batch_preserves_source_truth_and_completed_details(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(write_cache_module, "_BATCH_SIZE", 2)
    class PartialStore(_Store):
        async def apply_batch(
            self, changes: Sequence[StorageChange[Hashable, bytes]],
        ) -> StorageBatchResult[bytes, Hashable]:
            result = await super().apply_batch(changes[:1])
            raise StorageBatchPartialError(
                StorageBatchFailure(1, "second change failed", result.store_revision, result.results)
            )

    store = PartialStore()
    store.write_release.clear()
    buffer = StorageBuffer(store, read_pending=True, cache=InMemoryCacheBackend(max_bytes=1024))
    await buffer.enqueue("first", b"committed")
    await buffer.enqueue("second", b"not committed")
    await asyncio.wait_for(store.write_started.wait(), 1)
    await buffer.enqueue("tail", b"never dispatched")
    store.write_release.set()
    with pytest.raises(StorageBatchPartialError) as raised:
        await buffer.flush()
    assert raised.value.failure.failed_index == 1
    assert len(raised.value.failure.completed) == 1
    assert await buffer.get("first") == b"committed"
    assert await buffer.get("second") is None
    assert await buffer.get("tail") is None
    assert len(store.batches) == 1
    with pytest.raises(StorageBatchPartialError):
        await buffer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["asset-memory", "overlay-memory", "asset-sqlite", "custom"])
async def test_storage_buffer_existing_sources_need_no_adapter(kind: str, tmp_path: Path) -> None:
    engine = None
    if kind == "asset-sqlite":
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/assets.db")
        await provision_asset_database(engine)
        backend = SqlAssetBackend(engine, namespace="buffer-consumer")
    else:
        backend = InMemoryAssetBackend()
    overlay = StorageOverlay(backend, writer=backend)
    store = overlay if kind == "overlay-memory" else AssetStore(overlay)
    await store.initialize()

    class CustomSource:
        async def get(self, key: AssetKey) -> bytes | None:
            return await store.get(key)

        async def apply_batch(self, changes: Sequence[StorageChange[AssetKey, bytes]]) -> StorageBatchResult:
            return await store.apply_batch(changes)

    source = CustomSource() if kind == "custom" else store
    key = AssetKey("resource", "test")
    buffer = StorageBuffer(source, cache=InMemoryCacheBackend(max_bytes=1024), read_pending=True)
    try:
        await buffer.enqueue(key, b"first")
        await buffer.enqueue(key, b"last")
        assert await buffer.get(key) == b"last"
        assert await store.get(key) is None
        await buffer.flush()
        assert await store.get(key) == b"last"
        await buffer.enqueue_delete(key)
        assert await buffer.get(key) is None
        assert await store.get(key) == b"last"
        await buffer.flush()
        assert await store.get(key) is None
        await buffer.close()
        await store.put(key, b"borrowed source is still open")
        assert await store.get(key) == b"borrowed source is still open"
    finally:
        await buffer.close()
        await store.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_storage_buffer_without_cache_flush_waits_for_direct_write() -> None:
    store = _Store()
    store.write_release.clear()
    buffer = StorageBuffer(store, read_pending=True)
    write = asyncio.create_task(buffer.enqueue("key", b"value"))
    await store.write_started.wait()
    flush = asyncio.create_task(buffer.flush())
    await asyncio.sleep(0)
    assert not write.done()
    assert not flush.done()
    assert await buffer.get("key") is None
    store.write_release.set()
    await asyncio.gather(write, flush)
    assert await buffer.get("key") == b"value"
    await buffer.close()


@pytest.mark.asyncio
async def test_storage_buffer_cache_failure_invalidates_pending_view() -> None:
    class FailedRead(InMemoryCacheBackend):
        fail_read = True

        async def read_batch(self, scope: str, *, limit: int) -> tuple:
            if self.fail_read:
                self.fail_read = False
                raise OSError("cache read failed")
            return await super().read_batch(scope, limit=limit)

    cache = FailedRead(max_bytes=1024)
    store = _Store()
    store.values["key"] = b"durable"
    buffer = StorageBuffer(store, cache=cache, read_pending=True)
    await buffer.enqueue("key", b"pending")
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.flush()
    assert await buffer.get("key") == b"durable"
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.close()
    assert not store.batches
    # Failed-owner cleanup cannot strand non-evictable capacity.
    assert await cache.append("next-owner", "id", "key", b"x" * 1024, wait=False)
    await cache.ack("next-owner", ("id",))


@pytest.mark.asyncio
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("error_type", [OSError, ValueError])
async def test_storage_buffer_unknown_admission_fails_closed_and_releases_scope(
    published: bool, error_type: type[Exception],
) -> None:
    class LostReply(InMemoryCacheBackend):
        async def append(self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool) -> bool:
            if key == "unknown":
                if published:
                    await super().append(scope, id, key, content, wait=wait)
                raise error_type("unknown response")
            return await super().append(scope, id, key, content, wait=wait)

    cache = LostReply(max_bytes=4, max_size=1)
    store = _Store()
    buffer = StorageBuffer(store, cache=cache, read_pending=True)
    with pytest.raises(error_type, match="unknown response"):
        await buffer.enqueue("unknown", b"lost")
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.enqueue("later", b"real")
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.flush()
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.close()
    assert not store.batches
    assert await cache.append("other-owner", "id", "key", b"full", wait=False)
    await cache.ack("other-owner", ("id",))


@pytest.mark.asyncio
@pytest.mark.parametrize("permanent", [False, True])
async def test_storage_buffer_ack_failure_never_replays_successful_sink(
    permanent: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(write_cache_module, "_WRITE_TIMEOUT_SECONDS", 0 if permanent else 1)

    class FailedAck(InMemoryCacheBackend):
        calls = 0
        fail = True

        async def ack(self, scope: str, ids: Sequence[str]) -> None:
            self.calls += 1
            if self.fail and (permanent or self.calls == 1):
                raise OSError("ack response lost")
            await super().ack(scope, ids)

    cache = FailedAck(max_bytes=1024)
    store = _Store()
    buffer = StorageBuffer(store, cache=cache)
    await buffer.enqueue("key", b"value")
    if permanent:
        with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
            await buffer.flush()
        status = buffer._scope.status()
        assert (status.confirmed, status.pending, status.queue_size) == (1, 0, 1)
        cache.fail = False
        with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
            await buffer.close()
    else:
        await buffer.flush()
        await buffer.close()
    assert store.values == {"key": b"value"}
    assert len(store.batches) == 1
    assert cache.calls >= 2


@pytest.mark.asyncio
async def test_storage_buffer_cancelled_append_registers_actual_admission_before_propagating() -> None:
    class DelayedReply(InMemoryCacheBackend):
        entered = asyncio.Event()

        async def append(self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool) -> bool:
            accepted = await super().append(scope, id, key, content, wait=wait)
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return accepted

    store = _Store()
    store.values["key"] = b"durable"
    cache = DelayedReply(max_bytes=1024)
    buffer = StorageBuffer(store, cache=cache, read_pending=True)
    write = asyncio.create_task(buffer.enqueue("key", b"accepted"))
    await cache.entered.wait()
    assert await buffer.get("key") == b"durable"
    assert not store.batches
    write.cancel()
    with pytest.raises(asyncio.CancelledError):
        await write
    assert await buffer.get("key") == b"accepted"
    await buffer.flush()
    assert store.values == {"key": b"accepted"}
    await buffer.close()


@pytest.mark.asyncio
async def test_memory_cache_hot_content_and_unacknowledged_entries_share_budget() -> None:
    cache = InMemoryCacheBackend(max_bytes=4, max_size=2)
    await cache.put("hot", b"1234")
    assert await cache.append("owner", "first", "key", b"123", wait=False)
    assert await cache.get("hot") is None
    assert await cache.read_batch("owner", limit=1) == (("first", "key", b"123"),)
    # Reading a batch does not release its in-flight capacity.
    assert not await cache.append("other", "too-large", "key", b"12", wait=False)
    await cache.put("cannot-evict-pending", b"12")
    assert await cache.get("cannot-evict-pending") is None
    assert await cache.append("owner", "second", "key", None, wait=False)
    await cache.ack("owner", ("first", "first"))
    assert await cache.read_batch("owner", limit=2) == (("second", "key", None),)
    await cache.put("hot", b"123")
    assert await cache.get("hot") == b"123"
    await cache.ack("owner", ("second",))


@pytest.mark.asyncio
async def test_memory_buffer_exact_entry_retry_and_oversize_wait_are_bounded() -> None:
    cache = InMemoryCacheBackend(max_bytes=2)
    assert await cache.append("owner", "id", ("resource", 1), b"12", wait=False)
    assert await cache.append("owner", "id", ("resource", 1), b"12", wait=False)
    with pytest.raises(ValueError, match="different content or key"):
        await cache.append("owner", "id", ("resource", 2), b"12", wait=False)
    assert not await asyncio.wait_for(
        cache.append("owner", "large", "key", b"123", wait=True), 1,
    )
    await cache.ack("owner", ("id",))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "ack"])
@pytest.mark.parametrize("permanent", [False, True])
async def test_storage_buffer_cancelled_cache_operation_reports_unknown_instead_of_hanging(operation: str, permanent: bool) -> None:
    class CancelledCache(InMemoryCacheBackend):
        failed = False

        async def read_batch(self, scope: str, *, limit: int) -> tuple:
            if operation == "read" and (permanent or not self.failed):
                self.failed = True
                raise asyncio.CancelledError
            return await super().read_batch(scope, limit=limit)

        async def ack(self, scope: str, ids: Sequence[str]) -> None:
            if operation == "ack" and (permanent or not self.failed):
                self.failed = True
                raise asyncio.CancelledError
            await super().ack(scope, ids)

    store = _Store()
    buffer = StorageBuffer(store, cache=CancelledCache(max_bytes=1024), read_pending=True)
    await buffer.enqueue("key", b"value")
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await asyncio.wait_for(buffer.flush(), 1)
    assert await buffer.get("key") == (b"value" if operation == "ack" else None)
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.close()
    assert len(store.batches) == (1 if operation == "ack" else 0)


@pytest.mark.asyncio
async def test_storage_buffer_oversize_rejection_keeps_owner_available() -> None:
    store = _Store()
    buffer = StorageBuffer(store, cache=InMemoryCacheBackend(max_bytes=2))
    with pytest.raises(ValueError, match="rejected"):
        await asyncio.wait_for(buffer.enqueue("oversize", b"123"), 1)
    await buffer.enqueue("fits", b"12")
    await buffer.flush()
    assert store.values == {"fits": b"12"}
    await buffer.close()
