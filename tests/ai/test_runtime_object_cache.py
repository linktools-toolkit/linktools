#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime content caches preserve source authority and read ownership."""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.persistence._history import TranscriptRepository
from linktools.ai.persistence._object_router import (
    _RuntimeObjectRouter,
    build_runtime_object_router,
    runtime_object_source_reads,
)
from linktools.ai.storage import (
    InMemoryObjectStore,
    ObjectRef,
    ObjectStat,
    StoredPayload,
    StorageReadCache,
    _write_cache,
)


class RecordingCache:
    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.fail: str | None = None
        self.block: str | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def visit(self, operation: str) -> None:
        self.calls.append(operation)
        if self.block == operation:
            self.entered.set()
            await self.release.wait()
        if self.fail == operation:
            raise OSError("cache unavailable")

    async def get(self, key: str) -> bytes | None:
        await self.visit("get")
        return self.items.get(key)

    async def put(self, key: str, content: bytes) -> None:
        await self.visit("put")
        self.items[key] = content

    async def delete(self, key: str) -> None:
        await self.visit("delete")
        self.items.pop(key, None)

    async def contains_many(self, keys: Sequence[str]) -> frozenset[str]:
        return frozenset(key for key in keys if key in self.items)


class RecordingStore(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.opens = 0
        self.stats = 0
        self.block: str | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail: str | None = None
        self.failure = RuntimeError("origin unavailable")

    async def stat(self, key: str) -> ObjectStat | None:
        self.stats += 1
        if self.block == "stat":
            self.entered.set()
            await self.release.wait()
        if self.fail == "stat":
            raise self.failure
        return await super().stat(key)

    async def open(self, key: str) -> AsyncIterator[bytes]:
        self.opens += 1
        if self.fail == "open":
            raise self.failure
        async for content in super().open(key):
            if self.block == "open":
                self.entered.set()
                await self.release.wait()
            yield content


async def put(store: InMemoryObjectStore, payload: bytes = b"content", key: str = "object") -> ObjectRef:
    async def chunks() -> AsyncIterator[bytes]:
        if payload:
            yield payload

    digest = hashlib.sha256(payload).hexdigest()
    await store.put(key, chunks(), expected_digest=digest, expected_size=len(payload))
    return ObjectRef("runtime", key, digest, len(payload))


def router(store: RecordingStore, cache: RecordingCache | None, **scope: str) -> _RuntimeObjectRouter:
    return _RuntimeObjectRouter(
        {domain: store for domain in PersistenceDomain},
        close_guard_stores=(),
        cache=cache,
        cache_domains=tuple(PersistenceDomain),
        namespace=scope.get("namespace", "namespace"),
        tenant_id=scope.get("tenant_id", "tenant"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", (b"", b"content"))
async def test_cache_warm_read_avoids_all_source_reads(payload: bytes) -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store, payload)
    objects = router(store, cache)
    assert await objects.read_object(PersistenceDomain.ARTIFACT, ref) == payload
    assert store.opens == 1 and store.stats == 0
    assert await objects.read_object(PersistenceDomain.ARTIFACT, ref) == payload
    assert store.opens == 1 and store.stats == 0
    await objects.drain_reads()


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", (b"garbage", b"wrong size"))
async def test_corrupt_cached_bytes_are_evicted_and_read_from_origin(corrupt: bytes) -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    objects = router(store, cache)
    await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    cache.items = dict.fromkeys(cache.items, corrupt)
    assert await objects.read_object(PersistenceDomain.ARTIFACT, ref) == b"content"
    assert len(cache.items) == 1
    assert store.opens == 2
    await objects.drain_reads()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("get", "put"))
async def test_cache_errors_do_not_replace_origin_result(operation: str) -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    objects = router(store, cache)
    cache.fail = operation
    assert await objects.read_object(PersistenceDomain.ARTIFACT, ref) == b"content"
    await objects.drain_reads()


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", (False, True))
async def test_origin_errors_are_not_hidden_by_expired_cache(warm: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    objects = router(store, cache)
    now = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: now[0])
    if warm:
        await objects.read_object(PersistenceDomain.ARTIFACT, ref)
        now[0] += 3
    store.fail = "open"
    with pytest.raises(RuntimeError) as raised:
        await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    assert raised.value is store.failure
    await objects.drain_reads()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", (None, b"changed", b"changed size"))
async def test_cache_bounds_deleted_or_changed_source_visibility(replacement: bytes | None, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: now[0])
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    objects = router(store, cache)
    await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    await store.delete_object(ref.key, expected_digest=ref.digest)
    if replacement is not None:
        await put(store, replacement)
    now[0] = 1002.0
    assert await objects.read_object(PersistenceDomain.ARTIFACT, ref) == b"content"
    assert store.opens == 1 and store.stats == 0
    now[0] = 1003.0
    with pytest.raises(AIError) as raised:
        await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    expected = ErrorCode.STORAGE_NOT_FOUND if replacement is None else ErrorCode.STORAGE_INTEGRITY_ERROR
    assert raised.value.code is expected
    await objects.drain_reads()


@pytest.mark.asyncio
async def test_different_physical_sources_use_distinct_namespaces() -> None:
    first, second, cache = RecordingStore(), RecordingStore(), RecordingCache()
    ref = await put(first)
    one, two = router(first, cache, namespace="first"), router(second, cache, namespace="second")
    assert first.store_id == second.store_id
    await one.read_object(PersistenceDomain.ARTIFACT, ref)
    with pytest.raises(AIError) as raised:
        await two.read_object(PersistenceDomain.ARTIFACT, ref)
    assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    await put(second)
    assert await two.read_object(PersistenceDomain.ARTIFACT, ref) == b"content"
    assert second.opens == 2 and second.stats == 0
    await one.drain_reads()
    await two.drain_reads()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("namespace", "tenant_id", "domain", "reference"))
async def test_shared_cache_keeps_domain_tenant_namespace_and_reference_scopes(scope: str) -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    first = router(store, cache)
    await first.read_object(PersistenceDomain.ARTIFACT, ref)
    other = router(store, cache, **({scope: "other"} if scope in {"namespace", "tenant_id"} else {}))
    domain = PersistenceDomain.EXECUTION if scope == "domain" else PersistenceDomain.ARTIFACT
    if scope == "reference":
        ref = await put(store, key="another/object")
    assert await other.read_object(domain, ref) == b"content"
    assert store.opens == 2
    assert len(cache.items) == 2
    await first.drain_reads()
    await other.drain_reads()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_enabled", (False, True))
async def test_invalid_reference_metadata_still_fails_integrity(cache_enabled: bool) -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    objects = router(store, cache if cache_enabled else None)
    await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    invalid = ObjectRef(ref.store_id, ref.key, ref.digest, ref.size + 1)
    with pytest.raises(AIError) as raised:
        await objects.read_object(PersistenceDomain.ARTIFACT, invalid)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    await objects.drain_reads()


@pytest.mark.asyncio
async def test_only_durable_routes_use_cache_backend(tmp_path: Path) -> None:
    store, cache = RecordingStore(), RecordingCache()
    plan = PersistencePlan(
        conversation=StorageRoute.filesystem(tmp_path / "durable", cache=cache),
        artifact=StorageRoute.transient(cache=cache),
        memory=StorageRoute.memory(cache=cache),
    )
    objects = build_runtime_object_router(plan, store, {}, {})
    for domain in (PersistenceDomain.CONVERSATION, PersistenceDomain.ARTIFACT, PersistenceDomain.MEMORY):
        ref = await put(objects.object_store(domain))
        await objects.read_object(domain, ref)
        await objects.read_object(domain, ref)
    assert cache.calls == ["get", "put", "get"]
    await objects.clear_transient()
    transient_ref = ObjectRef("runtime", "object", hashlib.sha256(b"content").hexdigest(), 7)
    with pytest.raises(AIError) as raised:
        await objects.read_object(PersistenceDomain.ARTIFACT, transient_ref)
    assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    await objects.drain_reads()


@pytest.mark.asyncio
async def test_singleflight_survives_cancelled_waiters() -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    store.block = "open"
    objects = router(store, cache)
    first = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    await store.entered.wait()
    others = [asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref)) for _ in range(5)]
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.sleep(0)
    others[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await others[0]
    store.release.set()
    assert await asyncio.gather(*others[1:]) == [b"content"] * 4
    await objects.drain_reads()
    assert store.opens == 1
    assert len(cache.items) == 1
    assert store.stats == 0


@pytest.mark.asyncio
async def test_singleflight_capture_does_not_renew_source_membership_age(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: now[0])
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    store.block = "open"
    objects = router(store, cache)
    first = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    await store.entered.wait()
    await store.delete_object(ref.key, expected_digest=ref.digest)
    now[0] = 1002.0
    borrower = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    await asyncio.sleep(0)
    store.release.set()
    assert await first == await borrower == b"content"
    assert store.opens == 1 and store.stats == 0
    now[0] = 1003.0
    with pytest.raises(AIError) as raised:
        await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    await objects.drain_reads()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("get", "open", "put", "corrupt", "stat"))
async def test_close_drains_all_accepted_read_phases_even_after_caller_cancellation(phase: str) -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    objects = router(store, cache)
    if phase in {"get", "corrupt", "stat"}:
        await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    if phase == "corrupt":
        cache.items = dict.fromkeys(cache.items, b"corrupt")
    barrier = store if phase in {"open", "corrupt", "stat"} else cache
    barrier.block = "open" if phase == "corrupt" else phase
    if phase == "stat":
        with runtime_object_source_reads():
            read = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    else:
        read = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    await barrier.entered.wait()
    read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await read
    objects.begin_close()
    with pytest.raises(AIError) as raised:
        await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    assert raised.value.code is ErrorCode.STORAGE_CLOSED
    close = asyncio.create_task(objects.drain_reads())
    await asyncio.sleep(0)
    assert not close.done()
    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close
    retry = asyncio.create_task(objects.drain_reads())
    await asyncio.sleep(0)
    assert not retry.done()
    barrier.release.set()
    await retry
    assert store.opens == (2 if phase == "corrupt" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_cache", (False, True))
@pytest.mark.parametrize("cancel_during", ("read", "close"))
@pytest.mark.parametrize("close_fails", (False, True))
async def test_uncached_transcript_cancellation_waits_for_owned_stream_close(
    configured_cache: bool, cancel_during: str, close_fails: bool,
) -> None:
    from .test_transcript_payload_integrity import _Store, _Stream

    close_error = ValueError("close failed") if close_fails else None
    stream = _Stream(
        (b"abcd",), block_read=cancel_during == "read", block_close=True,
        close_error=close_error,
    )
    cache = RecordingCache() if configured_cache else None
    objects = _RuntimeObjectRouter(
        {PersistenceDomain.TRACE: _Store(stream)},  # type: ignore[dict-item]
        close_guard_stores=(), cache=cache, cache_domains=(),
    )
    ref = ObjectRef("runtime", "transcript", hashlib.sha256(b"abcd").hexdigest(), 4)
    reader = TranscriptRepository(
        object(),  # type: ignore[arg-type]
        object_store=_Store(stream),  # type: ignore[arg-type]
        object_reader=objects.read_object,
        namespace="history", tenant_id="tenant", runtime_domain=PersistenceDomain.TRACE,
    )
    read = asyncio.create_task(reader.read_payload(StoredPayload.object(ref)))
    close: asyncio.Task[None] | None = None
    try:
        started = stream.read_started if cancel_during == "read" else stream.close_started
        await asyncio.wait_for(started.wait(), timeout=2)
        read.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not read.done()
        await asyncio.wait_for(stream.close_started.wait(), timeout=2)
        close = asyncio.create_task(objects.drain_reads())
        await asyncio.sleep(0)
        assert not close.done()
        read.cancel()
        await asyncio.sleep(0)
        assert not read.done()
        assert not stream.closed
        stream.close_release.set()
        with pytest.raises(ValueError if close_fails else asyncio.CancelledError) as raised:
            await asyncio.wait_for(read, timeout=2)
        if close_fails:
            assert raised.value is close_error
        assert stream.closed
        await close
        if cache is not None:
            assert cache.calls == []
    finally:
        stream.read_release.set()
        stream.close_release.set()
        await asyncio.gather(read, return_exceptions=True)
        await objects.drain_reads()
        if close is not None:
            await close


@pytest.mark.asyncio
async def test_fresh_envelope_still_verifies_content_digest() -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    objects = router(store, cache)
    await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    key = next(iter(cache.items))
    await StorageReadCache(cache).put(key, b"changed")
    assert await objects.read_object(PersistenceDomain.ARTIFACT, ref) == b"content"
    assert store.opens == 2 and store.stats == 0
    await objects.drain_reads()


@pytest.mark.asyncio
async def test_new_borrower_does_not_join_expired_source_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence import _object_router

    now = [1000.0]
    monkeypatch.setattr(_object_router, "monotonic", lambda: now[0])
    monkeypatch.setattr(_write_cache, "time", lambda: now[0])
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    store.block = "open"
    objects = router(store, cache)
    first = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    await store.entered.wait()
    await store.delete_object(ref.key, expected_digest=ref.digest)
    now[0] = 1004.0
    with pytest.raises(AIError) as raised:
        await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    store.release.set()
    assert await first == b"content"
    assert store.opens == 2
    assert not cache.items
    await objects.drain_reads()


@pytest.mark.asyncio
async def test_old_flight_cleanup_keeps_new_flight_after_cancelled_waiter(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence import _object_router

    now = [1000.0]
    monkeypatch.setattr(_object_router, "monotonic", lambda: now[0])
    monkeypatch.setattr(_write_cache, "time", lambda: now[0])
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    class Store(RecordingStore):
        async def open(self, key: str) -> AsyncIterator[bytes]:
            index = self.opens
            self.opens += 1
            async for content in InMemoryObjectStore.open(self, key):
                entered[index].set()
                await release[index].wait()
                yield content

    store = Store()
    ref = await put(store)
    objects = router(store, RecordingCache())
    first = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    await entered[0].wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    now[0] = 1004.0
    second = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    await entered[1].wait()
    release[0].set()
    for _ in range(5):
        await asyncio.sleep(0)
    third = asyncio.create_task(objects.read_object(PersistenceDomain.ARTIFACT, ref))
    await asyncio.sleep(0)
    release[1].set()
    assert await second == await third == b"content"
    assert store.opens == 2
    await objects.drain_reads()


@pytest.mark.asyncio
async def test_source_scope_is_nested_and_task_local_and_resets_after_cancellation() -> None:
    store, cache = RecordingStore(), RecordingCache()
    ref = await put(store)
    objects = router(store, cache)
    await objects.read_object(PersistenceDomain.ARTIFACT, ref)
    await store.delete_object(ref.key, expected_digest=ref.digest)
    entered, release = asyncio.Event(), asyncio.Event()

    async def strong() -> bytes:
        try:
            with runtime_object_source_reads():
                with runtime_object_source_reads():
                    with pytest.raises(AIError) as raised:
                        await objects.read_object(PersistenceDomain.ARTIFACT, ref)
                    assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
                with pytest.raises(AIError):
                    await objects.read_object(PersistenceDomain.ARTIFACT, ref)
                entered.set()
                await release.wait()
        except asyncio.CancelledError:
            return await objects.read_object(PersistenceDomain.ARTIFACT, ref)
        raise AssertionError("test requires cancellation")

    task = asyncio.create_task(strong())
    await entered.wait()
    assert await objects.read_object(PersistenceDomain.ARTIFACT, ref) == b"content"
    task.cancel()
    assert await task == b"content"
    await objects.drain_reads()
