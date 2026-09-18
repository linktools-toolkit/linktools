#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage lifecycle ownership and cancellation settlement regressions."""

import asyncio
import hashlib
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
import linktools.ai.storage._object_filesystem as object_module
from linktools.ai.asset import (
    AssetCacheAdapter,
    AssetKey,
    AssetStore,
    InMemoryAssetBackend,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.storage import FilesystemObjectStore, SqlObjectStore, StorageOverlay
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.asyncio


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    yield value


async def test_filesystem_object_put_settles_before_propagating_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemObjectStore(tmp_path / "objects")
    payload = b"filesystem-object"
    digest = hashlib.sha256(payload).hexdigest()
    entered = threading.Event()
    release = threading.Event()
    original = object_module._publish_filesystem_object

    def publish(*args: object, **kwargs: object) -> bool:
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test publish release timed out")
        return original(*args, **kwargs)

    monkeypatch.setattr(object_module, "_publish_filesystem_object", publish)
    task = asyncio.create_task(
        store.put(
            "payload",
            _chunks(payload),
            expected_size=len(payload),
            expected_digest=digest,
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert store.pending_background_tasks == ()
    assert await store.stat("payload") is not None


async def test_sql_object_put_settles_before_propagating_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    await provision_database(engine)
    store = SqlObjectStore(engine)
    payload = b"sql-object"
    digest = hashlib.sha256(payload).hexdigest()
    entered = asyncio.Event()
    release = asyncio.Event()
    original = store._insert

    async def insert(key: str, path: Path, size: int, content_digest: str) -> None:
        entered.set()
        await release.wait()
        await original(key, path, size, content_digest)

    monkeypatch.setattr(store, "_insert", insert)
    task = asyncio.create_task(
        store.put(
            "payload",
            _chunks(payload),
            expected_size=len(payload),
            expected_digest=digest,
        )
    )
    await entered.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.pending_background_tasks == ()
        assert await store.stat("payload") is not None
    finally:
        await engine.dispose()


async def test_sql_object_delete_settles_before_propagating_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    await provision_database(engine)
    store = SqlObjectStore(engine)
    payload = b"sql-object"
    digest = hashlib.sha256(payload).hexdigest()
    await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original = store._delete_owned

    async def delete(key: str, *, expected_digest: str) -> bool:
        entered.set()
        await release.wait()
        return await original(key, expected_digest=expected_digest)

    monkeypatch.setattr(store, "_delete_owned", delete)
    task = asyncio.create_task(
        store.delete_object("payload", expected_digest=digest)
    )
    await entered.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.pending_background_tasks == ()
        assert await store.stat("payload") is None
    finally:
        await engine.dispose()


async def test_asset_store_close_owns_backend_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemoryAssetBackend()
    calls = 0
    original = backend.close

    async def close() -> None:
        nonlocal calls
        calls += 1
        await original()

    monkeypatch.setattr(backend, "close", close)
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()

    await store.close()
    await store.close()

    assert calls == 1
    assert not store.ready
    with pytest.raises(AIError) as error:
        await store.initialize()
    assert error.value.code is ErrorCode.STORAGE_CLOSED


async def test_asset_store_close_retries_failed_backend_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemoryAssetBackend()
    calls = 0
    original = backend.close

    async def close() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("retry close")
        await original()

    monkeypatch.setattr(backend, "close", close)
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()

    with pytest.raises(RuntimeError, match="retry close"):
        await store.close()
    assert not store.ready

    await store.close()
    assert calls == 2
    with pytest.raises(AIError) as error:
        await store.initialize()
    assert error.value.code is ErrorCode.STORAGE_CLOSED


async def test_asset_store_failed_initialize_retains_failed_cleanup_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemoryAssetBackend()
    close_calls = 0
    original_close = backend.close

    async def initialize() -> None:
        raise RuntimeError("initialize failed")

    async def close() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("cleanup failed")
        await original_close()

    monkeypatch.setattr(backend, "initialize", initialize)
    monkeypatch.setattr(backend, "close", close)
    store = AssetStore(StorageOverlay(backend, writer=backend))

    with pytest.raises(RuntimeError, match="initialize failed"):
        await store.initialize()
    assert not store.ready
    assert close_calls == 1

    await store.close()
    assert close_calls == 2
    with pytest.raises(AIError) as error:
        await store.initialize()
    assert error.value.code is ErrorCode.STORAGE_CLOSED


class _BlockingCache:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def get(self, key: str) -> bytes | None:
        del key
        self.entered.set()
        await self.release.wait()
        return None

    async def put(self, key: str, content: bytes) -> None:
        del key, content

    async def delete(self, key: str) -> None:
        del key

    async def contains_many(self, keys: Sequence[str]) -> frozenset[str]:
        del keys
        return frozenset()


async def test_asset_store_close_settles_detached_cache_singleflight() -> None:
    backend = InMemoryAssetBackend()
    cache = _BlockingCache()
    storage = StorageOverlay(
        backend,
        writer=backend,
        cache=cache,
        cache_adapter=AssetCacheAdapter(),
    )
    store = AssetStore(storage)
    await store.initialize()
    key = AssetKey("skill", "review/SKILL.md")
    await store.put(key, b"review")

    reader = asyncio.create_task(store.get(key))
    await cache.entered.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    assert storage._cache_tasks

    close_task = asyncio.create_task(store.close())
    await asyncio.sleep(0)
    assert not close_task.done()

    cache.release.set()
    await close_task
    assert storage._cache_tasks == {}
