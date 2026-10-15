#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset batch reads deliver validated values on consumer demand."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from linktools.ai.asset import (
    AssetKey,
    DirectoryAssetBackend,
    FilesystemAssetBackend,
    InMemoryAssetBackend,
    SqlAssetBackend,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy


_Backend = InMemoryAssetBackend | DirectoryAssetBackend | FilesystemAssetBackend | SqlAssetBackend


class _ObservedObjectStore(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.corrupt = False
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def _open(self, key: str) -> AsyncIterator[bytes]:
        self.opened.append(key)
        try:
            if self.block:
                self.entered.set()
                await self.release.wait()
            async for chunk in super()._open(key):
                yield b"!" * len(chunk) if self.corrupt else chunk
        finally:
            self.closed.append(key)


@pytest_asyncio.fixture(params=("memory", "directory", "filesystem", "sqlite-inline", "sqlite-object"))
async def asset_backend(
    request: pytest.FixtureRequest, tmp_path: Path,
) -> AsyncIterator[tuple[_Backend, _ObservedObjectStore]]:
    objects = _ObservedObjectStore()
    engine = None
    if request.param == "memory":
        backend = InMemoryAssetBackend()
    elif request.param == "directory":
        backend = DirectoryAssetBackend(str(tmp_path / "assets"))
    elif request.param == "filesystem":
        backend = FilesystemAssetBackend(tmp_path / "assets", object_store=objects)
    else:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
        await provision_asset_database(engine)
        backend = SqlAssetBackend(
            engine, namespace="batch-stream", object_store=objects,
            payload_policy=PayloadPolicy(0) if request.param == "sqlite-object" else None,
        )
    await backend.initialize()
    try:
        yield backend, objects
    finally:
        await backend.close()
        if engine is not None:
            await engine.dispose()


async def _write(backend: _Backend, key: AssetKey, value: bytes) -> None:
    if isinstance(backend, DirectoryAssetBackend):
        path = backend.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    else:
        await backend.put(key, value)


@pytest.mark.asyncio
async def test_asset_backend_stream_preserves_order_repeats_and_missing(
    asset_backend: tuple[_Backend, _ObservedObjectStore],
) -> None:
    backend, _ = asset_backend
    one, empty, missing = (AssetKey("resource", name) for name in ("one", "empty", "missing"))
    await _write(backend, one, b"one")
    await _write(backend, empty, b"")
    assert [value async for value in backend.get_many(())] == []
    assert [value async for value in backend.get_many((missing, one, empty, one, missing))] == [
        None, b"one", b"", b"one", None,
    ]
    if not isinstance(backend, DirectoryAssetBackend):
        await backend.delete(one)
        await backend.reset(empty)
        assert [value async for value in backend.get_many((one, empty, missing, one))] == [None] * 4


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_backend", ("memory", "filesystem", "sqlite-object"), indirect=True)
async def test_versioned_batch_snapshot_does_not_hold_mutation_lock_while_paused(
    asset_backend: tuple[_Backend, _ObservedObjectStore],
) -> None:
    backend, _ = asset_backend
    one, two = AssetKey("resource", "one"), AssetKey("resource", "two")
    await _write(backend, one, b"one")
    await _write(backend, two, b"old")
    stream = backend.get_many((one, two, one))
    try:
        assert await anext(stream) == b"one"
        await asyncio.wait_for(_write(backend, two, b"new"), timeout=2)
        assert [value async for value in stream] == [b"old", b"one"]
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_backend", ("filesystem", "sqlite-object"), indirect=True)
async def test_object_backed_batch_reads_only_demanded_values_and_closes_each_body(
    asset_backend: tuple[_Backend, _ObservedObjectStore],
) -> None:
    backend, objects = asset_backend
    one, two, unused = (AssetKey("resource", name) for name in ("one", "two", "unused"))
    for key in (one, two, unused):
        await _write(backend, key, key.id.encode())
    stream = backend.get_many((one, two, one, unused))
    try:
        assert objects.opened == []
        assert await anext(stream) == b"one"
        assert len(objects.opened) == 1
        assert objects.closed == objects.opened
        assert await anext(stream) == b"two"
        assert len(objects.opened) == 2
        assert objects.closed == objects.opened
        assert await anext(stream) == b"one"
        assert len(objects.opened) == 2
    finally:
        await stream.aclose()
    assert len(objects.opened) == 2
    assert objects.closed == objects.opened


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_backend", ("filesystem", "sqlite-object"), indirect=True)
async def test_batch_read_exposes_valid_prefix_before_later_integrity_failure(
    asset_backend: tuple[_Backend, _ObservedObjectStore],
) -> None:
    backend, objects = asset_backend
    one, two = AssetKey("resource", "one"), AssetKey("resource", "two")
    await _write(backend, one, b"one")
    await _write(backend, two, b"two")
    stream = backend.get_many((one, two))
    try:
        assert await anext(stream) == b"one"
        objects.corrupt = True
        with pytest.raises(AIError) as error:
            await anext(stream)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert objects.closed == objects.opened
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_backend", ("filesystem", "sqlite-object"), indirect=True)
async def test_batch_read_cancellation_closes_in_progress_body(
    asset_backend: tuple[_Backend, _ObservedObjectStore],
) -> None:
    backend, objects = asset_backend
    one, two = AssetKey("resource", "one"), AssetKey("resource", "two")
    await _write(backend, one, b"one")
    await _write(backend, two, b"two")
    stream = backend.get_many((one, two))
    task = None
    try:
        assert await anext(stream) == b"one"
        objects.block = True
        task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(objects.entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert objects.closed == objects.opened
    finally:
        objects.release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_backend", ("directory",), indirect=True)
async def test_directory_batch_reads_unseen_files_on_demand_and_reuses_duplicates(
    asset_backend: tuple[_Backend, _ObservedObjectStore],
) -> None:
    backend, _ = asset_backend
    one, two, missing = (AssetKey("resource", name) for name in ("one", "two", "missing"))
    await _write(backend, one, b"one")
    await _write(backend, two, b"old")
    stream = backend.get_many((one, two, one, missing, missing))
    try:
        assert await anext(stream) == b"one"
        await _write(backend, one, b"changed")
        await _write(backend, two, b"new")
        assert await anext(stream) == b"new"
        assert await anext(stream) == b"one"
        assert await anext(stream) is None
        await _write(backend, missing, b"created")
        assert await anext(stream) is None
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_backend", ("filesystem",), indirect=True)
async def test_filesystem_batch_loads_metadata_once(
    asset_backend: tuple[_Backend, _ObservedObjectStore], monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _ = asset_backend
    one, two = AssetKey("resource", "one"), AssetKey("resource", "two")
    await _write(backend, one, b"one")
    await _write(backend, two, b"two")
    reload = backend._reload
    reloads = 0

    async def counted_reload() -> None:
        nonlocal reloads
        reloads += 1
        await reload()

    monkeypatch.setattr(backend, "_reload", counted_reload)
    assert [value async for value in backend.get_many((one, two, one))] == [b"one", b"two", b"one"]
    assert reloads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asset_backend", ("sqlite-object",), indirect=True)
async def test_sql_batch_loads_metadata_once_and_releases_connection_before_yield(
    asset_backend: tuple[_Backend, _ObservedObjectStore],
) -> None:
    from sqlalchemy import event

    backend, _ = asset_backend
    one, two = AssetKey("resource", "one"), AssetKey("resource", "two")
    await _write(backend, one, b"one")
    await _write(backend, two, b"two")
    engine = backend._context.engine.sync_engine
    queries: list[str] = []
    connections = 0

    def capture(_connection, _cursor, statement, _parameters, _context, _many) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            queries.append(statement)

    def checkout(*_args: object) -> None:
        nonlocal connections
        connections += 1

    def checkin(*_args: object) -> None:
        nonlocal connections
        connections -= 1

    event.listen(engine, "before_cursor_execute", capture)
    event.listen(engine.pool, "checkout", checkout)
    event.listen(engine.pool, "checkin", checkin)
    stream = backend.get_many((one, two, one))
    try:
        assert await anext(stream) == b"one"
        assert connections == 0
        assert [value async for value in stream] == [b"two", b"one"]
        assert connections == 0
        assert len(queries) == 1
        assert "ai_asset_entries" in queries[0]
    finally:
        await stream.aclose()
        event.remove(engine, "before_cursor_execute", capture)
        event.remove(engine.pool, "checkout", checkout)
        event.remove(engine.pool, "checkin", checkin)
