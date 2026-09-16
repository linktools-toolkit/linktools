#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem content cache accounting and cancellation behavior."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from linktools.ai.storage import FilesystemContentCache


def _disk_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())


@pytest.mark.asyncio
async def test_preindex_delete_does_not_corrupt_capacity_accounting(
    tmp_path: Path,
) -> None:
    cache = FilesystemContentCache(tmp_path, max_bytes=10)
    (tmp_path / cache._name("old")).write_bytes(b"o" * 6)
    assert await cache.get("old") == b"o" * 6
    await cache.delete("old")
    await cache.put("a", b"a" * 6)
    await cache.put("b", b"b" * 6)
    assert cache._total == _disk_bytes(tmp_path) <= 10


@pytest.mark.asyncio
async def test_failed_index_scan_does_not_mark_cache_indexed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemContentCache(tmp_path, max_bytes=10)
    original_scan = cache._scan_index

    def fail_scan() -> tuple[tuple[str, int], ...]:
        raise OSError("scan failed")

    monkeypatch.setattr(cache, "_scan_index", fail_scan)
    await cache.put("key", b"value")
    assert cache._indexed is False
    assert not tuple(tmp_path.iterdir())
    monkeypatch.setattr(cache, "_scan_index", original_scan)
    await cache.put("key", b"value")
    assert cache._indexed is True
    assert await cache.get("key") == b"value"


@pytest.mark.asyncio
async def test_cancelled_write_settles_its_capacity_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemContentCache(tmp_path, max_bytes=10)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    original_put = cache._put_sync

    def delayed_put(key: str, content: bytes) -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        original_put(key, content)

    monkeypatch.setattr(cache, "_put_sync", delayed_put)
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        task = asyncio.create_task(cache.put("a", b"a" * 6))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await loop.run_in_executor(executor, lambda: None)
            monkeypatch.setattr(cache, "_put_sync", original_put)
            await cache.put("b", b"b" * 6)
            assert cache._total == _disk_bytes(tmp_path) <= 10
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", (False, True))
async def test_stale_read_cannot_change_newer_cache_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: bool,
) -> None:
    cache = FilesystemContentCache(tmp_path, max_bytes=10)
    await cache.put("seed", b"")
    if not missing:
        await cache.put("a", b"a" * 6)
    target = tmp_path / cache._name("a")
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    original_read = Path.read_bytes

    def delayed_read(path: Path) -> bytes:
        if path != target:
            return original_read(path)
        value = None if missing else original_read(path)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if value is None:
            raise FileNotFoundError(path)
        return value

    monkeypatch.setattr(Path, "read_bytes", delayed_read)
    task = asyncio.create_task(cache.get("a"))
    try:
        await asyncio.wait_for(started.wait(), 2)
        if missing:
            await cache.put("a", b"a" * 6)
        else:
            await cache.delete("a")
            await cache.put("b", b"b" * 6)
        release.set()
        assert await task == (None if missing else b"a" * 6)
        assert cache._total == _disk_bytes(tmp_path) == 6
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_delete_keeps_existing_file_charged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemContentCache(tmp_path, max_bytes=10)
    await cache.put("a", b"a" * 6)
    target = tmp_path / cache._name("a")
    original_unlink = Path.unlink

    def fail_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == target:
            raise PermissionError(path)
        original_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        await cache.delete("a")
    assert cache._total == _disk_bytes(tmp_path) == 6
    await cache.put("b", b"b" * 6)
    assert cache._total == _disk_bytes(tmp_path) <= 10


@pytest.mark.asyncio
async def test_content_reads_preserve_lru_order(tmp_path: Path) -> None:
    cache = FilesystemContentCache(tmp_path, max_bytes=8)
    await cache.put("a", b"aaaa")
    await cache.put("b", b"bbbb")
    assert await cache.get("a") == b"aaaa"
    assert await cache.contains_many(("b",)) == frozenset({"b"})
    await cache.put("c", b"cccc")
    assert await cache.contains_many(("a", "b", "c")) == frozenset({"a", "c"})
