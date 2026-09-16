#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for filesystem content cache accounting."""

from pathlib import Path

import pytest
from linktools.ai.storage import FilesystemContentCache


@pytest.mark.asyncio
async def test_preindex_delete_does_not_corrupt_capacity_accounting(
    tmp_path: Path,
) -> None:
    cache = FilesystemContentCache(tmp_path, max_bytes=10)
    old_path = tmp_path / cache._name("old")
    old_path.write_bytes(b"o" * 6)

    assert await cache.get("old") == b"o" * 6
    await cache.delete("old")
    await cache.put("a", b"a" * 6)
    await cache.put("b", b"b" * 6)

    files = tuple(path for path in tmp_path.iterdir() if path.is_file())
    actual_total = sum(path.stat().st_size for path in files)
    assert actual_total <= 10
    assert cache._total == actual_total


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
