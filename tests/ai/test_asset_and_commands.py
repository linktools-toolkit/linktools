#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw AssetStore file and public command checks."""

import asyncio
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from linktools.ai.asset import (
    AssetCacheAdapter,
    AssetInfo,
    AssetKey,
    AssetRoot,
    AssetStore,
    FilesystemAssetBackend,
    InMemoryAssetBackend,
    LocalDirectoryAssetBackend,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import (
    InMemoryContentCache,
    StorageComposition,
    StorageLayer,
)


class CountingAssetBackend(InMemoryAssetBackend):
    def __init__(self, root: AssetRoot) -> None:
        super().__init__(root)
        self.reads = 0

    async def get(self, key: AssetKey) -> "bytes | None":
        self.reads += 1
        return await super().get(key)

    async def get_many(self, keys: "Sequence[AssetKey]") -> "dict[AssetKey, bytes]":
        self.reads += 1
        return await super().get_many(keys)


def make_store(
    backend: "InMemoryAssetBackend | FilesystemAssetBackend | LocalDirectoryAssetBackend",
) -> "tuple[AssetStore, StorageComposition[AssetKey, bytes, AssetInfo]]":
    storage = StorageComposition(backend, writer=backend)
    return AssetStore(storage), storage


def test_in_memory_asset_store_cas_tombstone_and_history() -> None:
    async def run() -> None:
        backend = InMemoryAssetBackend(AssetRoot("memory:test", "memory", "test", "digest"))
        store, _ = make_store(backend)
        await store.initialize()
        key = AssetKey("sample", "one")
        first = await store.put(key, b"first")
        second = await store.put(key, b"second", expected_revision=first.revision)
        assert await store.get(key) == b"second"
        assert await store.get_at_version(key, first.revision.value) == b"first"
        assert len(await store.list_versions(key)) == 2
        deleted = await store.delete(key, expected_revision=second.revision)
        assert deleted.deleted is True
        tombstone = await store.stat(key)
        assert tombstone is not None and tombstone.deleted is True
        assert await store.get(key) is None
        assert await store.get_many((key,)) == (None,)
        assert (await store.list_info()).items == ()

    asyncio.run(run())


def test_asset_get_many_and_cursor_preserve_order() -> None:
    async def run() -> None:
        backend = InMemoryAssetBackend(AssetRoot("memory:test", "memory", "test", "digest"))
        store, _ = make_store(backend)
        await store.initialize()
        one = AssetKey("sample", "one")
        two = AssetKey("sample", "two")
        await store.put(one, b"one")
        await store.put(two, b"two")
        assert await store.get_many((two, AssetKey("sample", "missing"), one)) == (b"two", None, b"one")
        first_page = await store.list_info(limit=1)
        assert len(first_page.items) == 1
        assert first_page.next_cursor is not None
        second_page = await store.list_info(cursor=first_page.next_cursor, limit=1)
        assert len(second_page.items) == 1
        assert second_page.next_cursor is None
        with pytest.raises(AIError) as error:
            await store.list_info(cursor="not-a-cursor", limit=1)
        assert error.value.code is ErrorCode.ASSET_CURSOR_INVALID

    asyncio.run(run())


def test_asset_storage_uses_file_cache() -> None:
    async def run() -> None:
        backend = CountingAssetBackend(AssetRoot("memory:cache", "memory", "cache", "digest"))
        cache = InMemoryContentCache(max_bytes=1024 * 1024)
        storage = StorageComposition(
            backend,
            writer=backend,
            cache=cache,
            cache_adapter=AssetCacheAdapter(),
        )
        store = AssetStore(storage)
        await store.initialize()
        key = AssetKey("sample", "cached")
        await store.put(key, b"value")
        backend.reads = 0
        preload = await storage.preload((key,))
        assert preload.loaded == 1
        assert backend.reads == 1
        backend.reads = 0
        assert await store.get(key) == b"value"
        assert await store.get(key) == b"value"
        assert backend.reads == 0

    asyncio.run(run())


def test_read_only_asset_storage_can_cold_start() -> None:
    async def run() -> None:
        source = InMemoryAssetBackend(AssetRoot("memory:seed", "memory", "seed", "digest"))
        source_store, _ = make_store(source)
        await source_store.initialize()
        key = AssetKey("sample", "read-only")
        await source_store.put(key, b"value")
        backend = InMemoryAssetBackend(
            AssetRoot("memory:readonly", "memory", "readonly", "digest"),
            writable=False,
        )
        backend.import_state(source.export_state())
        store = AssetStore(StorageComposition(backend))
        await store.initialize()
        assert await store.get(key) == b"value"
        with pytest.raises(AIError) as error:
            await store.put(key, b"override")
        assert error.value.code is ErrorCode.STORAGE_READ_ONLY

    asyncio.run(run())


def test_filesystem_asset_store_recovers_history_after_restart(tmp_path: Path) -> None:
    async def run() -> None:
        root = AssetRoot("file:test", "file", str(tmp_path), "digest")
        backend = FilesystemAssetBackend(root)
        store, _ = make_store(backend)
        await store.initialize()
        key = AssetKey("sample", "one")
        await store.put(key, b"first")
        await store.put(key, b"second")
        restarted = FilesystemAssetBackend(root)
        restarted_store, _ = make_store(restarted)
        await restarted_store.initialize()
        assert await restarted_store.get(key) == b"second"
        assert len(await restarted_store.list_versions(key)) == 2

    asyncio.run(run())


def test_asset_store_reads_effective_layer_owner() -> None:
    async def run() -> None:
        primary = InMemoryAssetBackend(AssetRoot("memory:primary", "memory", "primary", "primary"))
        fallback = InMemoryAssetBackend(AssetRoot("memory:fallback", "memory", "fallback", "fallback"))
        key = AssetKey("sample", "fallback")
        await fallback.put(key, b"value")
        storage = StorageComposition(primary, layers=(StorageLayer("fallback", fallback),))
        store = AssetStore(storage)
        await store.initialize()
        assert await primary.get(key) is None
        assert await store.get(key) == b"value"

    asyncio.run(run())


def test_asset_store_reset_clears_writer_overlay_and_reveals_layer() -> None:
    async def run() -> None:
        primary = InMemoryAssetBackend(AssetRoot("memory:primary", "memory", "primary", "primary"))
        fallback = InMemoryAssetBackend(AssetRoot("memory:fallback", "memory", "fallback", "fallback"))
        key = AssetKey("sample", "reset")
        await fallback.put(key, b"builtin")
        storage = StorageComposition(
            primary,
            writer=primary,
            layers=(StorageLayer("fallback", fallback),),
        )
        store = AssetStore(storage)
        await store.initialize()
        await store.put(key, b"override")
        assert await store.get(key) == b"override"
        await store.reset()
        assert await store.get(key) == b"builtin"
        location = await storage.locate(key)
        assert location is not None and location.layer == "fallback"
        await store.put(key, b"override-again")
        await store.delete(key)
        assert await store.get(key) is None

    asyncio.run(run())


def test_ai_asset_command_is_removed() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(source_root / "linktools-ai/src"), str(source_root / "linktools/src")))
    result = subprocess.run(
        [sys.executable, "-m", "linktools", "ai", "asset", "--help"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid choice: 'asset'" in result.stderr
