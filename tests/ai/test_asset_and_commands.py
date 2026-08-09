#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AssetStore, file durability, and public command checks."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from linktools.ai.asset import (
    AssetCodecRegistry,
    AssetComposition,
    AssetEntryKey,
    AssetKey,
    AssetRequest,
    AssetRoot,
    AssetStore,
    FilesystemAssetBackend,
    InMemoryAssetBackend,
    LocalDirectoryAssetBackend,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import StorageLayer


@dataclass(frozen=True, slots=True)
class SampleAsset:
    asset_kind: str
    asset_id: str
    value: str


class SampleCodec:
    kind = "sample"
    primary_path = "asset.json"
    value_type = SampleAsset
    fingerprint = "sample-codec-v1"

    def encode(self, value: SampleAsset) -> bytes:
        return json.dumps({"kind": value.asset_kind, "id": value.asset_id, "value": value.value}, sort_keys=True).encode()

    def decode(self, data: bytes) -> SampleAsset:
        value = json.loads(data.decode())
        return SampleAsset(str(value["kind"]), str(value["id"]), str(value["value"]))

    def validate_key(self, key: AssetKey, value: SampleAsset) -> None:
        if (value.asset_kind, value.asset_id) != (key.kind, key.id):
            raise ValueError("asset key mismatch")


class SharedPrimaryCodec(SampleCodec):
    primary_path = "agent.md"

    def __init__(self, kind: str, *, fingerprint: "str | None" = None) -> None:
        self.kind = kind
        self.fingerprint = fingerprint or f"{kind}-codec-v1"


class EmptySource:
    async def list_assets(self, kind: str) -> "tuple[AssetKey, ...]":
        del kind
        return ()

    async def list_files(self, asset: AssetKey) -> "tuple[str, ...]":
        del asset
        return ()

    async def read_file(self, key: AssetEntryKey) -> bytes:
        del key
        raise AssertionError("empty source has no files")

    def identity(self, data: bytes) -> str:
        return str(len(data))


class StaticSource:
    def __init__(self, files: "dict[AssetKey, dict[str, bytes]]") -> None:
        self._files = files

    async def list_assets(self, kind: str) -> "tuple[AssetKey, ...]":
        return tuple(key for key in self._files if key.kind == kind)

    async def list_files(self, asset: AssetKey) -> "tuple[str, ...]":
        return tuple(self._files[asset])

    async def read_file(self, key: AssetEntryKey) -> bytes:
        return self._files[key.asset][key.rel_path]

    def identity(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


def make_store(backend: "InMemoryAssetBackend | FilesystemAssetBackend | LocalDirectoryAssetBackend") -> "tuple[AssetStore, AssetComposition]":
    storage = AssetComposition(backend)
    codecs = AssetCodecRegistry()
    codecs.register(SampleCodec())
    store = AssetStore(storage, codecs=codecs, sources=(EmptySource(),))
    return store, storage


def test_in_memory_asset_store_cas_tombstone_and_history() -> None:
    async def run() -> None:
        backend = InMemoryAssetBackend(AssetRoot("memory:test", "memory", "test", "digest"))
        store, _ = make_store(backend)
        await store.initialize()
        key = AssetKey("sample", "one")
        first = await store.put(key, SampleAsset("sample", "one", "first"))
        second = await store.put(key, SampleAsset("sample", "one", "second"), expected_revision=first.revision)
        assert await store.get(key, expected=SampleAsset) == SampleAsset("sample", "one", "second")
        assert await store.get_at_version(key, first.revision.value, expected=SampleAsset) == SampleAsset("sample", "one", "first")
        assert len(await store.list_versions(key)) == 2
        deleted = await store.delete(key, expected_revision=second.revision)
        assert deleted.deleted is True
        assert await store.get(key, expected=SampleAsset) is None
        assert (await store.list_info()).items == ()

    asyncio.run(run())


def test_asset_get_many_preserves_request_order() -> None:
    async def run() -> None:
        backend = InMemoryAssetBackend(AssetRoot("memory:test", "memory", "test", "digest"))
        store, storage = make_store(backend)
        await store.initialize()
        await store.put(AssetKey("sample", "one"), SampleAsset("sample", "one", "one"))
        await store.put(AssetKey("sample", "two"), SampleAsset("sample", "two", "two"))
        values = await store.get_many(
            (
                AssetRequest(AssetKey("sample", "two"), SampleAsset),
                AssetRequest(AssetKey("sample", "missing"), SampleAsset),
                AssetRequest(AssetKey("sample", "one"), SampleAsset),
            )
        )
        assert values == (
            SampleAsset("sample", "two", "two"),
            None,
            SampleAsset("sample", "one", "one"),
        )
        tree = await storage.get(AssetKey("sample", "one"))
        assert tree is not None
        assert SampleCodec().decode(tree[SampleCodec.primary_path]) == SampleAsset("sample", "one", "one")
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


def test_filesystem_asset_store_recovers_history_after_restart(tmp_path: Path) -> None:
    async def run() -> None:
        root = AssetRoot("file:test", "file", str(tmp_path), "digest")
        backend = FilesystemAssetBackend(root)
        store, _ = make_store(backend)
        await store.initialize()
        key = AssetKey("sample", "one")
        first = await store.put(key, SampleAsset("sample", "one", "first"))
        await store.put(key, SampleAsset("sample", "one", "second"), expected_revision=first.revision)
        restarted = FilesystemAssetBackend(root)
        restarted_store, _ = make_store(restarted)
        await restarted_store.initialize()
        assert await restarted_store.get(key, expected=SampleAsset) == SampleAsset("sample", "one", "second")
        assert len(await restarted_store.list_versions(key)) == 2

    asyncio.run(run())


def test_local_directory_backend_participates_in_storage_composition(tmp_path: Path) -> None:
    async def run() -> None:
        backend = LocalDirectoryAssetBackend(AssetRoot("file:directory", "file", str(tmp_path), "digest"))
        store, storage = make_store(backend)
        await store.initialize()
        key = AssetKey("sample", "local")
        await store.put(key, SampleAsset("sample", "local", "value"))
        tree = await storage.get(key)
        assert tree is not None
        assert SampleCodec().decode(tree[SampleCodec.primary_path]) == SampleAsset("sample", "local", "value")

    asyncio.run(run())


def test_asset_store_reads_file_tree_from_effective_layer_owner() -> None:
    async def run() -> None:
        primary = InMemoryAssetBackend(AssetRoot("memory:primary", "memory", "primary", "primary"))
        fallback = InMemoryAssetBackend(AssetRoot("memory:fallback", "memory", "fallback", "fallback"))
        key = AssetKey("sample", "fallback")
        content = SampleCodec().encode(SampleAsset("sample", "fallback", "value"))
        await fallback.put_file(
            AssetEntryKey(key, SampleCodec.primary_path),
            content,
            primary_path=SampleCodec.primary_path,
            expected_entry_revision=None,
            expected_revision=None,
        )
        storage = AssetComposition(primary, layers=(StorageLayer("fallback", fallback),))
        codecs = AssetCodecRegistry()
        codecs.register(SampleCodec())
        store = AssetStore(storage, codecs=codecs)

        await store.initialize()

        assert await primary.get(key) is None
        assert await store.get(key, expected=SampleAsset) == SampleAsset("sample", "fallback", "value")
        assert [item.key.rel_path for item in await store.list_files(key)] == [SampleCodec.primary_path]
        assert [item.revision.value for item in await store.list_versions(key)] == [1]
        assert await store.get_file_at_version(AssetEntryKey(key, SampleCodec.primary_path), 1) == content

    asyncio.run(run())


def test_codecs_can_share_primary_path_across_asset_kinds() -> None:
    async def run() -> None:
        keys = tuple(AssetKey(kind, "builtin") for kind in ("worker", "stage", "subagent"))
        contents = {
            key: {SharedPrimaryCodec.primary_path: SharedPrimaryCodec(key.kind).encode(SampleAsset(key.kind, key.id, f"{key.kind}-builtin"))}
            for key in keys
        }
        codecs = AssetCodecRegistry()
        for key in keys:
            codecs.register(SharedPrimaryCodec(key.kind))
        with pytest.raises(AIError) as error:
            codecs.register(SharedPrimaryCodec("worker", fingerprint="worker-codec-v2"))
        assert error.value.code is ErrorCode.ASSET_CODEC_CONFLICT

        backend = InMemoryAssetBackend(AssetRoot("memory:shared-primary", "memory", "shared-primary", "digest"))
        store = AssetStore(AssetComposition(backend), codecs=codecs, sources=(StaticSource(contents),))
        await store.initialize()

        for key in keys:
            expected = SampleAsset(key.kind, key.id, f"{key.kind}-builtin")
            assert await store.get(key, expected=SampleAsset) == expected
            info = await store.stat_file(AssetEntryKey(key, SharedPrimaryCodec.primary_path))
            assert info is not None
            assert info.origin == "SOURCE"

        worker = keys[0]
        override = SharedPrimaryCodec(worker.kind).encode(SampleAsset(worker.kind, worker.id, "worker-override"))
        await store.put_file(AssetEntryKey(worker, SharedPrimaryCodec.primary_path), override)
        await store.refresh_sources()
        info = await store.stat_file(AssetEntryKey(worker, SharedPrimaryCodec.primary_path))
        assert info is not None
        assert info.origin == "OVERRIDE"
        assert await store.get(worker, expected=SampleAsset) == SampleAsset(worker.kind, worker.id, "worker-override")

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
