#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AssetStore, file durability, and public command checks."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from linktools.ai.asset import AssetCodecRegistry, AssetKey, AssetRequest, AssetRoot, AssetStore
from linktools.ai.asset.files import FileAssetBackend, MemoryAssetBackend
from linktools.ai.asset.path import file_root
from linktools.ai.asset.model import AssetInfo, AssetRevision, AssetStoreRevision
from linktools.ai.core.paging import HmacCursorSigner
from linktools.ai.storage.composition import StorageAdapter, StorageComposition
from linktools.ai.storage.layer import StorageWriteVisibility


@dataclass(frozen=True, slots=True)
class SampleAsset:
    asset_kind: str
    asset_id: str
    value: str


class SampleCodec:
    kind = "sample"
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


class IdentityAdapter(StorageAdapter[AssetKey, bytes, AssetKey, bytes, AssetInfo]):
    def to_storage_key(self, key: AssetKey) -> AssetKey:
        return key

    def from_storage_key(self, key: AssetKey) -> AssetKey:
        return key

    def from_storage_value(self, value: bytes) -> bytes:
        return value

    def to_storage_value(self, value: bytes) -> bytes:
        return value

    def validate_value(self, key: AssetKey, value: bytes, info: AssetInfo) -> None:
        if len(value) != info.size:
            raise ValueError("asset size mismatch")


def make_store(backend: MemoryAssetBackend | FileAssetBackend) -> tuple[AssetStore, StorageComposition[AssetKey, bytes, AssetKey, bytes, AssetInfo, AssetRevision, AssetStoreRevision]]:
    codecs = AssetCodecRegistry()
    codecs.register(SampleCodec())
    codecs.freeze()
    storage = StorageComposition(
        backend,
        writer=backend,
        write_visibility=StorageWriteVisibility.READABLE,
        adapter=IdentityAdapter(),
    )
    return AssetStore(storage=storage, codecs=codecs, cursor_signer=HmacCursorSigner("test", b"cursor-key")), storage


def test_memory_asset_store_cas_tombstone_and_history() -> None:
    async def run() -> None:
        backend = MemoryAssetBackend(AssetRoot("memory:test", "memory", "test", "digest"))
        store, storage = make_store(backend)
        await storage.initialize()
        key = AssetKey("sample", "one")
        first = await store.put(key, SampleAsset("sample", "one", "first"))
        second = await store.put(key, SampleAsset("sample", "one", "second"), expected_entry_revision=first.entry_revision)
        assert await store.get(key, expected=SampleAsset) == SampleAsset("sample", "one", "second")
        assert await store.get_at_version(key, first.entry_revision.value, expected=SampleAsset) == SampleAsset("sample", "one", "first")
        assert len(await store.list_versions(key)) == 2
        deleted = await store.delete(key, expected_entry_revision=second.entry_revision)
        assert deleted.deleted is True
        assert await store.get(key, expected=SampleAsset) is None
        assert (await store.list_info()).items == ()

    asyncio.run(run())


def test_asset_get_many_preserves_request_order() -> None:
    async def run() -> None:
        backend = MemoryAssetBackend(AssetRoot("memory:test", "memory", "test", "digest"))
        store, storage = make_store(backend)
        await storage.initialize()
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

    asyncio.run(run())


def test_file_asset_store_recovers_history_after_restart(tmp_path: Path) -> None:
    async def run() -> None:
        backend = FileAssetBackend(file_root(str(tmp_path)))
        store, storage = make_store(backend)
        await storage.initialize()
        key = AssetKey("sample", "one")
        first = await store.put(key, SampleAsset("sample", "one", "first"))
        await store.put(key, SampleAsset("sample", "one", "second"), expected_entry_revision=first.entry_revision)
        restarted = FileAssetBackend(file_root(str(tmp_path)))
        restarted_store, restarted_storage = make_store(restarted)
        await restarted_storage.initialize()
        assert await restarted_store.get(key, expected=SampleAsset) == SampleAsset("sample", "one", "second")
        assert len(await restarted_store.list_versions(key)) == 2

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
