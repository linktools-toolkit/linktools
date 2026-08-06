#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generic Storage composition and cache conformance tests."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from linktools.ai.core.errors import ErrorCode, LinktoolsAIError
from linktools.ai.adapter.tool import SqlToolState
from linktools.ai.asset.sql import SqlAlchemyAssetBackend
from linktools.ai.capability.tool import ToolState
from linktools.ai.storage.cache import FilesystemContentCache, MemoryContentCache
from linktools.ai.storage.composition import StorageAdapter, StorageComposition
from linktools.ai.storage.database import SqlSchemaRegistry, build_sqlite_storage, close_storage
from linktools.ai.storage.initialize import initialize_storage
from linktools.ai.storage.layer import StorageLayer
from linktools.ai.storage.model import MetadataChange, MetadataLoad, MetadataLoadMode, StorageOwnedInfo


@dataclass(frozen=True, slots=True)
class Info:
    key: str
    revision: int


@dataclass(frozen=True, slots=True)
class Value:
    key: str
    revision: int
    content: bytes


class Backend:
    def __init__(self, values: tuple[Value, ...]) -> None:
        self.values = {value.key: value for value in values}
        self.revision = max((value.revision for value in values), default=1)
        self.get_calls: list[str] = []
        self.metadata_calls = 0

    async def head_revision(self) -> int:
        return self.revision

    async def load_metadata(self, after_revision: int | None) -> MetadataLoad[str, Info, int]:
        self.metadata_calls += 1
        mode = MetadataLoadMode.REPLACE if after_revision is None else MetadataLoadMode.PATCH
        changes = tuple(MetadataChange(value.key, Info(value.key, value.revision)) for value in self.values.values())
        return MetadataLoad(mode, self.revision, changes)

    async def get(self, key: str) -> Value | None:
        self.get_calls.append(key)
        return self.values.get(key)


class Adapter(StorageAdapter[str, Value, str, Value, Info]):
    def to_storage_key(self, key: str) -> str:
        return key

    def from_storage_key(self, key: str) -> str:
        return key

    def from_storage_value(self, value: Value) -> Value:
        return value

    def to_storage_value(self, value: Value) -> Value:
        return value

    def validate_value(self, key: str, value: Value, info: Info) -> None:
        if value.key != key or value.revision != info.revision:
            raise ValueError("storage value mismatch")


class CacheAdapter:
    def cache_key(self, key: str, info: Info) -> str:
        return f"{key}:{info.revision}"

    def to_cache(self, value: Value) -> bytes:
        return value.content

    def from_cache(self, value: bytes) -> Value:
        return Value("cached", 1, value)


@pytest.mark.asyncio
async def test_cache_contains_does_not_touch_lru_and_files_are_hashed(tmp_path: Path) -> None:
    cache = MemoryContentCache(max_bytes=2)
    await cache.put("a", b"a")
    await cache.put("b", b"b")
    assert await cache.contains_many(("a",)) == frozenset({"a"})
    await cache.put("c", b"c")
    assert await cache.get("a") is None

    filesystem = FilesystemContentCache(tmp_path, max_bytes=10)
    await filesystem.put("stable-key", b"value")
    assert await filesystem.contains_many(("stable-key",)) == frozenset({"stable-key"})


@pytest.mark.asyncio
async def test_composition_uses_metadata_owner_and_does_not_probe_absent_keys() -> None:
    primary = Backend((Value("primary", 1, b"p"),))
    fallback = Backend((Value("fallback", 1, b"f"),))
    storage = StorageComposition(
        primary,
        layers=(StorageLayer("fallback", fallback),),
        adapter=Adapter(),
    )
    assert await storage.get("missing") is None
    assert primary.get_calls == []
    assert fallback.get_calls == []
    assert (await storage.get("fallback")).content == b"f"
    assert fallback.get_calls == ["fallback"]
    assert (await storage.list_info_with_owners()) == (
        StorageOwnedInfo(Info("fallback", 1), "fallback", False),
        StorageOwnedInfo(Info("primary", 1), "primary", False),
    )


@pytest.mark.asyncio
async def test_refresh_single_flight_and_cancelled_waiter() -> None:
    class SlowBackend(Backend):
        async def load_metadata(self, after_revision: int | None) -> MetadataLoad[str, Info, int]:
            await asyncio.sleep(0.02)
            return await super().load_metadata(after_revision)

    backend = SlowBackend((Value("one", 1, b"1"),))
    storage = StorageComposition(backend, adapter=Adapter())
    first = asyncio.create_task(storage.refresh())
    second = asyncio.create_task(storage.refresh())
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    await first
    assert backend.metadata_calls == 1


@pytest.mark.asyncio
async def test_read_only_write_is_fail_closed() -> None:
    storage = StorageComposition(Backend(()), adapter=Adapter())
    with pytest.raises(LinktoolsAIError) as error:
        await storage.put("key", Value("key", 1, b"value"))
    assert error.value.code == ErrorCode.STORAGE_READ_ONLY


@pytest.mark.asyncio
async def test_sql_tool_state_is_durable_and_conflict_safe(tmp_path: Path) -> None:
    registry = SqlSchemaRegistry()
    SqlAlchemyAssetBackend.register_schema(registry)
    tool_table = SqlToolState.register_schema(registry)
    manifest = registry.freeze()
    database = build_sqlite_storage(
        tmp_path / "tool-state.db",
        metadata=registry.metadata,
        schema_manifest_digest=manifest.digest,
    )
    await initialize_storage(database)
    try:
        first = SqlToolState(database.session_factory, table=tool_table)
        await first.put(ToolState("operation", "completed", "digest"))
        second = SqlToolState(database.session_factory, table=tool_table)
        assert await second.get("operation") == ToolState("operation", "completed", "digest")
        with pytest.raises(LinktoolsAIError) as error:
            await second.put(ToolState("operation", "failed", "other"))
        assert error.value.code == ErrorCode.STORAGE_CONFLICT
    finally:
        await close_storage(database)
