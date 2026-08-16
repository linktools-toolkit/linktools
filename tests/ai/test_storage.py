#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generic Storage composition and cache conformance tests."""

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from linktools.ai.asset import build_asset_sql_metadata
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state._schema import build_runtime_sql_metadata, build_step_sql_metadata
from linktools.ai.storage import (
    FilesystemContentCache,
    InMemoryContentCache,
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageLayer,
    StorageOverlay,
    StorageOwnedInfo,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    StorageValueValidator,
    build_object_sql_metadata,
)


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
        self.revision = StorageRevision(str(max((value.revision for value in values), default=1)))
        self.get_calls: list[str] = []
        self.metadata_calls = 0

    async def head_revision(self) -> StorageRevision:
        return self.revision

    async def load_metadata(self, after_revision: "StorageRevision | None") -> "MetadataLoad[str, Info]":
        self.metadata_calls += 1
        mode = MetadataLoadMode.REPLACE if after_revision is None else MetadataLoadMode.PATCH
        changes = tuple(MetadataChange(value.key, Info(value.key, value.revision)) for value in self.values.values())
        return MetadataLoad(mode, self.revision, changes)

    async def get(self, key: str) -> Value | None:
        self.get_calls.append(key)
        return self.values.get(key)


class WritableBackend(Backend):
    async def put(
        self,
        key: str,
        value: Value,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "StoragePutResult[Info]":
        del expected_revision, metadata
        self.values[key] = value
        self.revision = StorageRevision(str(int(self.revision.value) + 1))
        return StoragePutResult(
            Info(key, value.revision),
            StorageEntryRevision(value.revision),
            self.revision,
            True,
        )

    async def delete(
        self,
        key: str,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "StorageDeleteResult[str]":
        del expected_revision, metadata
        value = self.values.pop(key, None)
        self.revision = StorageRevision(str(int(self.revision.value) + 1))
        return StorageDeleteResult(
            key,
            value is not None,
            None if value is None else StorageEntryRevision(value.revision),
            self.revision,
        )

    async def reset(
        self,
        key: str,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "StorageResetResult[str]":
        del metadata
        value = self.values.get(key)
        if expected_revision is not None and (
            value is None or value.revision != expected_revision.value
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if value is None:
            return StorageResetResult(key, False, self.revision)
        self.values.pop(key)
        self.revision = StorageRevision(str(int(self.revision.value) + 1))
        return StorageResetResult(key, True, self.revision)


class Validator(StorageValueValidator[str, Value, Info]):
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
    cache = InMemoryContentCache(max_bytes=2)
    await cache.put("a", b"a")
    await cache.put("b", b"b")
    assert await cache.contains_many(("a",)) == frozenset({"a"})
    await cache.put("c", b"c")
    assert await cache.get("a") is None

    filesystem = FilesystemContentCache(tmp_path, max_bytes=10)
    await filesystem.put("stable-key", b"value")
    assert await filesystem.contains_many(("stable-key",)) == frozenset({"stable-key"})


@pytest.mark.asyncio
async def test_memory_cache_enforces_entry_count_and_replacement_accounting() -> None:
    cache = InMemoryContentCache(max_bytes=4, max_size=2)
    await cache.put("a", b"ab")
    await cache.put("b", b"c")
    assert await cache.get("a") == b"ab"
    await cache.put("c", b"d")
    assert await cache.contains_many(("a", "b", "c")) == frozenset({"a", "c"})

    await cache.put("a", b"x")
    await cache.put("d", b"yz")
    assert await cache.contains_many(("a", "c", "d")) == frozenset({"a", "d"})
    await cache.delete("a")
    await cache.put("e", b"1234")
    assert await cache.contains_many(("c", "d", "e")) == frozenset({"e"})


def test_memory_cache_validates_entry_count() -> None:
    with pytest.raises(ValueError):
        InMemoryContentCache(max_bytes=1, max_size=-1)
    assert InMemoryContentCache(max_bytes=1, max_size=None).max_size is None


def test_composition_exposes_only_domain_generics() -> None:
    assert tuple(parameter.__name__ for parameter in StorageOverlay.__parameters__) == (
        "KeyT",
        "ValueT",
        "InfoT",
    )
    assert str(StorageEntryRevision(1)) == "1"
    assert str(StorageRevision("revision")) == "revision"
    with pytest.raises(ValueError):
        StorageEntryRevision(0)
    with pytest.raises(ValueError):
        StorageRevision("")


@pytest.mark.asyncio
async def test_composition_writer_must_be_readable() -> None:
    primary = WritableBackend(())
    primary_storage = StorageOverlay(primary, writer=primary, validator=Validator())
    assert primary_storage.writer_is_primary is True
    await primary_storage.list_info()
    await primary_storage.put("primary", Value("primary", 2, b"primary"))
    primary_location = await primary_storage.locate("primary")
    assert primary_location is not None
    assert primary_location.backend is primary
    assert primary_location.writable is True
    assert primary.metadata_calls == 1

    layer = WritableBackend(())
    layer_storage = StorageOverlay(
        Backend(()),
        writer=layer,
        layers=(StorageLayer("writer", layer),),
        validator=Validator(),
    )
    assert layer_storage.writer_is_primary is False
    await layer_storage.list_info()
    await layer_storage.put("layer", Value("layer", 2, b"layer"))
    layer_location = await layer_storage.locate("layer")
    assert layer_location is not None
    assert layer_location.backend is layer
    assert layer_location.writable is True
    assert layer.metadata_calls == 1

    with pytest.raises(ValueError, match="writer must be one of the read backends"):
        StorageOverlay(primary, writer=WritableBackend(()))


def test_sql_metadata_builders_are_owner_scoped() -> None:
    runtime_metadata = build_runtime_sql_metadata(frozenset(RuntimeDomain))
    asset_metadata = build_asset_sql_metadata()
    step_metadata = build_step_sql_metadata(RuntimeDomain.EXECUTION)
    object_metadata = build_object_sql_metadata()

    assert all(name.startswith("ai_runtime_") for name in runtime_metadata.tables)
    assert {"ai_runtime_sessions", "ai_runtime_executions"} <= set(runtime_metadata.tables)
    assert set(step_metadata.tables) == {"ai_step_runs", "ai_step_events", "ai_step_snapshots"}
    assert set(object_metadata.tables) == {"ai_storage_objects", "ai_storage_object_chunks"}
    assert set(asset_metadata.tables) == {"ai_asset_entries", "ai_asset_changes", "ai_asset_heads"}
    assert "ai_asset_entries" not in runtime_metadata.tables
    assert "ai_runtime_sessions" not in asset_metadata.tables

    recovery_metadata = build_runtime_sql_metadata(
        frozenset({RuntimeDomain.RECOVERY})
    )
    assert "ai_step_effects" not in recovery_metadata.tables
    assert "ai_runtime_sessions" not in recovery_metadata.tables


def test_sql_database_kernel_has_no_domain_schema_catalog() -> None:
    source = Path("linktools-ai/src/linktools/ai/storage/_database.py").read_text(encoding="utf-8")
    forbidden_terms = (
        "ai_runtime_",
        "ai_step_",
        "ai_storage_objects",
        "ai_storage_object_chunks",
        "ai_asset_",
        "ObjectStore",
    )
    assert not any(term in source for term in forbidden_terms)

    import linktools.ai.storage as storage

    assert not hasattr(storage, "sql_table_comment")
    assert not hasattr(storage, "sql_column_comment")

    ddl = Path("linktools-ai/migrations/init_schema.sql").read_text(encoding="utf-8")
    comments = re.findall(r"COMMENT(?:=| )'([^']*)'", ddl)
    assert comments
    assert all(all(ord(char) < 128 for char in comment) for comment in comments)


@pytest.mark.asyncio
async def test_composition_uses_metadata_owner_and_does_not_probe_absent_keys() -> None:
    primary = Backend((Value("primary", 1, b"p"),))
    fallback = Backend((Value("fallback", 1, b"f"),))
    storage = StorageOverlay(
        primary,
        layers=(StorageLayer("fallback", fallback),),
        validator=Validator(),
    )
    assert await storage.get("missing") is None
    assert primary.get_calls == []
    assert fallback.get_calls == []
    assert (await storage.get("fallback")).content == b"f"
    assert fallback.get_calls == ["fallback"]
    location = await storage.locate("fallback")
    assert location is not None
    assert location.backend is fallback
    assert location.layer == "fallback"
    assert (await storage.list_info_with_owners()) == (
        StorageOwnedInfo(Info("fallback", 1), "fallback", False),
        StorageOwnedInfo(Info("primary", 1), "primary", False),
    )


@pytest.mark.asyncio
async def test_refresh_single_flight_and_cancelled_waiter() -> None:
    class SlowBackend(Backend):
        async def load_metadata(self, after_revision: "StorageRevision | None") -> "MetadataLoad[str, Info]":
            await asyncio.sleep(0.02)
            return await super().load_metadata(after_revision)

    backend = SlowBackend((Value("one", 1, b"1"),))
    storage = StorageOverlay(backend, validator=Validator())
    first = asyncio.create_task(storage.refresh())
    second = asyncio.create_task(storage.refresh())
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    await first
    assert backend.metadata_calls == 1


@pytest.mark.asyncio
async def test_read_only_write_is_fail_closed() -> None:
    storage = StorageOverlay(Backend(()), validator=Validator())
    with pytest.raises(AIError) as error:
        await storage.put("key", Value("key", 1, b"value"))
    assert error.value.code == ErrorCode.STORAGE_READ_ONLY
