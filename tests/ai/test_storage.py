#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generic Storage composition and cache conformance tests."""

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai import RuntimeStorage
from linktools.ai.adapter import SqlRuntimeSchema
from linktools.ai.asset import SqlAssetSchema
from linktools.ai.core import ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ToolOperationRecord
from linktools.ai.storage import (
    FilesystemContentCache,
    InMemoryContentCache,
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    SqlSchemaRegistry,
    register_storage_schema,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageLayer,
    StorageOwnedInfo,
    StorageOverlay,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    StorageValueValidator,
)

from tests.ai.persistence.helper import open_sql_resources


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
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StoragePutResult[Info]":
        del expected_entry_revision
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
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageDeleteResult[str]":
        del expected_entry_revision
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
        expected_entry_revision: "StorageEntryRevision | None" = None,
    ) -> "StorageResetResult[str]":
        value = self.values.get(key)
        if expected_entry_revision is not None and (
            value is None or value.revision != expected_entry_revision.value
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


def test_mysql_init_schema_matches_runtime_metadata_and_dba_rules() -> None:
    schema_path = Path("linktools-ai/migrations/init_schema.sql")
    sql = schema_path.read_text(encoding="utf-8")
    assert "not a production bootstrap or migration" in sql
    assert "reviewed, environment-specific DBA migration tooling" in sql
    matrix = json.loads(Path("linktools-ai/scripts/build/matrix/sql-schema-manifest.json").read_text(encoding="utf-8"))
    storage_registry = SqlSchemaRegistry()
    register_storage_schema(storage_registry)
    registry = SqlSchemaRegistry()
    SqlRuntimeSchema.register_schema(registry)
    manifest = registry.freeze()
    asset_registry = SqlSchemaRegistry()
    asset_tables = SqlAssetSchema.register_schema(asset_registry)
    asset_table_values = (asset_tables.entry, asset_tables.change, asset_tables.blob, asset_tables.revision)
    assert tuple(table.name for table in asset_table_values) == tuple(matrix["asset_tables"])
    expected_tables = tuple(matrix["storage_tables"]) + tuple(table.name for table in manifest.tables) + tuple(matrix["step_tables"]) + tuple(matrix["asset_tables"])
    actual_tables = tuple(re.findall(r"CREATE TABLE `([^`]+)`", sql))
    assert actual_tables == expected_tables
    assert matrix["runtime_digest"] == manifest.digest
    assert "CHARACTER SET ascii" not in sql
    assert " COLLATE ascii_bin" not in sql
    for table in tuple(storage_registry.metadata.sorted_tables) + tuple(registry.metadata.sorted_tables) + tuple(asset_table_values):
        for index in table.indexes:
            assert not index.name.startswith("ai_")
            assert len(index.columns) <= 3
        for constraint in table.constraints:
            if constraint.name is not None and constraint.__class__.__name__ == "UniqueConstraint":
                assert not constraint.name.startswith("ai_")
                assert len(constraint.columns) <= 3
    assert registry.metadata.tables["ai_runtime_sessions"].c.session_id.nullable is False
    assert registry.metadata.tables["ai_runtime_tools"].c.run_id.nullable is False
    assert registry.metadata.tables["ai_runtime_tools"].c.tool_call_id.nullable is False
    for table in registry.metadata.sorted_tables:
        block = re.search(
            rf"CREATE TABLE `{re.escape(table.name)}` \((.*?)\n\) ENGINE=.*?;",
            sql,
            re.DOTALL,
        )
        assert block is not None
        body = block.group(1)
        columns = tuple(re.findall(r"^  `([^`]+)` ", body, re.MULTILINE))
        assert columns[-2:] == ("updated_at", "created_at")
        assert set(columns) == {column.name for column in table.columns}
        assert all(f"`{column.name}`" in body and " COMMENT '" in body for column in table.columns)
        assert "COMMENT='" in block.group(0)
        timestamp_indexes = {
            tuple(column.name for column in index.columns)
            for index in table.indexes
            if len(index.columns) == 1
        }
        assert ("updated_at",) in timestamp_indexes
        assert ("created_at",) in timestamp_indexes
        for index in table.indexes:
            assert f"KEY `{index.name}`" in body
            for column in index.columns:
                if isinstance(getattr(column.type, "length", None), int) and column.type.length > 128:
                    assert f"`{column.name}`(128)" in body
        for constraint in table.constraints:
            if constraint.name is not None and constraint.__class__.__name__ == "UniqueConstraint":
                assert f"UNIQUE KEY `{constraint.name}`" in body
                for column in constraint.columns:
                    if isinstance(getattr(column.type, "length", None), int) and column.type.length > 128:
                        assert f"`{column.name}`(128)" in body
    for table_name in expected_tables:
        block = re.search(
            rf"CREATE TABLE `{re.escape(table_name)}` \((.*?)\n\) ENGINE=.*?;",
            sql,
            re.DOTALL,
        )
        assert block is not None
        body = block.group(1)
        columns = tuple(re.findall(r"^  `([^`]+)` ", body, re.MULTILINE))
        assert columns[0] == "id"
        assert columns[-2:] == ("updated_at", "created_at")
        assert "PRIMARY KEY (`id`)" in body
        assert "COMMENT='" in block.group(0)
        assert all(re.search(rf"^  `{re.escape(column)}` .* COMMENT '", body, re.MULTILINE) for column in columns)
        assert "KEY `ix_updated_at` (`updated_at`)" in body
        assert "KEY `ix_created_at` (`created_at`)" in body
    for table in asset_table_values:
        block = re.search(
            rf"CREATE TABLE `{re.escape(table.name)}` \((.*?)\n\) ENGINE=.*?;",
            sql,
            re.DOTALL,
        )
        assert block is not None
        columns = tuple(re.findall(r"^  `([^`]+)` ", block.group(1), re.MULTILINE))
        assert columns == tuple(column.name for column in table.columns)
    step_indexes = {
        "ai_step_runs": ("uk_namespace_key_run_id", "uk_namespace_key_run_key"),
        "ai_step_effects": ("uk_namespace_key_run_id_tool_call_id",),
        "ai_step_media": ("uk_namespace_key_sha256",),
    }
    for table_name in matrix["step_tables"]:
        block = re.search(
            rf"CREATE TABLE `{re.escape(table_name)}` \((.*?)\n\) ENGINE=.*?;",
            sql,
            re.DOTALL,
        )
        assert block is not None
        body = block.group(1)
        columns = tuple(re.findall(r"^  `([^`]+)` ", body, re.MULTILINE))
        assert columns[0] == "id"
        assert columns[-2:] == ("updated_at", "created_at")
        assert "KEY `ix_updated_at` (`updated_at`)" in body
        assert "KEY `ix_created_at` (`created_at`)" in body
        for index_name in step_indexes.get(table_name, ()):
            assert f"KEY `{index_name}`" in body
        if table_name == "ai_step_runs":
            assert re.search(r"^  `conversation_id` .* NOT NULL COMMENT '", body, re.MULTILINE)
    expected_runtime_keys = {
        "ai_runtime_blob_chunks": ("uk_namespace_key_tenant_id_chunk_key",),
        "ai_runtime_executions": (
            "ix_namespace_key_tenant_id_parent_execution_id",
            "ix_namespace_key_tenant_id_session_id",
        ),
        "ai_runtime_idempotency": ("uk_namespace_key_tenant_id_identity_key",),
        "ai_runtime_operation_counters": ("uk_namespace_key_tenant_id_partition_key",),
        "ai_runtime_tools": ("uk_namespace_key_tenant_id_call_key",),
    }
    for table_name, index_names in expected_runtime_keys.items():
        block = re.search(
            rf"CREATE TABLE `{re.escape(table_name)}` \((.*?)\n\) ENGINE=.*?;",
            sql,
            re.DOTALL,
        )
        assert block is not None
        for index_name in index_names:
            assert f"KEY `{index_name}`" in block.group(1)
    for table_name in ("ai_step_events", "ai_step_snapshots"):
        block = re.search(
            rf"CREATE TABLE `{table_name}` \((.*?)\n\) ENGINE=.*?;",
            sql,
            re.DOTALL,
        )
        assert block is not None
        assert "KEY `ix_namespace_key_run_id` (`namespace_key`, `run_id`(128))" in block.group(1)
    assert "ai_session_turns" not in sql
    assert "ai_execution_trace" not in sql
    assert "ai_runtime_repository_records" not in sql


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


@pytest.mark.asyncio
async def test_sql_tool_state_is_durable_and_conflict_safe(tmp_path: Path) -> None:
    timestamp = datetime.now(timezone.utc)
    storage = RuntimeStorage.sqlite(str(tmp_path / "tool-state.db"))
    async with open_sql_resources(storage, namespace="runtime") as runtime:
        record = ToolOperationRecord(
            "operation", "tenant", "run", "call", "a" * 64, "tool", "arguments", "binding", True,
            ToolOperationStatus.PENDING, None, 0, None, None, None, None, timestamp, timestamp,
        )
        await runtime.domain.recovery.tools.reserve(record)
        assert await runtime.domain.recovery.tools.get_operation("operation", tenant_id="tenant") == record
        claimed = await runtime.domain.recovery.tools.claim(
            "operation",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        assert claimed.owner == "worker"
        completed = await runtime.domain.recovery.tools.complete(
            "operation",
            tenant_id="tenant",
            owner="worker",
            fence=claimed.fence,
            result_ref="result",
            result_digest="b" * 64,
        )
        assert completed.status is ToolOperationStatus.COMPLETED
        with pytest.raises(AIError) as error:
            await runtime.domain.recovery.tools.reserve(
                ToolOperationRecord(
                    "operation", "tenant", "run", "other-call", "a" * 64, "tool", "arguments", "binding", True,
                    ToolOperationStatus.PENDING, None, 0, None, None, None, None, timestamp, timestamp,
                )
            )
        assert error.value.code == ErrorCode.TOOL_OPERATION_CONFLICT
