#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failure-semantics regressions for the overdesign-defense refactor."""

from pathlib import Path

import pytest
import linktools.ai.asset as asset
import linktools.ai.errors as errors
from linktools.ai.asset import AssetCacheAdapter, AssetInfo, AssetKey, AssetRoot, InMemoryAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import (
    InMemoryContentCache,
    MetadataLoad,
    StorageOverlay,
    StorageRevision,
)
from linktools.ai.workspace._root import load_config


class _CountingAssetBackend(InMemoryAssetBackend):
    def __init__(self, root: AssetRoot) -> None:
        super().__init__(root)
        self.metadata_loads = 0
        self.reads = 0
        self.batch_reads = 0

    async def load_metadata(
        self,
        after_revision: "StorageRevision | None",
    ) -> "MetadataLoad[AssetKey, AssetInfo]":
        self.metadata_loads += 1
        return await super().load_metadata(after_revision)

    async def get(self, key: AssetKey) -> "bytes | None":
        self.reads += 1
        return await super().get(key)

    async def get_many(self, keys: "tuple[AssetKey, ...]") -> "dict[AssetKey, bytes]":
        self.batch_reads += 1
        return await super().get_many(keys)


class _MissingOriginBackend(_CountingAssetBackend):
    async def get(self, key: AssetKey) -> "bytes | None":
        self.reads += 1
        return None


class _ValidatorFailure(RuntimeError):
    pass


class _Validator:
    def __init__(self, failure: "BaseException | None" = None) -> None:
        self.failure = failure
        self.calls = 0

    def validate_value(self, key: AssetKey, value: bytes, info: AssetInfo) -> None:
        del key, value, info
        self.calls += 1
        if self.failure is not None:
            raise self.failure


def _asset_root(name: str) -> AssetRoot:
    return AssetRoot(f"memory:{name}", "memory", name, name)


def test_workspace_config_missing_empty_null_and_mapping(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    assert load_config(missing) == {}

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config(empty) == {}

    null = tmp_path / "null.yaml"
    null.write_text("null\n", encoding="utf-8")
    assert load_config(null) == {}

    valid = tmp_path / "valid.yaml"
    valid.write_text("name: demo\nnested:\n  enabled: true\n  count: 2\n", encoding="utf-8")
    assert load_config(valid) == {
        "name": "demo",
        "nested": {"enabled": True, "count": 2},
    }


@pytest.mark.parametrize(
    "content",
    (
        "[unterminated",
        "- one\n- two\n",
        "scalar\n",
        "1: value\n",
        "value: 2026-09-03\n",
        "value: !!set {one: null}\n",
    ),
)
def test_workspace_config_invalid_inputs_fail_closed(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(AIError) as raised:
        load_config(path)
    assert raised.value.code is ErrorCode.WORKSPACE_CONFIG_INVALID
    assert raised.value.category == "WORKSPACE"
    assert raised.value.retryable is False
    assert raised.value.safe_details == {}


def test_removed_specialized_error_names_are_not_exported() -> None:
    removed = {
        "StorageConflictError",
        "StorageCorruptionError",
        "AssetConflictError",
        "AssetNotFoundError",
        "AssetParseError",
        "InvalidAssetError",
    }
    assert removed.isdisjoint(errors.__all__)
    assert removed.isdisjoint(asset.__all__)
    assert all(not hasattr(errors, name) for name in removed)
    assert all(not hasattr(asset, name) for name in removed)
    assert "StorageError" in errors.__all__
    assert "InvalidStoragePathError" in errors.__all__
    assert "AssetError" in errors.__all__


@pytest.mark.asyncio
async def test_origin_validator_failure_propagates_without_metadata_refresh() -> None:
    backend = _CountingAssetBackend(_asset_root("validator-single"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    failure = _ValidatorFailure("domain failure")
    validator = _Validator(failure)
    storage = StorageOverlay(backend, validator=validator)

    with pytest.raises(_ValidatorFailure) as raised:
        await storage.get(key)
    assert raised.value is failure
    assert validator.calls == 1
    assert backend.metadata_loads == 1
    assert backend.reads == 1


@pytest.mark.asyncio
async def test_batch_origin_validator_failure_propagates_without_metadata_refresh() -> None:
    backend = _CountingAssetBackend(_asset_root("validator-batch"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    failure = _ValidatorFailure("domain failure")
    validator = _Validator(failure)
    storage = StorageOverlay(backend, validator=validator)

    with pytest.raises(_ValidatorFailure) as raised:
        await storage.get_many((key,))
    assert raised.value is failure
    assert validator.calls == 1
    assert backend.metadata_loads == 1
    assert backend.batch_reads == 1


@pytest.mark.asyncio
async def test_missing_origin_value_refreshes_once_then_reports_integrity_error() -> None:
    backend = _MissingOriginBackend(_asset_root("missing-origin"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    storage = StorageOverlay(backend)

    with pytest.raises(AIError) as raised:
        await storage.get(key)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert backend.metadata_loads == 2
    assert backend.reads == 2


@pytest.mark.asyncio
async def test_cache_conversion_failure_falls_back_to_origin() -> None:
    backend = _CountingAssetBackend(_asset_root("cache-conversion"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    info = await backend.stat(key)
    assert info is not None

    class _CorruptingCacheAdapter(AssetCacheAdapter):
        def from_cache(self, value: bytes) -> bytes:
            if value == b"corrupt":
                raise ValueError("corrupt cache")
            return super().from_cache(value)

    adapter = _CorruptingCacheAdapter()
    cache = InMemoryContentCache(max_bytes=1024)
    cache_key = adapter.cache_key(key, info)
    await cache.put(cache_key, b"corrupt")
    storage = StorageOverlay(
        backend,
        cache=cache,
        cache_adapter=adapter,
        validator=_Validator(),
    )

    assert await storage.get(key) == b"value"
    assert backend.reads == 1
    cached = await cache.get(cache_key)
    assert cached is not None
    assert adapter.from_cache(cached) == b"value"


@pytest.mark.asyncio
async def test_cached_validator_failure_falls_back_then_origin_failure_surfaces() -> None:
    backend = _CountingAssetBackend(_asset_root("cache-validator"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    info = await backend.stat(key)
    assert info is not None
    adapter = AssetCacheAdapter()
    cache = InMemoryContentCache(max_bytes=1024)
    await cache.put(adapter.cache_key(key, info), adapter.to_cache(b"value"))
    failure = _ValidatorFailure("validator failure")
    validator = _Validator(failure)
    storage = StorageOverlay(
        backend,
        cache=cache,
        cache_adapter=adapter,
        validator=validator,
    )

    with pytest.raises(_ValidatorFailure) as raised:
        await storage.get(key)
    assert raised.value is failure
    assert validator.calls == 2
    assert backend.reads == 1


@pytest.mark.asyncio
async def test_preload_origin_validator_failure_propagates() -> None:
    backend = _CountingAssetBackend(_asset_root("preload-validator"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    failure = _ValidatorFailure("preload validator failure")
    storage = StorageOverlay(
        backend,
        cache=InMemoryContentCache(max_bytes=1024),
        cache_adapter=AssetCacheAdapter(),
        validator=_Validator(failure),
    )

    with pytest.raises(_ValidatorFailure) as raised:
        await storage.preload((key,))
    assert raised.value is failure
