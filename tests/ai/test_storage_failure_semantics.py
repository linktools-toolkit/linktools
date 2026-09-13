#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace configuration and storage failure semantics."""

from pathlib import Path

import pytest
from linktools.ai.asset import (
    AssetCacheAdapter,
    AssetKey,
    AssetRoot,
    InMemoryAssetBackend,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryContentCache, StorageOverlay
from linktools.ai.workspace._root import load_config


class _ValidatorFailure(RuntimeError):
    pass


class _Validator:
    def __init__(self, *, fail_once: bool = False) -> None:
        self._fail_once = fail_once
        self._calls = 0

    def validate_value(self, key: AssetKey, value: bytes, info: object) -> None:
        del key, value, info
        self._calls += 1
        if not self._fail_once or self._calls == 1:
            raise _ValidatorFailure("validator failure")


class _MissingOriginBackend(InMemoryAssetBackend):
    async def get(self, key: AssetKey) -> "bytes | None":
        del key
        return None


def _asset_root(name: str) -> AssetRoot:
    return AssetRoot(f"memory:{name}", "memory", name, name)


def test_workspace_config_defaults_and_mapping(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.yaml") == {}

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config(empty) == {}

    valid = tmp_path / "valid.yaml"
    valid.write_text(
        "name: demo\nnested:\n  enabled: true\n  count: 2\n",
        encoding="utf-8",
    )
    assert load_config(valid) == {
        "name": "demo",
        "nested": {"enabled": True, "count": 2},
    }


@pytest.mark.parametrize(
    "content",
    (
        "[unterminated",
        "- one\n- two\n",
        "value: 2026-09-03\n",
    ),
)
def test_workspace_config_rejects_invalid_documents(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(AIError) as raised:
        load_config(path)

    assert raised.value.code is ErrorCode.WORKSPACE_CONFIG_INVALID


@pytest.mark.asyncio
async def test_origin_validator_failure_propagates() -> None:
    backend = InMemoryAssetBackend(_asset_root("validator-single"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    storage = StorageOverlay(backend, validator=_Validator())

    with pytest.raises(_ValidatorFailure):
        await storage.get(key)


@pytest.mark.asyncio
async def test_batch_origin_validator_failure_propagates() -> None:
    backend = InMemoryAssetBackend(_asset_root("validator-batch"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    storage = StorageOverlay(backend, validator=_Validator())

    with pytest.raises(_ValidatorFailure):
        await storage.get_many((key,))


@pytest.mark.asyncio
async def test_missing_origin_value_reports_integrity_error() -> None:
    backend = _MissingOriginBackend(_asset_root("missing-origin"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    storage = StorageOverlay(backend)

    with pytest.raises(AIError) as raised:
        await storage.get(key)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_corrupt_cache_falls_back_to_origin() -> None:
    backend = InMemoryAssetBackend(_asset_root("cache-conversion"))
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
    await cache.put(adapter.cache_key(key, info), b"corrupt")
    storage = StorageOverlay(backend, cache=cache, cache_adapter=adapter)

    assert await storage.get(key) == b"value"


@pytest.mark.asyncio
async def test_invalid_cached_value_falls_back_to_valid_origin() -> None:
    backend = InMemoryAssetBackend(_asset_root("cache-validator"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    info = await backend.stat(key)
    assert info is not None

    adapter = AssetCacheAdapter()
    cache = InMemoryContentCache(max_bytes=1024)
    await cache.put(adapter.cache_key(key, info), adapter.to_cache(b"value"))
    storage = StorageOverlay(
        backend,
        cache=cache,
        cache_adapter=adapter,
        validator=_Validator(fail_once=True),
    )

    assert await storage.get(key) == b"value"


@pytest.mark.asyncio
async def test_preload_propagates_origin_validation_failure() -> None:
    backend = InMemoryAssetBackend(_asset_root("preload-validator"))
    key = AssetKey("sample", "one")
    await backend.put(key, b"value")
    storage = StorageOverlay(
        backend,
        cache=InMemoryContentCache(max_bytes=1024),
        cache_adapter=AssetCacheAdapter(),
        validator=_Validator(),
    )

    with pytest.raises(_ValidatorFailure):
        await storage.preload((key,))
