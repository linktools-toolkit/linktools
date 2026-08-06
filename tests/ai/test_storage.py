#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Storage kernel conformance tests. MT-23 MT-29 MT-35 MT-36 MT-37 MT-38 MT-39."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from linktools.ai.storage.cache import FilesystemContentCache, MemoryContentCache
from linktools.ai.storage.composition import StorageComposition, StorageLayer
from linktools.ai.storage.database import CoordinationScope, build_sqlite_storage
from linktools.ai.storage.initialization import initialize_storage
from linktools.ai.storage.revision import MetadataLoad, MetadataLoadMode, StorageChange


@dataclass(frozen=True)
class _Info:
    key: str
    revision: int


@dataclass(frozen=True)
class _Value:
    key: str
    revision: int
    content: bytes


class _Backend:
    def __init__(self, values: tuple[_Value, ...]) -> None:
        self.values = {value.key: value for value in values}
        self.revision = max((value.revision for value in values), default=1)
        self.get_calls: list[str] = []
        self.metadata_calls = 0

    async def head_revision(self) -> int:
        return self.revision

    async def load_metadata(self, after_revision: int | None) -> MetadataLoad[int, str, _Info]:
        self.metadata_calls += 1
        mode = MetadataLoadMode.REPLACE if after_revision is None else MetadataLoadMode.PATCH
        changes = tuple(StorageChange(key, _Info(value.key, value.revision)) for key, value in self.values.items())
        return MetadataLoad(self.revision, mode, changes)

    async def list_info(self) -> tuple[_Info, ...]:
        return tuple(_Info(value.key, value.revision) for value in self.values.values())

    async def get(self, key: str) -> _Value | None:
        self.get_calls.append(key)
        return self.values.get(key)


class _Adapter:
    def info_key(self, info: _Info) -> str:
        return info.key

    def value_info(self, value: _Value) -> _Info:
        return _Info(value.key, value.revision)


class _CacheAdapter:
    def cache_key(self, key: str, info: _Info) -> str:
        return f"{key}:{info.revision}"

    def cache_content(self, value: _Value) -> bytes:
        return value.content

    def from_cache(self, info: _Info, content: bytes) -> _Value:
        return _Value(info.key, info.revision, content)


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
    assert (tmp_path / __import__("hashlib").sha256(b"stable-key").hexdigest()).exists()


@pytest.mark.asyncio
async def test_composition_uses_metadata_owner_and_does_not_probe_absent_keys() -> None:
    primary = _Backend((_Value("primary", 1, b"p"),))
    fallback = _Backend((_Value("fallback", 1, b"f"),))
    storage = StorageComposition(
        primary,
        layers=(StorageLayer(fallback),),
        adapter=_Adapter(),
        cache_adapter=_CacheAdapter(),
    )
    assert await storage.get("missing") is None
    assert primary.get_calls == []
    assert fallback.get_calls == []
    assert (await storage.get("fallback")).content == b"f"
    assert fallback.get_calls == ["fallback"]


@pytest.mark.asyncio
async def test_sqlite_construction_is_lazy_and_initialization_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "storage.db"
    storage = build_sqlite_storage(path)
    assert storage.coordination_scope is CoordinationScope.PROCESS
    assert not path.exists()
    await initialize_storage(storage)
    assert path.exists()
    await storage.engine.dispose()
