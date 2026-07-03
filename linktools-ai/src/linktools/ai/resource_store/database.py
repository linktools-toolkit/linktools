#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DatabaseBackend: abstract ResourceBackend absorbing what used to be a separate
injected Repository protocol -- an external subclass fills in the `_raw_*` methods
with real SQL access (this repo has never contained that implementation and still
doesn't). Owns all cache/revision/idempotency logic concretely: a resident content
tier (unbounded, populated by get_by_name -- see this plan's "Gaps found during
planning" §3), a bounded LRU tier for everything else, a bounded historical-version
LRU, an optional disk L2 cache (checksum-validated sidecar files), and an optional
Redis-backed revision counter + distributed write lock (absent Redis, revision
tracking degrades to a plain in-process counter and locking falls back to
asyncio.Lock -- matching registry_store's prior behavior exactly)."""

import asyncio
import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator, Protocol, runtime_checkable

from .protocols import Operation, ResourceFile

logger = logging.getLogger("linktools.ai.resource_store.database")

_REVISION_KEY = "resource_store:revision"
_LRU_MAX_FILES = 256
_MAX_CACHE_BYTES = 1 * 1024 * 1024
_SYNC_LOOKBACK = timedelta(minutes=1)


@dataclass(slots=True, frozen=True)
class _RawRow:
    """Internal-only row shape for the _raw_* contract -- never exposed publicly."""

    row_id: int
    path: str
    content: str
    version: int
    status: str  # "active" | "deleted"
    updated_at: datetime


@runtime_checkable
class _RedisConfigProtocol(Protocol):
    enabled: bool


@runtime_checkable
class RedisCoordinatorProtocol(Protocol):
    config: _RedisConfigProtocol

    async def get(self, key: str) -> "str | None": ...
    async def incr(self, key: str) -> int: ...
    async def delete(self, key: str) -> None: ...
    async def try_acquire(self, key: str, value: str, ttl: int) -> bool: ...
    async def release_if_owner(self, key: str, value: str) -> bool: ...


def _exceeds_cache_limit(content: str) -> bool:
    return len(content) > _MAX_CACHE_BYTES or len(content.encode("utf-8")) > _MAX_CACHE_BYTES


def _checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class DatabaseBackend(ABC):
    def __init__(self, *, redis: "RedisCoordinatorProtocol", workspace_root: "Path | None" = None) -> None:
        self.redis = redis
        self._workspace_root = workspace_root
        self._local_revision: int = 0
        self._last_sync_at: "datetime | None" = None
        self._initialized: bool = False
        self._lock = asyncio.Lock()
        # Resident tier: populated exclusively by get_by_name, never evicted -- see
        # this plan's "Gaps found during planning" §3 for why residency is earned by
        # access pattern instead of a pre-registered kind->primary-filename table.
        self._resident: "dict[str, str]" = {}
        self._lru: "OrderedDict[str, str]" = OrderedDict()
        self._index: "dict[str, tuple[int, int]]" = {}  # path -> (row_id, version)
        self._deleted: "set[str]" = set()
        self._hist: "OrderedDict[tuple[int, int], str]" = OrderedDict()
        self._write_locks: "dict[str, asyncio.Lock]" = {}

    # ── Abstract raw-storage contract (external subclass implements against real SQL) ──

    @abstractmethod
    async def _raw_get(self, path: str) -> "_RawRow | None": ...

    @abstractmethod
    async def _raw_get_by_name(self, namespace: str, name: str) -> "list[_RawRow]": ...

    @abstractmethod
    async def _raw_upsert(self, path: str, content: str, checksum: str, updated_by: str) -> "tuple[int, int, bool]": ...

    @abstractmethod
    async def _raw_delete(self, path: str, updated_by: str) -> bool: ...

    @abstractmethod
    async def _raw_move(self, src_path: str, dst_path: str, updated_by: str) -> "tuple[int, int] | None": ...

    @abstractmethod
    async def _raw_list_since(self, since: "datetime | None") -> "list[_RawRow]": ...

    @abstractmethod
    async def _raw_apply_batch(self, ops: "list[Operation]", *, expected_revision: str, updated_by: str) -> "tuple[str, list[_RawRow]]":
        """Apply every op transactionally; raise on `expected_revision` mismatch.

        Return only the rows touched by these ops (one per distinct path written or
        deleted) -- not the full table. A batch here can span arbitrary, unrelated
        paths, unlike the old capability-scoped save_batch, so "the complete state of
        one capability" has no analog; only per-op results generalize."""
        ...

    # ── Content cache tiers ──

    def _content_get(self, path: str) -> "str | None":
        cached = self._resident.get(path)
        if cached is not None:
            return cached
        cached = self._lru.get(path)
        if cached is not None:
            self._lru.move_to_end(path)
        return cached

    def _content_put(self, path: str, content: str, *, resident: bool) -> None:
        if resident:
            self._resident[path] = content
            return
        if _exceeds_cache_limit(content):
            self._lru.pop(path, None)
            return
        self._lru[path] = content
        self._lru.move_to_end(path)
        while len(self._lru) > _LRU_MAX_FILES:
            self._lru.popitem(last=False)

    def _content_pop(self, path: str) -> None:
        self._resident.pop(path, None)
        self._lru.pop(path, None)

    def _hist_get(self, row_id: int, version: int) -> "str | None":
        cached = self._hist.get((row_id, version))
        if cached is not None:
            self._hist.move_to_end((row_id, version))
        return cached

    def _hist_put(self, row_id: int, version: int, content: str) -> None:
        if _exceeds_cache_limit(content):
            return
        self._hist[(row_id, version)] = content
        self._hist.move_to_end((row_id, version))
        while len(self._hist) > _LRU_MAX_FILES:
            self._hist.popitem(last=False)

    # ── Disk L2 cache ──

    def _write_workspace(self, row_id: int, version: int, content: str) -> None:
        if self._workspace_root is None:
            return
        dest = self._workspace_root / f"{row_id}.v{version}"
        meta = self._workspace_root / f"{row_id}.v{version}.meta.json"
        try:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            meta.write_text(json.dumps({"checksum": _checksum(content)}), encoding="utf-8")
        except OSError as exc:
            logger.warning("resource_store: workspace write failed (%s.v%s): %s", row_id, version, exc)

    def _read_workspace(self, row_id: int, version: int) -> "str | None":
        if self._workspace_root is None:
            return None
        dest = self._workspace_root / f"{row_id}.v{version}"
        meta = self._workspace_root / f"{row_id}.v{version}.meta.json"
        try:
            if not dest.exists() or not meta.exists():
                return None
            content = dest.read_text(encoding="utf-8")
            expected = json.loads(meta.read_text(encoding="utf-8")).get("checksum")
            if not expected or _checksum(content) != expected:
                dest.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                return None
            return content
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("resource_store: workspace read failed (%s.v%s): %s", row_id, version, exc)
            return None

    # ── Locking ──

    @asynccontextmanager
    async def write_lock(self, path: str) -> "AsyncIterator[None]":
        if not self.redis.config.enabled:
            if path not in self._write_locks:
                self._write_locks[path] = asyncio.Lock()
            async with self._write_locks[path]:
                yield
            return
        lock_key = f"resource_store:wlock:{path}"
        lock_val = uuid.uuid4().hex
        ttl = 30
        acquired = False
        for _ in range(100):
            if await self.redis.try_acquire(lock_key, lock_val, ttl):
                acquired = True
                break
            await asyncio.sleep(0.1)
        if not acquired:
            raise TimeoutError(f"write_lock timeout for {path}")
        try:
            yield
        finally:
            try:
                await self.redis.release_if_owner(lock_key, lock_val)
            except Exception as exc:
                logger.warning("write_lock release failed (%s): %s", path, exc)

    # ── Revision / sync ──

    async def get_revision(self) -> int:
        if not self.redis.config.enabled:
            return self._local_revision
        try:
            value = await self.redis.get(_REVISION_KEY)
            return int(value) if value else 0
        except Exception as exc:
            logger.warning("resource_store: redis get failed, forcing resync: %s", exc)
            return self._local_revision + 1

    async def _bump_revision(self) -> None:
        self._local_revision += 1
        if not self.redis.config.enabled:
            return
        try:
            await self.redis.incr(_REVISION_KEY)
        except Exception as exc:
            logger.warning("resource_store: redis revision bump failed: %s", exc)

    async def _ensure_fresh(self) -> None:
        if not self._initialized:
            await self._sync()
            return
        if not self.redis.config.enabled:
            return
        if await self.get_revision() != self._local_revision:
            await self._sync()

    async def _sync(self) -> None:
        async with self._lock:
            try:
                remote = await self.get_revision()
            except Exception as exc:
                logger.warning("resource_store: sync failed, using cached memory: %s", exc)
                return
            full_reconcile = self._initialized and remote != self._local_revision
            since = None if (full_reconcile or self._last_sync_at is None) else self._last_sync_at - _SYNC_LOOKBACK
            try:
                rows = await self._raw_list_since(since)
            except Exception as exc:
                logger.warning("resource_store: sync query failed: %s", exc)
                return
            if full_reconcile:
                self._resident.clear()
                self._lru.clear()
                self._index.clear()
                self._deleted.clear()
            max_updated_at = self._last_sync_at
            for row in rows:
                self._apply_row(row)
                if max_updated_at is None or row.updated_at > max_updated_at:
                    max_updated_at = row.updated_at
            self._last_sync_at = max_updated_at
            self._local_revision = remote
            self._initialized = True

    def _apply_row(self, row: "_RawRow") -> None:
        if row.status == "deleted":
            self._content_pop(row.path)
            self._index.pop(row.path, None)
            self._deleted.add(row.path)
            return
        self._index[row.path] = (row.row_id, row.version)
        self._deleted.discard(row.path)
        self._write_workspace(row.row_id, row.version, row.content)
        # Sync doesn't know which paths were reached via get_by_name -- only promote
        # to resident if this path is already there; otherwise route through the LRU.
        self._content_put(row.path, row.content, resident=row.path in self._resident)

    # ── Public ResourceBackend surface ──

    async def get(self, path: str) -> "ResourceFile | None":
        await self._ensure_fresh()
        cached = self._content_get(path)
        if cached is not None:
            return ResourceFile(path=path, content=cached, version=self._index[path][1])
        entry = self._index.get(path)
        if entry is not None:
            disk_content = self._read_workspace(*entry)
            if disk_content is not None:
                self._content_put(path, disk_content, resident=False)
                return ResourceFile(path=path, content=disk_content, version=entry[1])
        row = await self._raw_get(path)
        if row is None:
            return None
        self._index[path] = (row.row_id, row.version)
        self._write_workspace(row.row_id, row.version, row.content)
        self._content_put(path, row.content, resident=False)
        return ResourceFile(path=path, content=row.content, version=row.version)

    async def get_at_version(self, path: str, version: int) -> "ResourceFile | None":
        entry = self._index.get(path)
        row_id = entry[0] if entry else None
        if entry and entry[1] == version:
            cached = self._content_get(path)
            if cached is not None:
                return ResourceFile(path=path, content=cached, version=version)
        if row_id is not None:
            hist = self._hist_get(row_id, version)
            if hist is not None:
                return ResourceFile(path=path, content=hist, version=version)
            disk_content = self._read_workspace(row_id, version)
            if disk_content is not None:
                self._hist_put(row_id, version, disk_content)
                return ResourceFile(path=path, content=disk_content, version=version)
        return None

    async def get_by_name(self, namespace: str, name: str) -> "list[ResourceFile]":
        await self._ensure_fresh()
        prefix, suffix = f"/{namespace}/", f"/{name}"
        if self._initialized:
            results: "list[ResourceFile]" = []
            for path, (row_id, version) in self._index.items():
                if path.startswith(prefix) and path.endswith(suffix):
                    content = self._content_get(path)
                    if content is None:
                        content = self._read_workspace(row_id, version) or ""
                    self._content_put(path, content, resident=True)
                    results.append(ResourceFile(path=path, content=content, version=version))
            return results
        rows = await self._raw_get_by_name(namespace, name)
        results = []
        for row in rows:
            self._index[row.path] = (row.row_id, row.version)
            self._write_workspace(row.row_id, row.version, row.content)
            self._content_put(row.path, row.content, resident=True)
            results.append(ResourceFile(path=row.path, content=row.content, version=row.version))
        return results

    async def propfind(self, path: str) -> "list[ResourceFile]":
        await self._ensure_fresh()
        results: "list[ResourceFile]" = []
        for p, (row_id, version) in self._index.items():
            if p.startswith(path):
                content = self._content_get(p) or self._read_workspace(row_id, version) or ""
                results.append(ResourceFile(path=p, content=content, version=version))
        return results

    async def put(self, path: str, content: str, *, updated_by: str = "engine") -> ResourceFile:
        async with self.write_lock(path):
            checksum = _checksum(content)
            row_id, version, changed = await self._raw_upsert(path, content, checksum, updated_by)
            self._index[path] = (row_id, version)
            self._deleted.discard(path)
            self._content_put(path, content, resident=path in self._resident)
            self._write_workspace(row_id, version, content)
            if changed:
                await self._bump_revision()
            return ResourceFile(path=path, content=content, version=version)

    async def delete(self, path: str, *, updated_by: str = "engine") -> bool:
        async with self.write_lock(path):
            changed = await self._raw_delete(path, updated_by)
            if changed:
                self._content_pop(path)
                self._index.pop(path, None)
                self._deleted.add(path)
                await self._bump_revision()
            return changed

    async def move(self, src_path: str, dst_path: str, *, updated_by: str = "engine") -> "ResourceFile | None":
        async with self.write_lock(src_path):
            result = await self._raw_move(src_path, dst_path, updated_by)
            if result is None:
                return None
            row_id, new_version = result
            content = self._content_get(src_path)
            self._content_pop(src_path)
            self._index.pop(src_path, None)
            self._deleted.add(src_path)
            self._index[dst_path] = (row_id, new_version)
            if content is not None:
                self._content_put(dst_path, content, resident=False)
                self._write_workspace(row_id, new_version, content)
                await self._bump_revision()
                return ResourceFile(path=dst_path, content=content, version=new_version)
            fetched = await self.get(dst_path)
            await self._bump_revision()
            return fetched

    async def list_since(self, since: "datetime | None") -> "list[ResourceFile]":
        rows = await self._raw_list_since(since)
        return [ResourceFile(path=r.path, content=r.content, version=r.version) for r in rows if r.status == "active"]

    async def apply_batch(self, ops: "list[Operation]", *, updated_by: str = "engine") -> "list[ResourceFile]":
        expected_revision = str(await self.get_revision())
        _, rows = await self._raw_apply_batch(ops, expected_revision=expected_revision, updated_by=updated_by)
        for row in rows:
            self._apply_row(row)
        await self._bump_revision()
        return [ResourceFile(path=r.path, content=r.content, version=r.version) for r in rows if r.status == "active"]
