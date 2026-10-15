#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generic best-effort caches for immutable versioned content.

The cache key is a stable domain-identity string (e.g.
``domain:{key}:{version}:{digest}``); filesystem cache file names are its SHA-256.
All caches are best-effort: any read/write error is swallowed and treated as a
miss/failed write, never propagated as the origin's result.

``contains_many`` answers whether keys already exist WITHOUT reading content --
preload uses it to avoid reading whole files just to decide a cache miss."""


import asyncio
import hashlib
import json
import os
import re
import stat
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TypeAlias

from ._files import sync_directory
from ._lock import FilesystemMutationLock

ContentCacheKey: TypeAlias = str


def _content_charge(size: int) -> int:
    """Charge at least one byte so zero-length entries remain capacity-bounded."""
    return max(1, size)


class ContentCache(Protocol):
    async def get(self, key: ContentCacheKey) -> "bytes | None": ...

    async def put(self, key: ContentCacheKey, content: bytes) -> None: ...

    async def delete(self, key: ContentCacheKey) -> None: ...

    async def contains_many(
        self,
        keys: "Sequence[ContentCacheKey]",
    ) -> "frozenset[ContentCacheKey]": ...


async def read_cache(cache: "ContentCache | None", key: ContentCacheKey) -> "bytes | None":
    if cache is None:
        return None
    try:
        return await cache.get(key)
    except Exception:  # noqa: BLE001
        return None


async def write_cache(
    cache: "ContentCache | None",
    key: ContentCacheKey,
    content: bytes,
) -> bool:
    if cache is None:
        return False
    try:
        await cache.put(key, content)
        return True
    except Exception:  # noqa: BLE001
        return False


async def contains_many(
    cache: "ContentCache | None",
    keys: "tuple[ContentCacheKey, ...]",
) -> "frozenset[ContentCacheKey]":
    if cache is None or not keys:
        return frozenset()
    try:
        return await cache.contains_many(keys)
    except Exception:  # noqa: BLE001
        return frozenset()


class InMemoryContentCache:
    """Bounded LRU content cache.

    ``max_bytes`` is a payload budget with a one-byte minimum charge per entry,
    so zero-length values cannot grow the key table without bound. ``max_size``
    remains an optional stricter item-count limit. ``contains_many`` checks all
    keys under one lock and does not touch LRU order.
    """

    def __init__(self, *, max_bytes: int, max_size: "int | None" = None) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if max_size is not None and max_size < 0:
            raise ValueError("max_size must be non-negative")
        self.max_bytes = max_bytes
        self.max_size = max_size
        self._items: OrderedDict[ContentCacheKey, bytes] = OrderedDict()
        self._size = 0
        self._lock = asyncio.Lock()

    async def get(self, key: ContentCacheKey) -> "bytes | None":
        async with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    async def put(self, key: ContentCacheKey, content: bytes) -> None:
        charge = _content_charge(len(content))
        if charge > self.max_bytes:
            return
        async with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._size -= _content_charge(len(previous))
            self._items[key] = content
            self._size += charge
            while (
                self._size > self.max_bytes
                or (self.max_size is not None and len(self._items) > self.max_size)
            ) and self._items:
                _, removed = self._items.popitem(last=False)
                self._size -= _content_charge(len(removed))

    async def delete(self, key: ContentCacheKey) -> None:
        async with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._size -= _content_charge(len(previous))

    async def contains_many(
        self,
        keys: "Sequence[ContentCacheKey]",
    ) -> "frozenset[ContentCacheKey]":
        if not keys:
            return frozenset()
        async with self._lock:
            return frozenset(key for key in keys if key in self._items)


class FilesystemContentCache:
    """Bounded second-tier cache with unlocked content reads.

    Writers sharing a root coordinate through one filesystem mutation lock and
    a recoverable total-charge record. A successful admitted put meets that
    writer's budget; different instance budgets do not establish a fixed quota.
    Each instance calibrates on its first mutation, and interrupted mutations
    trigger recovery. Normal writes stat only the target and eviction candidates.
    Local access order is approximate across instances, not a global LRU.
    Empty entries consume one byte of budget. Only recognized regular content
    files are charged or evicted. This disposable cache does not promise
    persistence through machine power loss.
    """

    def __init__(self, root: "str | Path", *, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self._lock = asyncio.Lock()
        # Tuple values are intentionally replaced on writes. Identity lets an
        # unlocked read detect that a newer entry replaced the one it observed.
        self._entries: OrderedDict[str, tuple[int]] = OrderedDict()
        self._total = 0
        self._indexed = False

    @staticmethod
    def _name(key: ContentCacheKey) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _path(self, key: ContentCacheKey) -> Path:
        return self.root / self._name(key)

    async def get(self, key: ContentCacheKey) -> "bytes | None":
        path = self._path(key)
        async with self._lock:
            observed = self._entries.get(path.name)
        try:
            value = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            async with self._lock:
                if observed is not None and self._entries.get(path.name) is observed:
                    del self._entries[path.name]
            return None
        except OSError:
            return None
        async with self._lock:
            if observed is not None and self._entries.get(path.name) is observed:
                self._entries.move_to_end(path.name)
        return value

    async def put(self, key: ContentCacheKey, content: bytes) -> None:
        if _content_charge(len(content)) > self.max_bytes:
            return
        await self._mutate(lambda: self._put(key, content))

    def _put(self, key: ContentCacheKey, content: bytes) -> None:
        try:
            if not self._begin_mutation():
                return
            name = self._name(key)
            previous = self._file_charge(self.root / name)
            if previous is None:
                self._write_control(dirty=False)
                return
            charge = _content_charge(len(content))
            self._evict(self.max_bytes - charge + previous, exclude=name)
            self._put_sync(key, content)
            self._entries.pop(name, None)
            self._entries[name] = (charge,)
            self._total += charge - previous
            self._write_control(dirty=False)
        except OSError:
            # Dirty accounting survives failures before or after publication.
            # The next writer must reconcile actual files before changing them.
            self._indexed = False
            self._ensure_index()

    async def delete(self, key: ContentCacheKey) -> None:
        await self._mutate(lambda: self._delete_sync(key))

    def _delete_sync(self, key: ContentCacheKey) -> None:
        try:
            if not self._begin_mutation():
                return
            name = self._name(key)
            path = self.root / name
            previous = self._file_charge(path)
            if previous:
                path.unlink()
                self._total -= previous
            self._entries.pop(name, None)
            self._write_control(dirty=False)
        except OSError:
            self._indexed = False
            self._ensure_index()

    def _file_charge(self, path: Path) -> "int | None":
        try:
            info = path.lstat()
        except FileNotFoundError:
            return 0
        if not stat.S_ISREG(info.st_mode):
            return None
        return _content_charge(info.st_size)

    def _evict(self, limit: int, *, exclude: str) -> None:
        refreshed = False
        while self._total > limit:
            victim = next((name for name in self._entries if name != exclude), None)
            if victim is None:
                if refreshed:
                    raise OSError("cache accounting has no remaining eviction candidates")
                self._refresh_candidates()
                refreshed = True
                continue
            self._entries.pop(victim)
            path = self.root / victim
            charge = self._file_charge(path)
            if charge:
                path.unlink()
                self._total -= charge
            elif not refreshed:
                # One stale candidate can mean another writer replaced the
                # whole local set. Reconcile names once, not a stat per miss.
                self._refresh_candidates()
                self._entries.pop(victim, None)
                refreshed = True

    def _refresh_candidates(self) -> None:
        with os.scandir(self.root) as items:
            names = dict.fromkeys(
                item.name for item in items
                if re.fullmatch(r"[0-9a-f]{64}", item.name) is not None
            )
        entries = OrderedDict(
            (name, entry) for name, entry in self._entries.items() if name in names
        )
        entries.update((name, (0,)) for name in names if name not in entries)
        self._entries = entries

    def _begin_mutation(self) -> bool:
        temporary = self.root / ".content-cache.control.tmp"
        charge = self._file_charge(temporary)
        if charge is None:
            raise OSError("cache control temporary is not a regular file")
        if charge:
            temporary.unlink()
        control = self._read_control()
        if not self._indexed or control is None or control[1]:
            if not self._ensure_index():
                return False
        else:
            self._total = control[0]
        self._write_control(dirty=True)
        self._indexed = True
        return True

    def _read_control(self) -> "tuple[int, bool] | None":
        path = self.root / ".content-cache.control"
        charge = self._file_charge(path)
        if charge is None:
            raise OSError("cache control is not a regular file")
        if not charge:
            return None
        with path.open("rb") as stream:
            raw = stream.read(4097)
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            return None
        if (
            len(raw) > 4096
            or not isinstance(value, dict)
            or set(value) != {"total", "dirty"}
            or type(value["total"]) is not int
            or value["total"] < 0
            or type(value["dirty"]) is not bool
        ):
            return None
        return value["total"], value["dirty"]

    def _write_control(self, *, dirty: bool) -> None:
        # A fixed reserved temporary cannot accumulate if a writer dies before
        # publishing dirty. Its lifetime is protected by the root mutation lock.
        temporary = self.root / ".content-cache.control.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(json.dumps(
                    {"total": self._total, "dirty": dirty}, separators=(",", ":"),
                ).encode("utf-8"))
            os.replace(temporary, self.root / ".content-cache.control")
        finally:
            temporary.unlink(missing_ok=True)

    async def _mutate(self, operation: Callable[[], None]) -> None:
        async with self._lock, FilesystemMutationLock(self.root / ".content-cache.lock"):
            worker = asyncio.create_task(asyncio.to_thread(self._run_mutation, operation))
            cancelled = False
            try:
                while not worker.done():
                    try:
                        await asyncio.wait((worker,))
                    except asyncio.CancelledError:
                        cancelled = True
                worker.result()
            finally:
                if cancelled:
                    raise asyncio.CancelledError

    def _run_mutation(self, operation: Callable[[], None]) -> None:
        try:
            operation()
        finally:
            # Other writers can repeatedly delete our newest entries without
            # causing capacity pressure. Bound retained heat independently of
            # that shared total; names can be rediscovered when eviction needs it.
            while len(self._entries) > self.max_bytes:
                self._entries.popitem(last=False)

    async def contains_many(
        self,
        keys: "Sequence[ContentCacheKey]",
    ) -> "frozenset[ContentCacheKey]":
        if not keys:
            return frozenset()
        paths = {self._name(key): key for key in keys}

        def _existing() -> "set[str]":
            return {name for name in paths if (self.root / name).exists()}

        names = await asyncio.to_thread(_existing)
        return frozenset(paths[name] for name in names)

    def _ensure_index(self) -> bool:
        try:
            entries = dict(self._scan_index())
        except OSError:
            return False
        # Reads started before this scan must not mutate its fresh accounting.
        ordered = tuple(name for name in self._entries if name in entries)
        ordered += tuple(name for name in entries if name not in self._entries)
        self._entries = OrderedDict(
            (name, (_content_charge(entries[name]),)) for name in ordered
        )
        self._total = sum(entry[0] for entry in self._entries.values())
        return True

    def _scan_index(self) -> "tuple[tuple[str, int], ...]":
        entries: list[tuple[str, int]] = []
        with os.scandir(self.root) as items:
            for item in items:
                content_file = re.fullmatch(r"[0-9a-f]{64}", item.name) is not None
                temporary_file = re.fullmatch(
                    r"\.content-cache-[0-9a-f]{64}\.[a-z0-9_]+\.tmp", item.name,
                ) is not None
                if not (content_file or temporary_file):
                    continue
                info = item.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    continue
                if temporary_file:
                    # The root lock covers every legitimate writer's temporary
                    # file lifetime, so these can only belong to dead writers.
                    os.unlink(item.path)
                else:
                    entries.append((item.name, info.st_size))
        return tuple(entries)

    def _put_sync(self, key: ContentCacheKey, content: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        fd, temporary = tempfile.mkstemp(
            prefix=f".content-cache-{path.name}.", suffix=".tmp", dir=self.root,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_directory(self.root)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class TieredContentCache:
    """L1/L2 cache composition. ``get`` promotes an L2 hit into L1; failures are
    misses. ``contains_many`` is satisfied if EITHER tier holds the key, and
    does NOT promote L2 content into L1 (existence is not an access)."""

    def __init__(self, l1: ContentCache, l2: "ContentCache | None" = None) -> None:
        self.l1 = l1
        self.l2 = l2

    async def get(self, key: ContentCacheKey) -> "bytes | None":
        try:
            value = await self.l1.get(key)
        except Exception:  # noqa: BLE001
            value = None
        if value is not None:
            return value
        if self.l2 is None:
            return None
        try:
            value = await self.l2.get(key)
        except Exception:  # noqa: BLE001
            return None
        if value is not None:
            try:
                await self.l1.put(key, value)
            except Exception:  # noqa: BLE001, S110
                pass
        return value

    async def put(self, key: ContentCacheKey, content: bytes) -> None:
        for cache in (self.l1, self.l2):
            if cache is None:
                continue
            try:
                await cache.put(key, content)
            except Exception:  # noqa: BLE001, S112
                continue

    async def delete(self, key: ContentCacheKey) -> None:
        for cache in (self.l1, self.l2):
            if cache is None:
                continue
            try:
                await cache.delete(key)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001, S112
                continue

    async def contains_many(
        self,
        keys: "Sequence[ContentCacheKey]",
    ) -> "frozenset[ContentCacheKey]":
        if not keys:
            return frozenset()
        try:
            present = await self.l1.contains_many(keys)
        except Exception:  # noqa: BLE001
            present = frozenset()
        if self.l2 is None:
            return present
        missing = tuple(key for key in keys if key not in present)
        if missing:
            try:
                present |= await self.l2.contains_many(missing)
            except Exception:  # noqa: BLE001, S110
                pass
        return present


__all__ = [
    "ContentCache",
    "ContentCacheKey",
    "FilesystemContentCache",
    "InMemoryContentCache",
    "TieredContentCache",
    "contains_many",
    "read_cache",
    "write_cache",
]
