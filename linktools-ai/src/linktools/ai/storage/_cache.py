#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional hot-content and buffered-write capabilities.

Hot reads and writes are best-effort; their helpers treat errors as misses.
Buffered admission and acknowledgement have explicit outcomes instead: accepted
entries remain non-evictable until their owner settles and acknowledges them.
Content keys are domain identity strings; pending keys use the source's key type.
Live turn coordination has explicit ownership outcomes and propagates errors.
"""


import asyncio
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol, TypeAlias, runtime_checkable
from uuid import uuid4

from ..errors import AIError, ErrorCode
from ._files import sync_directory
from ._lock import FilesystemMutationLock, Lease
from ._turn import TurnCoordination

ContentCacheKey: TypeAlias = str


def _content_charge(size: int) -> int:
    """Charge at least one byte so zero-length entries remain capacity-bounded."""
    return max(1, size)


@runtime_checkable
class CacheBackend(Protocol):
    """Optional hot-content capability of an injected cache.

    A cache may instead, or also, implement append/read_batch/ack for buffered
    writes, open_stream/read_stream for preview streams, and the five-method
    TurnCoordinationBackend group for live turn ownership. Consumers select
    complete groups only; no inheritance or placeholder methods are required.
    """
    async def get(self, key: ContentCacheKey) -> "bytes | None": ...

    async def put(self, key: ContentCacheKey, content: bytes) -> None: ...

    async def delete(self, key: ContentCacheKey) -> None: ...



async def read_cache(cache: "CacheBackend | None", key: ContentCacheKey) -> "bytes | None":
    if cache is None:
        return None
    try:
        return await cache.get(key)
    except Exception:  # noqa: BLE001
        return None


async def write_cache(
    cache: "CacheBackend | None",
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
    cache: "CacheBackend | None",
    keys: "tuple[ContentCacheKey, ...]",
) -> "frozenset[ContentCacheKey]":
    if cache is None or not keys:
        return frozenset()
    try:
        probe = getattr(cache, "contains_many", None)
        return frozenset() if probe is None else await probe(keys)
    except Exception:  # noqa: BLE001
        return frozenset()


@dataclass(slots=True)
class _MemoryTurn:
    scope: str
    session_key: str
    request_key: str
    request_digest: str
    execution_id: str
    owner_id: str
    lease: Lease
    expires_at: float
    cancel_requested: bool = False

    def coordination(self, *, owned: bool = False) -> TurnCoordination:
        return TurnCoordination(
            self.execution_id, self.owner_id, self.cancel_requested,
            self.lease if owned else None,
        )


def _require_turn_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_turn_duration(lease_seconds: float) -> None:
    if (
        isinstance(lease_seconds, bool)
        or not isinstance(lease_seconds, (int, float))
        or not math.isfinite(lease_seconds)
        or lease_seconds <= 0
    ):
        raise ValueError("lease_seconds must be finite and positive")


class InMemoryCacheBackend:
    """Bounded hot content and owner-scoped pending writes.

    ``max_bytes`` is a payload budget with a one-byte minimum charge per entry,
    so zero-length values cannot grow the key table without bound. ``max_size``
    remains an optional stricter item-count limit. ``contains_many`` checks all
    keys under one lock and does not touch LRU order. Pending entries share
    these same limits, including in-flight batches, and cannot be evicted.
    Hot entries yield capacity to admitted pending writes.

    Live turn coordination is local to this cache instance and event loop, not
    cross-process or cross-pod. Its metadata is separate from the byte budget:
    live ownership is never evicted for content or pending-write pressure.
    Turn metadata expires with its lease and is reclaimed on coordination calls.
    """

    def __init__(self, *, max_bytes: int, max_size: "int | None" = None) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        if max_size is not None and (
            isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 0
        ):
            raise ValueError("max_size must be a non-negative integer")
        self.max_bytes = max_bytes
        self.max_size = max_size
        self._items: OrderedDict[ContentCacheKey, bytes] = OrderedDict()
        self._size = 0
        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._pending: dict[str, OrderedDict[str, tuple[Hashable, bytes | None]]] = {}
        self._pending_size = 0
        self._pending_count = 0
        self._turns: dict[str, _MemoryTurn] = {}
        self._turn_requests: dict[tuple[str, str], str] = {}
        self._turn_executions: dict[tuple[str, str], str] = {}
        self._turn_fence = 0

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
                self._size + self._pending_size > self.max_bytes
                or (self.max_size is not None
                    and len(self._items) + self._pending_count > self.max_size)
            ) and self._items:
                _, removed = self._items.popitem(last=False)
                self._size -= _content_charge(len(removed))

    async def delete(self, key: ContentCacheKey) -> None:
        async with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._size -= _content_charge(len(previous))
                self._changed.notify_all()

    async def contains_many(
        self,
        keys: "Sequence[ContentCacheKey]",
    ) -> "frozenset[ContentCacheKey]":
        if not keys:
            return frozenset()
        async with self._lock:
            return frozenset(key for key in keys if key in self._items)


    async def append(
        self, scope: str, id: str, key: Hashable, content: "bytes | None", *, wait: bool,
    ) -> bool:
        """Admit an immutable write, retaining it until exact-ID acknowledgement.

        Capacity includes in-flight entries and shares the hot-content budget.
        False means definitely not admitted. Cancellation before admission is
        safe; after admission this method returns True without another await.
        Same scope/ID retries must carry the identical key and content.
        """
        hash(key)
        if content is not None and not isinstance(content, bytes):
            raise TypeError("buffer content must be immutable bytes or None")
        charge = _content_charge(0 if content is None else len(content))
        if charge > self.max_bytes or self.max_size == 0:
            return False
        async with self._changed:
            while True:
                entries = self._pending.get(scope)
                if entries is not None and id in entries:
                    if entries[id] != (key, content):
                        raise ValueError("buffer entry ID reused with different content or key")
                    return True
                while self._items and (
                    self._size + self._pending_size + charge > self.max_bytes
                    or (self.max_size is not None
                        and len(self._items) + self._pending_count + 1 > self.max_size)
                ):
                    _, removed = self._items.popitem(last=False)
                    self._size -= _content_charge(len(removed))
                if (
                    self._pending_size + charge <= self.max_bytes
                    and (self.max_size is None or self._pending_count < self.max_size)
                ):
                    self._pending.setdefault(scope, OrderedDict())[id] = (key, content)
                    self._pending_size += charge
                    self._pending_count += 1
                    return True
                if not wait:
                    return False
                await self._changed.wait()

    async def read_batch(
        self, scope: str, *, limit: int,
    ) -> "tuple[tuple[str, Hashable, bytes | None], ...]":
        """Read the ordered unacknowledged prefix without consuming it."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("batch limit must be a positive integer")
        async with self._lock:
            entries = self._pending.get(scope)
            if entries is None:
                return ()
            result = []
            for id, (key, content) in entries.items():
                result.append((id, key, content))
                if len(result) == limit:
                    break
            return tuple(result)

    async def ack(self, scope: str, ids: "Sequence[str]") -> None:
        """Release only the listed entries; repeating an acknowledgement is safe."""
        async with self._changed:
            entries = self._pending.get(scope)
            if entries is None:
                return
            for id in ids:
                entry = entries.pop(id, None)
                if entry is not None:
                    content = entry[1]
                    self._pending_size -= _content_charge(0 if content is None else len(content))
                    self._pending_count -= 1
            if not entries:
                self._pending.pop(scope, None)
            self._changed.notify_all()


    async def claim_turn(
        self,
        scope: str,
        *,
        session_key: str,
        request_key: str,
        request_digest: str,
        execution_id: str,
        owner_id: str,
        lease_seconds: float,
    ) -> "TurnCoordination | None":
        """Atomically acquire a live turn or return its token-free replay."""
        for name, value in (
            ("scope", scope), ("session_key", session_key),
            ("request_key", request_key), ("request_digest", request_digest),
            ("execution_id", execution_id), ("owner_id", owner_id),
        ):
            _require_turn_text(value, name)
        _require_turn_duration(lease_seconds)
        async with self._lock:
            now = monotonic()
            self._expire_turns(now)
            previous_key = self._turn_requests.get((scope, request_key))
            if previous_key is not None:
                previous = self._turns[previous_key]
                if previous.session_key != session_key or previous.request_digest != request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return previous.coordination()
            if (scope, execution_id) in self._turn_executions:
                raise AIError(ErrorCode.STORAGE_CONFLICT, "execution ID already owns a live turn")
            key = json.dumps(("turn", scope, session_key), separators=(",", ":"))
            if key in self._turns:
                return None
            self._turn_fence += 1
            lease = Lease(key, self._turn_fence, uuid4().hex)
            turn = _MemoryTurn(
                scope, session_key, request_key, request_digest, execution_id,
                owner_id, lease, now + lease_seconds,
            )
            self._turns[key] = turn
            self._turn_requests[scope, request_key] = key
            self._turn_executions[scope, execution_id] = key
            return turn.coordination(owned=True)

    async def read_turn(self, scope: str, execution_id: str) -> "TurnCoordination | None":
        """Read live execution ownership without extending or exposing its lease."""
        _require_turn_text(scope, "scope")
        _require_turn_text(execution_id, "execution_id")
        async with self._lock:
            self._expire_turns(monotonic())
            key = self._turn_executions.get((scope, execution_id))
            return None if key is None else self._turns[key].coordination()

    async def renew_turn(
        self, lease: Lease, *, lease_seconds: float,
    ) -> "TurnCoordination | None":
        """Renew only the matching live owner and observe its cancellation flag."""
        _require_turn_duration(lease_seconds)
        async with self._lock:
            now = monotonic()
            self._expire_turns(now)
            turn = self._turns.get(lease.key)
            if turn is None or turn.lease != lease:
                return None
            turn.expires_at = now + lease_seconds
            return turn.coordination(owned=True)

    async def release_turn(self, lease: Lease) -> bool:
        """Release the exact live generation without disturbing a newer owner."""
        async with self._lock:
            self._expire_turns(monotonic())
            turn = self._turns.get(lease.key)
            if turn is None or turn.lease != lease:
                return False
            self._discard_turn(turn)
            return True

    async def cancel_turn(self, scope: str, execution_id: str) -> bool:
        """Request cancellation of this execution without extending its lease."""
        _require_turn_text(scope, "scope")
        _require_turn_text(execution_id, "execution_id")
        async with self._lock:
            self._expire_turns(monotonic())
            key = self._turn_executions.get((scope, execution_id))
            if key is None:
                return False
            self._turns[key].cancel_requested = True
            return True

    def _expire_turns(self, now: float) -> None:
        for turn in tuple(self._turns.values()):
            if turn.expires_at <= now:
                self._discard_turn(turn)

    def _discard_turn(self, turn: _MemoryTurn) -> None:
        del self._turns[turn.lease.key]
        del self._turn_requests[turn.scope, turn.request_key]
        del self._turn_executions[turn.scope, turn.execution_id]


class FilesystemCacheBackend:
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
        # Tokens are replaced on writes so an unlocked read cannot change
        # a newer entry's access order.
        self._entries: OrderedDict[str, object] = OrderedDict()
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
            self._entries[name] = object()
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
        entries.update((name, object()) for name in names if name not in entries)
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
        self._entries = OrderedDict((name, object()) for name in ordered)
        self._total = sum(_content_charge(size) for size in entries.values())
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


class TieredCacheBackend:
    """L1/L2 cache composition. ``get`` promotes an L2 hit into L1; failures are
    misses. ``contains_many`` is satisfied if EITHER tier holds the key, and
    does NOT promote L2 content into L1 (existence is not an access)."""

    def __init__(self, l1: CacheBackend, l2: "CacheBackend | None" = None) -> None:
        self.l1 = l1
        self.l2 = l2

    async def get(self, key: ContentCacheKey) -> "bytes | None":
        value = await read_cache(self.l1, key)
        if value is not None:
            return value
        value = await read_cache(self.l2, key)
        if value is not None:
            await write_cache(self.l1, key, value)
        return value

    async def put(self, key: ContentCacheKey, content: bytes) -> None:
        for cache in (self.l1, self.l2):
            await write_cache(cache, key, content)

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
    "CacheBackend",
    "ContentCacheKey",
    "FilesystemCacheBackend",
    "InMemoryCacheBackend",
    "TieredCacheBackend",
    "contains_many",
    "read_cache",
    "write_cache",
]
