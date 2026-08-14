#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process and shared-filesystem coordination primitives."""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout
from linktools.core import environ

from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.storage.lock")


class KeyedAsyncLock:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def acquire(self, key: str) -> None:
        async with self._guard:
            lock = self._locks.setdefault(key, asyncio.Lock())
        await lock.acquire()

    async def release(self, key: str) -> None:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is not None and lock.locked():
                lock.release()


@dataclass(frozen=True, slots=True)
class Lease:
    key: str
    fence: int
    token: str


class ProcessLeaseCoordinator:
    def __init__(self) -> None:
        self._locks = KeyedAsyncLock()
        self._fences: dict[str, int] = {}
        self._active: dict[str, Lease] = {}

    async def acquire(self, key: str) -> Lease:
        if not key:
            raise ValueError("lease key must not be empty")
        await self._locks.acquire(key)
        fence = self._fences.get(key, 0) + 1
        lease = Lease(key, fence, uuid4().hex)
        self._fences[key] = fence
        self._active[key] = lease
        return lease

    async def renew(self, lease: Lease) -> Lease:
        active = self._active.get(lease.key)
        if active != lease:
            return lease
        return lease

    async def release(self, lease: Lease) -> None:
        active = self._active.get(lease.key)
        if active != lease:
            return
        self._active.pop(lease.key, None)
        await self._locks.release(lease.key)


class FilesystemLeaseCoordinator:
    """Coordinate one key across processes sharing a filesystem root."""

    def __init__(self, root: 'str | Path', *, lease_seconds: float = 30.0) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.root = Path(root).expanduser().resolve()
        self.lease_seconds = lease_seconds
        self._locks = KeyedAsyncLock()
        self._active: dict[str, Lease] = {}

    async def acquire(self, key: str, *, timeout: float = 30.0) -> Lease:
        if not key:
            raise ValueError("lease key must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        await self._locks.acquire(key)
        try:
            deadline = time.monotonic() + timeout
            while True:
                lease = await asyncio.to_thread(self._try_acquire, key)
                if lease is not None:
                    self._active[key] = lease
                    _logger.debug(
                        "file lease acquired: key=%s fence=%s",
                        key,
                        lease.fence,
                    )
                    return lease
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring lease: {key}")
                await asyncio.sleep(0.01)
        except BaseException:
            await self._locks.release(key)
            raise

    async def renew(self, lease: Lease) -> Lease:
        if self._active.get(lease.key) != lease:
            return lease
        renewed = await asyncio.to_thread(self._renew, lease)
        if not renewed:
            self._active.pop(lease.key, None)
        return lease if renewed else lease

    async def release(self, lease: Lease) -> None:
        if self._active.get(lease.key) != lease:
            return
        self._active.pop(lease.key, None)
        await asyncio.to_thread(self._release, lease)
        await self._locks.release(lease.key)

    def _try_acquire(self, key: str) -> 'Lease | None':
        self.root.mkdir(parents=True, exist_ok=True)
        name = _lease_name(key)
        lease_path = self.root / f"{name}.lease"
        with _lease_guard(self.root, key):
            now = time.time()
            try:
                current = _read_record(lease_path)
            except FileNotFoundError:
                current = None
            if current is not None and float(current.get("expires_at", 0)) > now:
                return None
            if current is not None:
                lease_path.unlink(missing_ok=True)
            fence = _next_fence(self.root / f"{name}.fence")
            token = uuid4().hex
            record = {"key": key, "fence": fence, "token": token, "expires_at": now + self.lease_seconds}
            descriptor: int | None = None
            try:
                descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    descriptor = None
                    json.dump(record, handle, sort_keys=True, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                if descriptor is not None:
                    os.close(descriptor)
                lease_path.unlink(missing_ok=True)
                raise
            return Lease(key, fence, token)

    def _renew(self, lease: Lease) -> bool:
        path = self.root / f"{_lease_name(lease.key)}.lease"
        with _lease_guard(self.root, lease.key):
            try:
                record = _read_record(path)
            except FileNotFoundError:
                return False
            if record.get("token") != lease.token or int(record.get("fence", -1)) != lease.fence:
                return False
            record["expires_at"] = time.time() + self.lease_seconds
            _write_record(path, record)
            return True

    def _release(self, lease: Lease) -> None:
        path = self.root / f"{_lease_name(lease.key)}.lease"
        with _lease_guard(self.root, lease.key):
            try:
                record = _read_record(path)
            except FileNotFoundError:
                return
            if record.get("token") == lease.token and int(record.get("fence", -1)) == lease.fence:
                path.unlink(missing_ok=True)


def _lease_name(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@contextmanager
def _lease_guard(root: Path, key: str) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    guard_path = root / f"{_lease_name(key)}.guard"
    with FileLock(str(guard_path), thread_local=False):
        yield


class FilesystemWriterLock:
    """Hold a non-blocking advisory lock for a runtime lifetime."""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self._lock: FileLock | None = None

    @property
    def acquired(self) -> bool:
        return self._lock is not None

    async def acquire(self) -> None:
        if self._lock is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.path), thread_local=False)
        try:
            await asyncio.to_thread(lock.acquire, timeout=0)
        except Timeout as error:
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        self._lock = lock
        _logger.debug("runtime writer lock acquired: path=%s", self.path)

    async def release(self) -> None:
        lock = self._lock
        if lock is None:
            return
        await asyncio.to_thread(lock.release)
        self._lock = None
        _logger.debug("runtime writer lock released: path=%s", self.path)


class FilesystemMutationLock:
    """Acquire a short-lived filesystem mutation lock without blocking cancellation."""

    def __init__(self, path: "str | Path", *, poll_interval: float = 0.01) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.path = Path(path)
        self.poll_interval = poll_interval
        self._lock = FileLock(str(self.path), thread_local=False)
        self._acquired = False

    async def __aenter__(self) -> "FilesystemMutationLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            while not self._try_acquire():
                await asyncio.sleep(self.poll_interval)
        except BaseException:
            if self._acquired:
                self._release()
                self._acquired = False
            raise
        _logger.debug("filesystem mutation lock acquired: path=%s", self.path)
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._acquired:
            return
        self._acquired = False
        self._release()
        _logger.debug("filesystem mutation lock released: path=%s", self.path)

    def _try_acquire(self) -> bool:
        try:
            self._lock.acquire(timeout=0)
        except Timeout:
            return False
        self._acquired = True
        return True

    def _release(self) -> None:
        self._lock.release()


def _read_record(path: Path) -> 'dict[str, str | int | float]':
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid lease record")
    return value


def _read_fence(path: Path) -> int:
    try:
        value = int(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0
    if value < 0:
        raise ValueError("invalid lease fence")
    return value


def _next_fence(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        with path.open("a+", encoding="utf-8") as handle:
            handle.seek(0)
            raw = handle.read().strip()
            fence = 0 if not raw else int(raw)
            if fence < 0:
                raise ValueError("invalid lease fence")
            fence += 1
            handle.seek(0)
            handle.truncate()
            handle.write(str(fence))
            handle.flush()
            os.fsync(handle.fileno())
            return fence


def _write_fence(path: Path, fence: int) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(str(fence), encoding="utf-8")
    os.replace(temporary, path)


def _write_record(path: Path, record: 'dict[str, str | int | float]') -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


__all__ = [
    "FilesystemLeaseCoordinator",
    "FilesystemMutationLock",
    "FilesystemWriterLock",
    "KeyedAsyncLock",
    "Lease",
    "ProcessLeaseCoordinator",
]
