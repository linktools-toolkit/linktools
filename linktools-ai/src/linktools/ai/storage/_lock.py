#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process and shared-filesystem coordination primitives."""

import asyncio
import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from uuid import uuid4

from linktools.core import environ
from ..core import ErrorCode, AIError


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
    with guard_path.open("a+", encoding="utf-8") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)


class FilesystemWriterLock:
    """Hold a non-blocking advisory lock for a runtime lifetime."""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    async def acquire(self) -> None:
        if self._descriptor is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = await asyncio.to_thread(os.open, self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            await asyncio.to_thread(fcntl.flock, descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        _logger.info("runtime writer lock acquired: path=%s", self.path)

    async def release(self) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        await asyncio.to_thread(fcntl.flock, descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        _logger.info("runtime writer lock released: path=%s", self.path)


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
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
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
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_fence(path: Path, fence: int) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(str(fence), encoding="utf-8")
    os.replace(temporary, path)


def _write_record(path: Path, record: 'dict[str, str | int | float]') -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


__all__ = ["FilesystemLeaseCoordinator", "FilesystemWriterLock", "KeyedAsyncLock", "Lease", "ProcessLeaseCoordinator"]
