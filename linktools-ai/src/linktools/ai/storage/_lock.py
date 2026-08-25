#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process and shared-filesystem coordination primitives."""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from filelock import FileLock, Timeout
from linktools.core import environ

from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.storage.lock")
_DETACHED_LOCK_TASKS: set[asyncio.Task[object]] = set()


class KeyedAsyncLock:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._owners: dict[str, asyncio.Task[object]] = {}
        self._guard = asyncio.Lock()

    async def acquire(self, key: str) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("keyed lock requires an asyncio task")
        async with self._guard:
            if self._owners.get(key) is task:
                raise RuntimeError(f"recursive keyed lock acquisition: {key}")
            lock = self._locks.setdefault(key, asyncio.Lock())
        await lock.acquire()
        self._owners[key] = cast("asyncio.Task[object]", task)

    async def release(self, key: str) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("keyed lock release requires an asyncio task")
        await self._release_owned(key, cast("asyncio.Task[object]", task))

    async def _release_owned(
        self,
        key: str,
        owner: "asyncio.Task[object]",
    ) -> None:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None or not lock.locked():
                return
            if self._owners.get(key) is not owner:
                raise RuntimeError(f"keyed lock owner mismatch: {key}")
            self._owners.pop(key, None)
            lock.release()

    async def _release_if_owned(
        self,
        key: str,
        owner: "asyncio.Task[object]",
    ) -> bool:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None or not lock.locked():
                return False
            if self._owners.get(key) is not owner:
                return False
            self._owners.pop(key, None)
            lock.release()
            return True


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

    def __init__(self, root: "str | Path", *, lease_seconds: float = 30.0) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.root = Path(root).expanduser().resolve()
        self.lease_seconds = lease_seconds
        self._locks = KeyedAsyncLock()
        self._active: dict[str, Lease] = {}
        self._lock_owners: dict[str, "asyncio.Task[object]"] = {}

    async def acquire(self, key: str, *, timeout: float = 30.0) -> Lease:
        if not key:
            raise ValueError("lease key must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        owner = asyncio.current_task()
        if owner is None:
            raise RuntimeError("filesystem lease requires an asyncio task")
        owner = cast("asyncio.Task[object]", owner)
        lock_acquired = False
        attempt_cleanup_detached = False
        try:
            await self._locks.acquire(key)
            lock_acquired = True
            deadline = time.monotonic() + timeout
            while True:
                attempt_task = asyncio.create_task(
                    asyncio.to_thread(self._try_acquire, key)
                )
                try:
                    lease = await asyncio.shield(attempt_task)
                except asyncio.CancelledError:
                    attempt_cleanup_detached = True
                    _detach_lock_task(
                        asyncio.create_task(
                            self._finish_cancelled_attempt(
                                attempt_task,
                                key,
                                owner,
                            )
                        ),
                        "filesystem lease acquire cleanup",
                    )
                    raise
                if lease is not None:
                    self._active[key] = lease
                    self._lock_owners[key] = owner
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
            if not attempt_cleanup_detached:
                if lock_acquired:
                    await self._locks._release_owned(key, owner)
                else:
                    _detach_lock_task(
                        asyncio.create_task(
                            self._finish_cancelled_lock(key, owner)
                        ),
                        "filesystem lease lock cleanup",
                    )
            raise

    async def _finish_cancelled_lock(
        self,
        key: str,
        owner: "asyncio.Task[object]",
    ) -> None:
        await self._locks._release_if_owned(key, owner)

    async def _finish_cancelled_attempt(
        self,
        attempt_task: "asyncio.Task[Lease | None]",
        key: str,
        owner: "asyncio.Task[object]",
    ) -> None:
        lease: Lease | None = None
        try:
            lease = await attempt_task
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception(
                "filesystem lease acquisition failed: key=%s",
                key,
            )
        if lease is not None:
            try:
                await asyncio.to_thread(self._release, lease)
            except BaseException:  # noqa: BLE001
                _logger.exception(
                    "filesystem lease acquire cleanup failed: key=%s",
                    key,
                )
        await self._locks._release_owned(key, owner)

    async def renew(self, lease: Lease) -> Lease:
        if self._active.get(lease.key) != lease:
            return lease
        renewed = await asyncio.to_thread(self._renew, lease)
        if not renewed:
            self._active.pop(lease.key, None)
        return lease

    async def _finish_cancelled_release(
        self,
        release_task: "asyncio.Task[None]",
        key: str,
        owner: "asyncio.Task[object]",
    ) -> None:
        try:
            await release_task
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception(
                "filesystem lease release failed during cancellation cleanup: key=%s",
                key,
            )
        await self._locks._release_owned(key, owner)

    async def release(self, lease: Lease) -> None:
        if self._active.get(lease.key) != lease:
            return
        self._active.pop(lease.key, None)
        owner = self._lock_owners.pop(lease.key, None)
        if owner is None:
            raise RuntimeError(f"filesystem lease lock owner missing: {lease.key}")
        release_task = asyncio.create_task(asyncio.to_thread(self._release, lease))
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            _detach_lock_task(
                asyncio.create_task(
                    self._finish_cancelled_release(release_task, lease.key, owner)
                ),
                "filesystem lease release cleanup",
            )
            raise
        except BaseException as error:  # noqa: BLE001
            await self._locks._release_owned(lease.key, owner)
            raise error
        await self._locks._release_owned(lease.key, owner)

    def _try_acquire(self, key: str) -> "Lease | None":
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
            record = {
                "key": key,
                "fence": fence,
                "token": token,
                "expires_at": now + self.lease_seconds,
            }
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    lease_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    descriptor = None
                    json.dump(
                        record,
                        handle,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
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
            if (
                record.get("token") != lease.token
                or int(record.get("fence", -1)) != lease.fence
            ):
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
            if (
                record.get("token") == lease.token
                and int(record.get("fence", -1)) == lease.fence
            ):
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
        self._release_task: asyncio.Task[None] | None = None

    @property
    def acquired(self) -> bool:
        return self._lock is not None

    async def acquire(self) -> None:
        if self._lock is not None:
            return
        if self._release_task is not None and not self._release_task.done():
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        lock = FileLock(str(self.path), thread_local=False)
        acquire_task = asyncio.create_task(asyncio.to_thread(self._acquire, lock))
        try:
            acquired = await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            _detach_lock_task(
                asyncio.create_task(
                    _finish_cancelled_writer_acquire(acquire_task, lock)
                ),
                "writer lock acquire cleanup",
            )
            raise
        except Timeout as error:
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        if not acquired:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._lock = lock
        _logger.debug("runtime writer lock acquired: path=%s", self.path)

    async def release(self) -> None:
        task = self._release_task
        if task is None:
            lock = self._lock
            if lock is None:
                return
            task = asyncio.create_task(asyncio.to_thread(lock.release))
            self._release_task = task
            task.add_done_callback(
                lambda done, selected=lock: self._release_done(done, selected)
            )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        _logger.debug("runtime writer lock released: path=%s", self.path)

    def _release_done(
        self,
        task: "asyncio.Task[None]",
        lock: FileLock,
    ) -> None:
        succeeded = False
        try:
            task.result()
            succeeded = True
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("runtime writer lock release failed: path=%s", self.path)
        finally:
            if succeeded and self._lock is lock:
                self._lock = None
            if self._release_task is task:
                self._release_task = None

    def _acquire(self, lock: FileLock) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock.acquire(timeout=0)
        return True


_filesystem_mutation_locks = KeyedAsyncLock()


class FilesystemMutationLock:
    """Acquire a short-lived process and filesystem mutation lock."""

    def __init__(self, path: "str | Path", *, poll_interval: float = 0.01) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.path = Path(path).expanduser().resolve(strict=False)
        self.poll_interval = poll_interval
        self._lock = FileLock(str(self.path), thread_local=False)
        self._acquired = False
        self._key = str(self.path)
        self._owner_task: asyncio.Task[object] | None = None

    @property
    def acquired(self) -> bool:
        return self._acquired

    async def __aenter__(self) -> "FilesystemMutationLock":  # noqa: PYI034
        try:
            await asyncio.to_thread(
                self.path.parent.mkdir,
                parents=True,
                exist_ok=True,
            )
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        await _filesystem_mutation_locks.acquire(self._key)
        owner = asyncio.current_task()
        if owner is None:
            raise RuntimeError("filesystem mutation lock requires an asyncio task")
        self._owner_task = cast("asyncio.Task[object]", owner)
        cleanup_detached = False
        try:
            while True:
                acquire_task = asyncio.create_task(asyncio.to_thread(self._try_acquire))
                try:
                    acquired = await asyncio.shield(acquire_task)
                except asyncio.CancelledError:
                    cleanup_detached = True
                    _detach_lock_task(
                        asyncio.create_task(
                            self._finish_cancelled_acquire(acquire_task)
                        ),
                        "filesystem mutation acquire cleanup",
                    )
                    raise
                if acquired:
                    break
                await asyncio.sleep(self.poll_interval)
        except OSError as error:
            await self._release_process_lock()
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        except asyncio.CancelledError:
            if not cleanup_detached:
                _detach_lock_task(
                    asyncio.create_task(self._release_process_lock()),
                    "filesystem mutation process lock cleanup",
                )
            raise
        except BaseException:
            if self._acquired:
                await asyncio.to_thread(self._release)
                self._acquired = False
            await self._release_process_lock()
            raise
        _logger.debug("filesystem mutation lock acquired: path=%s", self.path)
        return self

    async def _finish_cancelled_acquire(
        self,
        acquire_task: "asyncio.Task[bool]",
    ) -> None:
        acquired = False
        try:
            acquired = await acquire_task
        except BaseException:  # noqa: BLE001
            _logger.exception(
                "filesystem mutation acquire cleanup failed: path=%s",
                self.path,
            )
        if acquired or self._acquired:
            try:
                await asyncio.to_thread(self._release)
                self._acquired = False
            except BaseException:  # noqa: BLE001
                _logger.exception(
                    "filesystem mutation late lock release failed: path=%s",
                    self.path,
                )
                return
        await self._release_process_lock()

    @asynccontextmanager
    async def local(self) -> AsyncIterator[None]:
        await _filesystem_mutation_locks.acquire(self._key)
        try:
            yield
        finally:
            await _filesystem_mutation_locks.release(self._key)

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        if not self._acquired:
            return
        release_task = asyncio.create_task(asyncio.to_thread(self._release))
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            _detach_lock_task(
                asyncio.create_task(
                    self._finish_cancelled_release(release_task)
                ),
                "filesystem mutation release cleanup",
            )
            raise
        self._acquired = False
        await self._release_process_lock()
        _logger.debug("filesystem mutation lock released: path=%s", self.path)

    async def _finish_cancelled_release(
        self,
        release_task: "asyncio.Task[None]",
    ) -> None:
        try:
            await release_task
        except BaseException:  # noqa: BLE001
            _logger.exception(
                "filesystem mutation lock release failed: path=%s",
                self.path,
            )
            return
        self._acquired = False
        await self._release_process_lock()
        _logger.debug("filesystem mutation lock released: path=%s", self.path)

    async def _release_process_lock(self) -> None:
        owner = self._owner_task
        if owner is None:
            return
        await _filesystem_mutation_locks._release_owned(self._key, owner)
        self._owner_task = None

    def _try_acquire(self) -> bool:
        try:
            self._lock.acquire(timeout=0)
        except Timeout:
            return False
        self._acquired = True
        return True

    def _release(self) -> None:
        self._lock.release()


def _read_record(path: Path) -> "dict[str, str | int | float]":
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid lease record")  # noqa: TRY004
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
    with FileLock(str(path) + ".lock"), path.open("a+", encoding="utf-8") as handle:
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


def _write_record(
    path: Path,
    record: "dict[str, str | int | float]",
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def _finish_cancelled_writer_acquire(
    acquire_task: "asyncio.Task[bool]",
    lock: FileLock,
) -> None:
    try:
        acquired = await acquire_task
    except BaseException:  # noqa: BLE001
        return
    if not acquired:
        return
    try:
        await asyncio.to_thread(lock.release)
    except BaseException:  # noqa: BLE001
        _logger.exception("cancelled writer acquire cleanup failed")


def _detach_lock_task(task: "asyncio.Task[object]", label: str) -> None:
    _DETACHED_LOCK_TASKS.add(task)

    def consume(done: "asyncio.Task[object]") -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("detached %s failed", label)
        finally:
            _DETACHED_LOCK_TASKS.discard(done)

    task.add_done_callback(consume)


__all__ = [
    "FilesystemLeaseCoordinator",
    "FilesystemMutationLock",
    "FilesystemWriterLock",
    "KeyedAsyncLock",
    "Lease",
    "ProcessLeaseCoordinator",
]
