#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Owner-local batch writes and readable final-value buffers."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Hashable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Generic, Protocol, TypeVar, runtime_checkable

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._contracts import StorageBatchResult, StorageChange, StorageOperation

_Item = TypeVar("_Item")
_Key = TypeVar("_Key", bound=Hashable)
_Info = TypeVar("_Info")
_logger = environ.get_logger("ai.storage.write_cache")
_WRITE_TIMEOUT_SECONDS = 5.0
_CLOSE_DEADLINE_SECONDS = 10.0
_WARNING_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class WriteCacheStatus:
    """Lifetime counts; unconfirmed does not imply the sink wrote nothing."""

    accepting: bool
    accepted: int
    confirmed: int
    unconfirmed: int
    pending: int
    rejected: int
    queue_size: int
    queue_capacity: int
    high_watermark: int
    write_failures: int
    last_failure_code: str | None


@dataclass(frozen=True, slots=True)
class WriteCacheFlushResult:
    """Prefix settlement and a lifetime snapshot, not a durability receipt."""

    completed: bool
    status: WriteCacheStatus


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteCache:
    """Reusable options; each open call owns an independent pending queue."""

    queue_capacity: int = 1024
    batch_size: int = 128
    flush_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        for value in (self.queue_capacity, self.batch_size):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("write cache capacities must be positive integers")
        interval = self.flush_interval_seconds
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError("flush interval must be finite and positive")

    def open(
        self,
        write_batch: Callable[[tuple[_Item, ...]], Awaitable[None]],
    ) -> WriteCacheScope[_Item]:
        if not callable(write_batch):
            raise TypeError("write_batch must be callable")
        return WriteCacheScope(self, write_batch)


class WriteCacheScope(Generic[_Item]):
    """A single writer whose queued capacity excludes its one owned batch."""

    def __init__(
        self,
        cache: WriteCache,
        write_batch: Callable[[tuple[_Item, ...]], Awaitable[None]],
    ) -> None:
        self._cache = cache
        self._write_batch = write_batch
        self._queue: asyncio.Queue[tuple[_Item, float] | None] = asyncio.Queue(
            maxsize=cache.queue_capacity
        )
        self._accepting = True
        self._stop_requested = False
        self._accepted = 0
        self._confirmed = 0
        self._unconfirmed = 0
        self._rejected = 0
        self._write_failures = 0
        self._high_watermark = 0
        self._last_failure_code: str | None = None
        self._last_warning_at = 0.0
        self._writer: asyncio.Task[None] | None = None
        self._resolution_event = asyncio.Event()
        self._batch_ready = asyncio.Event()
        self._capacity_available = asyncio.Event()
        self._capacity_available.set()
        self._flush_target = 0
        self._close_task: asyncio.Task[None] | None = None
        self._writer_cancel_requested = False
        self._sentinel_queued = False

    def status(self) -> WriteCacheStatus:
        return WriteCacheStatus(
            accepting=self._accepting,
            accepted=self._accepted,
            confirmed=self._confirmed,
            unconfirmed=self._unconfirmed,
            pending=self._accepted - self._resolved,
            rejected=self._rejected,
            queue_size=self._queue.qsize() - int(self._sentinel_queued),
            queue_capacity=self._queue.maxsize,
            high_watermark=self._high_watermark,
            write_failures=self._write_failures,
            last_failure_code=self._last_failure_code,
        )

    @property
    def _resolved(self) -> int:
        return self._confirmed + self._unconfirmed

    async def try_submit(self, item: _Item) -> bool:
        """Accept an immutable item without waiting for the bound sink."""
        if not self._accepting:
            self._reject("write cache closed", "CACHE_CLOSED")
            return False
        if self._writer is None:
            self._writer = asyncio.get_running_loop().create_task(
                self._run(), name="linktools-write-cache"
            )
        try:
            self._queue.put_nowait((item, monotonic()))
        except asyncio.QueueFull:
            self._reject("write cache queue full", "QUEUE_FULL")
            return False
        self._accepted += 1
        self._batch_ready.set()
        self._high_watermark = max(self._high_watermark, self._queue.qsize())
        return True

    async def _submit(self, item: _Item) -> None:
        while True:
            if not self._accepting:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            if not self._queue.full():
                if await self.try_submit(item):
                    return
                continue
            self._capacity_available.clear()
            await self._capacity_available.wait()

    def _stop_accepting(self) -> None:
        self._accepting = False
        self._capacity_available.set()
        self._batch_ready.set()

    async def _wait_resolved(self, target: int) -> None:
        self._flush_target = max(self._flush_target, target)
        self._batch_ready.set()
        while self._resolved < target:
            self._resolution_event.clear()
            if self._resolved >= target:
                return
            await self._resolution_event.wait()

    async def flush(self, *, timeout_seconds: float = 5.0) -> WriteCacheFlushResult:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        target = self._accepted
        if self._resolved >= target:
            return WriteCacheFlushResult(True, self.status())
        self._flush_target = max(self._flush_target, target)
        self._batch_ready.set()
        try:
            await asyncio.wait_for(self._wait_resolved(target), float(timeout_seconds))
        except asyncio.TimeoutError:
            return WriteCacheFlushResult(False, self.status())
        return WriteCacheFlushResult(True, self.status())

    async def close(self) -> None:
        """Share one teardown owner; interrupted waiters cannot abandon live I/O."""
        self._stop_accepting()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_owned(), name="linktools-write-cache-close"
            )
            self._close_task.add_done_callback(self._consume_close_result)
        await asyncio.shield(self._close_task)

    def _consume_close_result(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except BaseException:
            self._last_failure_code = "CLOSE_FAILED"
            self._warn("write cache teardown failed")

    def _request_stop(self, code: str) -> None:
        self._stop_requested = True
        if self._last_failure_code != "CLOSE_TIMEOUT":
            self._last_failure_code = code
        writer = self._writer
        if writer is not None and not writer.done() and not self._writer_cancel_requested:
            self._writer_cancel_requested = True
            writer.cancel()

    async def _close_owned(self) -> None:
        writer = self._writer
        if writer is None:
            self._log_close()
            return
        # Wake an idle writer. A full queue already provides work; the writer
        # exits at its next empty boundary once admission has closed.
        try:
            self._queue.put_nowait(None)
            self._sentinel_queued = True
        except asyncio.QueueFull:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(writer), timeout=_CLOSE_DEADLINE_SECONDS)
        except asyncio.TimeoutError:
            self._request_stop("CLOSE_TIMEOUT")
        except BaseException:
            self._request_stop("CLOSE_FAILED")
        await self._consume_cancelled_writer(writer)
        self._discard_remaining("write cache close cleanup")
        self._log_close()

    async def _run(self) -> None:
        while not self._stop_requested:
            if not self._accepting and self._queue.empty():
                return
            first = await self._queue.get()
            self._capacity_available.set()
            if first is None:
                self._sentinel_queued = False
                self._queue.task_done()
                return
            item, accepted_at = first
            batch = [item]
            try:
                stop_after_batch = False
                deadline = accepted_at + self._cache.flush_interval_seconds
                while len(batch) < self._cache.batch_size and not self._stop_requested:
                    self._batch_ready.clear()
                    try:
                        queued = self._queue.get_nowait()
                        self._capacity_available.set()
                    except asyncio.QueueEmpty:
                        remaining = deadline - monotonic()
                        if (
                            remaining <= 0
                            or not self._accepting
                            or self._resolved < self._flush_target
                        ):
                            break
                        try:
                            await asyncio.wait_for(self._batch_ready.wait(), timeout=remaining)
                        except asyncio.TimeoutError:
                            break
                        continue
                    if queued is None:
                        self._sentinel_queued = False
                        self._queue.task_done()
                        stop_after_batch = True
                        break
                    batch.append(queued[0])
                if self._stop_requested:
                    raise asyncio.CancelledError
                await self._write(tuple(batch))
            except BaseException:
                self._unconfirmed += len(batch)
                self._resolution_event.set()
                raise
            finally:
                for _ in batch:
                    self._queue.task_done()
            if stop_after_batch:
                return

    async def _write(self, batch: tuple[_Item, ...]) -> None:
        try:
            # Retain callback ownership until its actual completion, so
            # cancellation cannot hide the final result or leave live I/O.
            write = asyncio.ensure_future(self._write_batch(batch))
        except asyncio.CancelledError:
            self._write_failed(len(batch), "WRITE_CANCELLED")
            return
        except Exception:
            self._write_failed(len(batch), "WRITE_FAILED")
            return
        failure_code = "WRITE_FAILED"
        try:
            await asyncio.wait_for(asyncio.shield(write), timeout=_WRITE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            failure_code = "WRITE_TIMEOUT"
            if self._last_failure_code not in {"CLOSE_TIMEOUT", "CLOSE_FAILED"}:
                self._last_failure_code = failure_code
            write.cancel()
        except asyncio.CancelledError:
            failure_code = "WRITE_CANCELLED"
            if not write.cancelled():
                # A cancelled callback is a failed batch, not a dead writer.
                self._stop_requested = True
                self._accepting = False
                if not write.done():
                    write.cancel()
        except Exception:
            pass
        while not write.done():
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                if write.done():
                    break
                # Only the first stop request cancels the sink. Later waiter
                # cancellation cannot interrupt its cancellation cleanup.
                self._stop_requested = True
                self._accepting = False
            except BaseException:
                break
        try:
            write.result()
        except BaseException:
            self._write_failed(len(batch), failure_code)
        else:
            self._confirmed += len(batch)
            self._resolution_event.set()

    def _write_failed(self, count: int, code: str) -> None:
        self._write_failures += 1
        self._unconfirmed += count
        if self._last_failure_code not in {"CLOSE_TIMEOUT", "CLOSE_FAILED"}:
            self._last_failure_code = code
        self._resolution_event.set()
        self._warn("write cache batch not confirmed")

    def _reject(self, message: str, code: str) -> None:
        self._rejected += 1
        self._last_failure_code = code
        self._warn(message)

    def _discard_remaining(self, message: str) -> None:
        discarded = 0
        while True:
            try:
                item = self._queue.get_nowait()
                self._capacity_available.set()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            if item is not None:
                discarded += 1
            else:
                self._sentinel_queued = False
        if discarded:
            self._unconfirmed += discarded
            self._resolution_event.set()
            self._warn(message)

    def _warn(self, message: str) -> None:
        now = monotonic()
        if now - self._last_warning_at < _WARNING_INTERVAL_SECONDS:
            return
        self._last_warning_at = now
        _logger.warning(
            "%s: accepted=%s confirmed=%s rejected=%s unconfirmed=%s write_failures=%s",
            message, self._accepted, self._confirmed, self._rejected,
            self._unconfirmed, self._write_failures,
        )

    def _log_close(self) -> None:
        if self._rejected or self._unconfirmed or self._write_failures:
            self._warn("write cache closed with unconfirmed delivery")

    async def _consume_cancelled_writer(self, writer: asyncio.Task[None]) -> None:
        while not writer.done():
            try:
                await asyncio.shield(writer)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            writer.result()
        except asyncio.CancelledError:
            pass
        except BaseException:
            self._write_failures += 1
            if self._last_failure_code not in {"CLOSE_TIMEOUT", "CLOSE_FAILED"}:
                self._last_failure_code = "WRITER_FAILED"
            self._warn("write cache writer failed")


@runtime_checkable
class _BufferStore(Protocol[_Key, _Info]):
    async def get(self, key: _Key) -> bytes | None: ...

    async def apply_batch(
        self, changes: Sequence[StorageChange[_Key, bytes]],
    ) -> StorageBatchResult[_Info, _Key]: ...


@dataclass(frozen=True, slots=True)
class _BufferedChange(Generic[_Key]):
    key: _Key
    value: bytes | None


@dataclass(slots=True)
class _BufferKey(Generic[_Key]):
    mutation: int = 0
    readers: int = 0
    pending: _BufferedChange[_Key] | None = None


class StorageBuffer(Generic[_Key, _Info]):
    """Final-value write intentions over a borrowed byte store.

    Enqueue confirms admission, not persistence. Same-key intentions within a
    dispatched batch coalesce; flush confirms their final values, not a
    transactional prefix snapshot. Use one buffer for all writes to its keys.
    Capacity bounds queued entries plus one in-flight batch, not payload bytes.
    The source retains its own initialization, authorization and close lifetime.
    """

    def __init__(
        self,
        store: _BufferStore[_Key, _Info],
        *,
        read_pending: bool = False,
        write_cache: WriteCache | None = None,
    ) -> None:
        if not isinstance(store, _BufferStore):
            raise TypeError("store must provide get and apply_batch")
        if not isinstance(read_pending, bool):
            raise TypeError("read_pending must be bool")
        if write_cache is not None and not isinstance(write_cache, WriteCache):
            raise TypeError("write_cache must be WriteCache")
        self._store = store
        self._read_pending = read_pending
        self._scope = (write_cache or WriteCache()).open(self._consume_batch)
        self._keys: dict[_Key, _BufferKey[_Key]] = {}
        self._failure: BaseException | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._reads: set[asyncio.Task[bytes | None]] = set()
        self._reads_done = asyncio.Event()
        self._reads_done.set()

    async def enqueue(self, key: _Key, value: bytes) -> None:
        """Wait for entry capacity and accept immutable bytes for later writing."""
        if not isinstance(value, bytes):
            raise TypeError("buffer values must be immutable bytes")
        await self._enqueue(_BufferedChange(key, value))

    async def enqueue_delete(self, key: _Key) -> None:
        """Accept a tombstone in the same order as other final-value intentions."""
        await self._enqueue(_BufferedChange(key, None))

    async def _enqueue(self, change: _BufferedChange[_Key]) -> None:
        hash(change.key)
        self._raise_failure()
        if self._closing:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        try:
            await self._scope._submit(change)
        except AIError:
            self._raise_failure()
            raise
        state = self._keys.setdefault(change.key, _BufferKey())
        state.mutation += 1
        state.pending = change

    async def get(self, key: _Key) -> bytes | None:
        """Read the fixed view; failed buffers always use the authoritative source."""
        hash(key)
        if self._closing:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        state = self._keys.setdefault(key, _BufferKey())
        state.readers += 1
        self._reads_done.clear()
        read = asyncio.create_task(self._get_owned(key, state))
        self._reads.add(read)
        read.add_done_callback(lambda task: self._read_finished(key, state, task))
        try:
            return await asyncio.shield(read)
        except asyncio.CancelledError:
            if not read.done():
                read.cancel()
            raise

    async def _get_owned(self, key: _Key, state: _BufferKey[_Key]) -> bytes | None:
        while True:
            if self._failure is None and self._read_pending and state.pending is not None:
                return state.pending.value
            mutation = state.mutation
            value = await self._store.get(key)
            if not self._read_pending or mutation == state.mutation:
                return value

    def _read_finished(
        self, key: _Key, state: _BufferKey[_Key], task: asyncio.Task[bytes | None],
    ) -> None:
        self._consume_result(task)
        state.readers -= 1
        self._release_key(key, state)
        self._reads.discard(task)
        if not self._reads:
            self._reads_done.set()

    def _consume_result(self, task: asyncio.Task[object]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def flush(self) -> None:
        """Confirm prior intentions, including successfully persisted superseders.

        Cancellation stops only this wait. Failed or unknown source outcomes
        reject new admission and are reported instead of acknowledged as durable.
        """
        await self._scope._wait_resolved(self._scope._accepted)
        self._check_settlement()

    async def close(self) -> None:
        """Drain admitted reads and actual sink work without closing the source."""
        self._closing = True
        self._scope._stop_accepting()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_owned(), name="linktools-storage-buffer-close"
            )
            self._close_task.add_done_callback(self._consume_result)
        await asyncio.shield(self._close_task)

    async def _close_owned(self) -> None:
        try:
            await self._scope.close()
        finally:
            # Queue cleanup may settle work without calling the batch consumer.
            # Reclaim those payloads only after the writer has actually stopped.
            for key, state in tuple(self._keys.items()):
                if state.pending is not None:
                    state.pending = None
                    state.mutation += 1
                    self._release_key(key, state)
            await self._reads_done.wait()
        self._check_settlement()

    async def _consume_batch(self, batch: tuple[_BufferedChange[_Key], ...]) -> None:
        latest: dict[_Key, _BufferedChange[_Key]] = {}
        for change in batch:
            latest.pop(change.key, None)
            latest[change.key] = change
        changes = tuple(
            StorageChange(
                StorageOperation.DELETE if change.value is None else StorageOperation.PUT,
                change.key,
                change.value,
                expected_revision=None,
            )
            for change in latest.values()
        )
        try:
            self._raise_failure()
            await self._store.apply_batch(changes)
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            for change in latest.values():
                state = self._keys.get(change.key)
                if state is not None:
                    state.mutation += 1
                    if state.pending is change:
                        state.pending = None
                    self._release_key(change.key, state)

    def _release_key(self, key: _Key, state: _BufferKey[_Key]) -> None:
        if state.pending is None and not state.readers:
            self._keys.pop(key, None)

    def _fail(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
            self._scope._stop_accepting()
            for state in self._keys.values():
                state.mutation += 1

    def _check_settlement(self) -> None:
        status = self._scope.status()
        if status.unconfirmed and self._failure is None:
            self._fail(AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN))
        self._raise_failure()

    def _raise_failure(self) -> None:
        error = self._failure
        if isinstance(error, Exception):
            # A fail-stopped owner may reject indefinitely; do not retain every
            # rejected caller through an ever-growing exception traceback.
            raise error.with_traceback(None)
        if error is not None:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error


__all__ = [
    "WriteCache", "WriteCacheScope", "WriteCacheStatus", "WriteCacheFlushResult", "StorageBuffer",
]
