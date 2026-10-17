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
from uuid import uuid4

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._contracts import StorageBatchResult, StorageChange, StorageOperation

_Key = TypeVar("_Key", bound=Hashable)
_Info = TypeVar("_Info")
_logger = environ.get_logger("ai.storage.write_cache")
_WRITE_TIMEOUT_SECONDS = 5.0
_CLOSE_DEADLINE_SECONDS = 10.0
_WARNING_INTERVAL_SECONDS = 30.0
_BATCH_SIZE = 128
_FLUSH_INTERVAL_SECONDS = 60.0
_Entry = tuple[str, Hashable, bytes | None]


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
    queue_capacity: int | None
    high_watermark: int
    write_failures: int
    last_failure_code: str | None


@dataclass(frozen=True, slots=True)
class WriteCacheFlushResult:
    """Prefix settlement and a lifetime snapshot, not a durability receipt."""

    completed: bool
    status: WriteCacheStatus


@runtime_checkable
class _WriteBuffer(Protocol):
    async def append(
        self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool,
    ) -> bool: ...

    async def read_batch(self, scope: str, *, limit: int) -> tuple[_Entry, ...]: ...

    async def ack(self, scope: str, ids: Sequence[str]) -> None: ...


class WriteCacheScope:
    """One owner schedules encoded writes held by a borrowed cache.

    The cache owns the queue index, payloads and capacity, including in-flight
    entries. Its append/read_batch/ack capability must retain an ordered,
    non-destructive prefix for this single consumer. IDs are owner-local;
    distributed claiming and crash recovery are not implied.
    """

    def __init__(
        self, cache: object,
        write_batch: Callable[[tuple[_Entry, ...]], Awaitable[None]],
    ) -> None:
        if not isinstance(cache, _WriteBuffer) or not all(
            callable(getattr(cache, name, None)) for name in ("append", "read_batch", "ack")
        ):
            raise TypeError("cache must provide append, read_batch and ack")
        if not callable(write_batch):
            raise TypeError("write_batch must be callable")
        self._cache = cache
        self._scope = uuid4().hex
        self._write_batch = write_batch
        self._batch_size = _BATCH_SIZE
        self._flush_interval = _FLUSH_INTERVAL_SECONDS
        self._accepting = True
        self._stop_requested = False
        self._accepted = 0
        self._confirmed = 0
        self._unconfirmed = 0
        self._settled = 0
        self._rejected = 0
        self._write_failures = 0
        self._high_watermark = 0
        self._last_failure_code: str | None = None
        self._last_warning_at = 0.0
        self._writer: asyncio.Task[None] | None = None
        self._resolution_event = asyncio.Event()
        self._batch_ready = asyncio.Event()
        self._flush_target = 0
        self._first_pending_at: float | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._writer_cancel_requested = False
        self._admission_lock = asyncio.Lock()
        self._admissions: set[asyncio.Task[bool]] = set()
        self._closed_admissions: set[asyncio.Task[bool]] = set()
        self._cancelled_admissions: set[asyncio.Task[bool]] = set()
        self._active_admissions = 0
        self._admissions_done = asyncio.Event()
        self._admissions_done.set()
        self._backend_failure: BaseException | None = None
        self._unknown_ids: set[str] = set()

    def status(self) -> WriteCacheStatus:
        return WriteCacheStatus(
            accepting=self._accepting, accepted=self._accepted,
            confirmed=self._confirmed, unconfirmed=self._unconfirmed,
            pending=self._accepted - self._confirmed - self._unconfirmed, rejected=self._rejected,
            queue_size=self._accepted - self._resolved, queue_capacity=None,
            high_watermark=self._high_watermark, write_failures=self._write_failures,
            last_failure_code=self._last_failure_code,
        )

    @property
    def _resolved(self) -> int:
        return self._settled

    async def try_submit(self, key: Hashable, content: bytes | None) -> bool:
        """Try admission; True is not a receipt for source persistence."""
        return await self._submit(key, content, wait=False) is not None

    async def submit(self, key: Hashable, content: bytes | None) -> str:
        """Wait for capacity, returning the accepted entry's owner-local ID."""
        result = await self._submit(key, content, wait=True)
        assert result is not None
        return result

    async def _submit(
        self, key: Hashable, content: bytes | None, *, wait: bool = True,
        on_admitted: Callable[[str], None] | None = None,
    ) -> str | None:
        hash(key)
        if content is not None and not isinstance(content, bytes):
            raise TypeError("buffer content must be immutable bytes or None")
        self._active_admissions += 1
        self._admissions_done.clear()
        try:
            async with self._admission_lock:
                if not self._accepting:
                    if wait:
                        self._raise_backend_failure()
                        raise AIError(ErrorCode.STORAGE_CLOSED)
                    self._reject("write cache closed", "CACHE_CLOSED")
                    return None
                id = uuid4().hex
                operation = asyncio.create_task(
                    self._cache.append(self._scope, id, key, content, wait=wait)
                )
                self._admissions.add(operation)
                cancelled = False
                try:
                    try:
                        await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        cancelled = operation not in self._closed_admissions
                        if not operation.done() and operation not in self._cancelled_admissions:
                            self._cancelled_admissions.add(operation)
                            operation.cancel()
                    except Exception:
                        pass
                    while not operation.done():
                        try:
                            await asyncio.shield(operation)
                        except asyncio.CancelledError:
                            cancelled = True
                        except BaseException:
                            break
                    try:
                        admitted = operation.result()
                    except asyncio.CancelledError:
                        if cancelled:
                            raise
                        raise AIError(ErrorCode.STORAGE_CLOSED) from None
                    except BaseException as error:
                        self._unknown_ids.add(id)
                        self._backend_failed(error)
                        if cancelled:
                            raise asyncio.CancelledError from None
                        raise
                    if not isinstance(admitted, bool):
                        error = TypeError("cache append must return bool")
                        self._unknown_ids.add(id)
                        self._backend_failed(error)
                        raise error
                    if admitted:
                        self._accepted += 1
                        if self._first_pending_at is None:
                            self._first_pending_at = monotonic()
                        if on_admitted is not None:
                            on_admitted(id)
                        self._high_watermark = max(
                            self._high_watermark, self._accepted - self._resolved,
                        )
                        if self._writer is None:
                            self._writer = asyncio.create_task(self._run(), name="linktools-write-cache")
                            self._writer.add_done_callback(self._writer_finished)
                        self._batch_ready.set()
                    else:
                        self._reject("write cache capacity exhausted", "QUEUE_FULL")
                    if cancelled:
                        raise asyncio.CancelledError
                    if not admitted and wait:
                        raise ValueError("cache rejected a waiting buffer admission")
                    return id if admitted else None
                finally:
                    self._admissions.discard(operation)
                    self._closed_admissions.discard(operation)
                    self._cancelled_admissions.discard(operation)
        finally:
            self._active_admissions -= 1
            if not self._active_admissions:
                self._admissions_done.set()

    def _writer_finished(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            self._backend_failed(error)
        self._resolution_event.set()

    def _stop_accepting(self) -> None:
        self._accepting = False
        for operation in tuple(self._admissions):
            if not operation.done() and operation not in self._cancelled_admissions:
                self._closed_admissions.add(operation)
                self._cancelled_admissions.add(operation)
                operation.cancel()
        self._batch_ready.set()

    def _backend_failed(self, error: BaseException) -> None:
        if self._backend_failure is None:
            self._backend_failure = error
        self._last_failure_code = "CACHE_OPERATION_FAILED"
        self._stop_requested = True
        self._stop_accepting()
        self._resolution_event.set()

    def _raise_backend_failure(self) -> None:
        if self._backend_failure is not None:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from self._backend_failure

    async def _wait_resolved(self, target: int) -> None:
        self._flush_target = max(self._flush_target, target)
        self._batch_ready.set()
        while self._resolved < target:
            self._raise_backend_failure()
            self._resolution_event.clear()
            if self._resolved >= target:
                break
            await self._resolution_event.wait()
        self._raise_backend_failure()

    async def flush(self, *, timeout_seconds: float = 5.0) -> WriteCacheFlushResult:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self._resolved >= self._accepted:
            self._raise_backend_failure()
            return WriteCacheFlushResult(True, self.status())
        self._flush_target = max(self._flush_target, self._accepted)
        self._batch_ready.set()
        try:
            await asyncio.wait_for(self._wait_resolved(self._accepted), float(timeout_seconds))
        except asyncio.TimeoutError:
            return WriteCacheFlushResult(False, self.status())
        return WriteCacheFlushResult(True, self.status())

    async def close(self) -> None:
        """Drain owned operations without closing the borrowed cache or source."""
        self._stop_accepting()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned(), name="linktools-write-cache-close")
            self._close_task.add_done_callback(self._consume_close_result)
        await asyncio.shield(self._close_task)

    def _consume_close_result(self, task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    def _request_stop(self, code: str) -> None:
        self._stop_requested = True
        self._last_failure_code = code
        writer = self._writer
        if writer is not None and not writer.done() and not self._writer_cancel_requested:
            self._writer_cancel_requested = True
            writer.cancel()

    async def _close_owned(self) -> None:
        await self._admissions_done.wait()
        writer = self._writer
        if writer is not None:
            try:
                await asyncio.wait_for(asyncio.shield(writer), _CLOSE_DEADLINE_SECONDS)
            except asyncio.TimeoutError:
                self._request_stop("CLOSE_TIMEOUT")
            except BaseException:
                pass
            while not writer.done():
                try:
                    await asyncio.shield(writer)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
        # Release only this owner's undispatched tail. A failed or unknown
        # source write remains unconfirmed even when its queue entry is freed.
        try:
            while self._accepted or self._unknown_ids:
                batch = await self._cache.read_batch(self._scope, limit=self._batch_size)
                if not batch:
                    break
                await self._ack(batch)
                unknown = self._unknown_ids.intersection(entry[0] for entry in batch)
                self._accepted += len(unknown)
                self._unconfirmed += len(unknown)
                self._unknown_ids.difference_update(unknown)
        except BaseException as error:
            self._backend_failed(error)
            self._raise_backend_failure()
        remaining = self._accepted - self._confirmed - self._unconfirmed
        self._unconfirmed += max(0, remaining)
        self._settled = self._accepted
        self._resolution_event.set()
        self._raise_backend_failure()
        if self._rejected or self._unconfirmed or self._write_failures:
            self._warn("write cache closed with unconfirmed delivery")

    async def _run(self) -> None:
        while not self._stop_requested:
            pending = self._accepted - self._resolved
            if not pending:
                self._first_pending_at = None
                if not self._accepting:
                    return
                self._batch_ready.clear()
                await self._batch_ready.wait()
                continue
            self._batch_ready.clear()
            accepted_at = monotonic() if self._first_pending_at is None else self._first_pending_at
            remaining = accepted_at + self._flush_interval - monotonic()
            if (
                pending < self._batch_size and self._accepting
                and self._resolved >= self._flush_target and remaining > 0
            ):
                try:
                    await asyncio.wait_for(self._batch_ready.wait(), remaining)
                except asyncio.TimeoutError:
                    pass
                continue
            # An append may already have published its tail while its reply is
            # still pending. Only the confirmed admission prefix may be consumed.
            try:
                batch = await self._cache.read_batch(self._scope, limit=min(pending, self._batch_size))
            except asyncio.CancelledError as error:
                if not self._stop_requested:
                    self._backend_failed(error)
                raise
            if not batch or len(batch) > min(pending, self._batch_size):
                raise ValueError("cache lost or exceeded the admitted buffer prefix")
            if self._stop_requested:
                return
            await self._write(batch)
            # Source success is never replayed because acknowledgement failed.
            await self._ack(batch)
            self._settled += len(batch)
            self._resolution_event.set()

    async def _ack(self, batch: tuple[_Entry, ...]) -> None:
        ids = tuple(entry[0] for entry in batch)
        deadline = monotonic() + _WRITE_TIMEOUT_SECONDS
        while True:
            operation = asyncio.create_task(self._cache.ack(self._scope, ids))
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                if not operation.done():
                    operation.cancel()
                while not operation.done():
                    try:
                        await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                try:
                    operation.result()
                except asyncio.CancelledError as error:
                    if not self._stop_requested:
                        self._backend_failed(error)
                    raise
                raise
            except Exception:
                if monotonic() >= deadline:
                    raise
                await asyncio.sleep(min(0.01, max(0, deadline - monotonic())))
                continue
            return

    async def _write(self, batch: tuple[_Entry, ...]) -> None:
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
    id: str = ""


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
    The injected cache owns capacity, including in-flight entries. Without its
    write capability, enqueue waits for the source instead of retaining a queue.
    The source retains its own initialization, authorization and close lifetime.
    """

    def __init__(
        self,
        store: _BufferStore[_Key, _Info],
        *,
        read_pending: bool = False,
        cache: object | None = None,
    ) -> None:
        if not isinstance(store, _BufferStore):
            raise TypeError("store must provide get and apply_batch")
        if not isinstance(read_pending, bool):
            raise TypeError("read_pending must be bool")
        methods = tuple(callable(getattr(cache, name, None)) for name in ("append", "read_batch", "ack"))
        if any(methods) and not all(methods):
            raise TypeError("cache must provide the complete append/read_batch/ack capability")
        self._store = store
        self._read_pending = read_pending
        self._scope = WriteCacheScope(cache, self._consume_batch) if all(methods) else None
        self._direct_active = 0
        self._direct_done = asyncio.Event()
        self._direct_done.set()
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
        if self._scope is None:
            self._direct_active += 1
            self._direct_done.clear()
            try:
                await self._consume_batch(((uuid4().hex, change.key, change.value),))
            finally:
                self._direct_active -= 1
                if not self._direct_active:
                    self._direct_done.set()
            return

        def admitted(id: str) -> None:
            state = self._keys.setdefault(change.key, _BufferKey())
            state.mutation += 1
            state.pending = _BufferedChange(change.key, change.value, id)

        try:
            await self._scope._submit(change.key, change.value, on_admitted=admitted)
        except AIError:
            self._raise_failure()
            raise

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
            if self._scope is not None and self._scope._backend_failure is not None:
                self._fail(AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN))
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
        if self._scope is not None:
            try:
                await self._scope._wait_resolved(self._scope._accepted)
            except Exception as error:
                self._fail(error)
                raise
        else:
            await self._direct_done.wait()
        self._check_settlement()

    async def close(self) -> None:
        """Drain admitted reads and actual sink work without closing the source."""
        self._closing = True
        if self._scope is not None:
            self._scope._stop_accepting()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_owned(), name="linktools-storage-buffer-close"
            )
            self._close_task.add_done_callback(self._consume_result)
        await asyncio.shield(self._close_task)

    async def _close_owned(self) -> None:
        try:
            if self._scope is not None:
                await self._scope.close()
            await self._direct_done.wait()
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

    async def _consume_batch(self, batch: tuple[_Entry, ...]) -> None:
        latest: dict[_Key, _BufferedChange[_Key]] = {}
        for id, key, value in batch:
            change = _BufferedChange(key, value, id)
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
                    if state.pending is not None and state.pending.id == change.id:
                        state.pending = None
                    self._release_key(change.key, state)

    def _release_key(self, key: _Key, state: _BufferKey[_Key]) -> None:
        if state.pending is None and not state.readers:
            self._keys.pop(key, None)

    def _fail(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
            if self._scope is not None:
                self._scope._stop_accepting()
            for state in self._keys.values():
                state.mutation += 1

    def _check_settlement(self) -> None:
        status = None if self._scope is None else self._scope.status()
        if status is not None and status.unconfirmed and self._failure is None:
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
    "WriteCacheScope", "WriteCacheStatus", "WriteCacheFlushResult", "StorageBuffer",
]
