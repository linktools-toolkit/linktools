#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-owned, source-confirmed Trace payload buffering."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol
from uuid import uuid4

from linktools.core import environ

from ..core import DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL
from ..storage import WriteBufferBackend, WriteBufferEntry
from ..errors import AIError, ErrorCode
from .durability import await_owned_task
from ._projection_payload import decode_projection_payload
from ._step_archive import CapturedExecutionProjection, _AgentRunHistoryLock, _ProjectionOffset
from ._step_contracts import AgentRunStore

_logger = environ.get_logger("ai.runtime.state.trace")
_BUFFER_TIMEOUT_SECONDS = 5.0


class _FlushProjection(Protocol):
    async def __call__(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None = None,
    ) -> None: ...


class _CaptureProjection(Protocol):
    async def __call__(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None,
    ) -> bytes | None: ...


class _PublishProjection(Protocol):
    async def __call__(self, captured: CapturedExecutionProjection, *, execution_id: str) -> None: ...


@dataclass
class _Admission:
    id: str
    digest: bytes
    admitted_at: float
    confirmed: bool = False


@dataclass
class _ObservationBuffer:
    backend: WriteBufferBackend
    key: tuple[str, str, int | None]
    id: str = field(default_factory=lambda: uuid4().hex)
    entries: list[_Admission] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    publish_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    paused: bool = False
    error: BaseException | None = None
    source_error: BaseException | None = None
    admission_failed: bool = False
    admission_rejected: bool = False
    uncertain: WriteBufferEntry | None = None
    task: asyncio.Task[object] | None = None
    ack_task: asyncio.Task[object] | None = None
    inflight: tuple[str, ...] = ()
    admitted_revision: int | None = None

    @property
    def pending(self) -> tuple[_Admission, ...]:
        return tuple(value for value in self.entries if not value.confirmed)


@dataclass
class _FailedAdmission:
    key: tuple[str, str, int | None]
    error: BaseException
    terminal_confirmed: bool = False


class TracePersistence:
    """Borrow a buffer for immutable data, never as evidence of source durability.

    Admission and publication have separate locks: producers may capture a new
    tail while a fixed prefix is publishing. Only source confirmation permits
    acknowledgement; failed cleanup never replays an already-published prefix.
    """

    def __init__(
        self, archive: AgentRunStore | None, *, history_lock: _AgentRunHistoryLock,
        ensure_ready: Callable[[], Awaitable[None]],
        require_mutable: Callable[[str], None],
        has_flight: Callable[[str], bool],
        flush_boundary: _FlushProjection,
        capture: _CaptureProjection,
        publish: _PublishProjection,
        background_tasks: set[asyncio.Task[object]],
    ) -> None:
        self.archive = archive
        self.offsets: dict[str, _ProjectionOffset] = {}
        self.dirty: set[str] = set()
        self._revisions: dict[str, int] = {}
        self._history_lock = history_lock
        self._ensure_ready = ensure_ready
        self._require_mutable = require_mutable
        self._has_flight = has_flight
        self._flush_boundary = flush_boundary
        self._capture = capture
        self._publish = publish
        self._background_tasks = background_tasks
        self._caches: dict[str, WriteBufferBackend | None] = {}
        self._buffers: dict[str, _ObservationBuffer] = {}
        self._cleanup: dict[str, _ObservationBuffer] = {}
        self._terminal_runs: set[str] = set()
        self._failed_runs: dict[str, _FailedAdmission] = {}
        self._boundary_locks: dict[str, asyncio.Lock] = {}
        self._timer: asyncio.TimerHandle | None = None

    def configure(
        self, agent_run_id: str, *, buffer: WriteBufferBackend | None = None,
        batch_size: int = DEFAULT_TRACE_BATCH_SIZE,
        flush_interval: float = DEFAULT_TRACE_FLUSH_INTERVAL,
        terminal: bool = False,
    ) -> None:
        if agent_run_id in self._buffers:
            raise RuntimeError("observation batching cannot change during an active run")
        if terminal and buffer is not None:
            raise ValueError("terminal observations do not use a write buffer")
        if buffer is not None and not isinstance(buffer, WriteBufferBackend):
            raise TypeError("buffer must implement append, read_batch and ack")
        self._caches[agent_run_id] = buffer
        if terminal:
            self._terminal_runs.add(agent_run_id)
        else:
            self._terminal_runs.discard(agent_run_id)
        offset = self.offsets.setdefault(agent_run_id, _ProjectionOffset())
        offset.observation_batch_size = batch_size
        offset.observation_flush_interval = flush_interval

    def reset(self) -> None:
        self._require_failed_controls_settled()
        if self._buffers or self._cleanup:
            raise RuntimeError("Trace payload buffers are not settled")
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.offsets.clear()
        self.dirty.clear()
        self._revisions.clear()
        self._caches.clear()
        self._terminal_runs.clear()
        self._failed_runs.clear()
        self._boundary_locks.clear()

    async def settle_buffers(self) -> None:
        self._require_failed_controls_settled()
        for agent_run_id in tuple(self._buffers):
            await self.stop(agent_run_id)
        for buffer in tuple(self._cleanup.values()):
            while buffer.ack_task is not None:
                await asyncio.shield(buffer.ack_task)
            await self._ack(buffer)

    def release(self, agent_run_id: str) -> None:
        failed = self._failed_runs.get(agent_run_id)
        if failed is not None and not failed.terminal_confirmed:
            raise failed.error
        if agent_run_id in self._buffers:
            raise RuntimeError("observation buffer is not quiescent")
        self._caches.pop(agent_run_id, None)
        self._revisions.pop(agent_run_id, None)
        self._terminal_runs.discard(agent_run_id)
        self._failed_runs.pop(agent_run_id, None)
        self._boundary_locks.pop(agent_run_id, None)

    def mark_dirty(self, agent_run_id: str) -> None:
        """Track semantic staging changes separately from source-confirmed offsets."""
        self._revisions[agent_run_id] = self._revisions.get(agent_run_id, 0) + 1
        self.dirty.add(agent_run_id)

    async def _owned(self, operation: Awaitable[None]) -> None:
        async def run() -> None:
            await operation
        _, error, cancelled = await await_owned_task(
            asyncio.create_task(run()), self._background_tasks,
        )
        if error is not None:
            raise error
        if cancelled:
            raise asyncio.CancelledError

    async def defer(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None,
    ) -> None:
        await self._owned(self._defer(
            agent_run_id, execution_id=execution_id, producer_generation=producer_generation,
        ))

    async def _defer(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None, closing: bool = False,
    ) -> None:
        await self._ensure_ready()
        failed = self._failed_runs.get(agent_run_id)
        if failed is not None:
            raise failed.error
        async with self._history_lock.hold(agent_run_id):
            self._require_mutable(agent_run_id)
            if agent_run_id not in self.dirty:
                return
            offset = self.offsets.setdefault(agent_run_id, _ProjectionOffset())
            offset.observation_execution_id = execution_id
            offset.observation_producer_generation = producer_generation
            if agent_run_id in self._terminal_runs:
                return
        backend = self._caches.get(agent_run_id)
        if backend is None:
            await self._flush_boundary(
                agent_run_id, execution_id=execution_id, producer_generation=producer_generation,
            )
            return
        key = (agent_run_id, execution_id, producer_generation)
        buffer = self._buffers.setdefault(agent_run_id, _ObservationBuffer(backend, key))
        async with buffer.lock:
            if buffer.paused and not closing or buffer.key != key:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if buffer.error is not None:
                raise buffer.error
            if buffer.admitted_revision == self._revisions.get(agent_run_id, 0):
                return
            content = await self._capture(
                agent_run_id, execution_id=execution_id, producer_generation=producer_generation,
            )
            if content is None:
                return
            revision = self._revisions.get(agent_run_id, 0)
            digest = hashlib.sha256(content).digest()
            if buffer.entries and buffer.entries[-1].digest == digest:
                buffer.admitted_revision = revision
                return
            entry = (uuid4().hex, key, content)
            admission = _Admission(entry[0], digest, monotonic())
            # Register the exact candidate before the first external await.
            buffer.uncertain = entry
            try:
                async with asyncio.timeout(_BUFFER_TIMEOUT_SECONDS):
                    accepted = await backend.append(buffer.id, *entry, wait=False)
            except BaseException as error:
                try:
                    async with asyncio.timeout(_BUFFER_TIMEOUT_SECONDS):
                        queued = await backend.read_batch(buffer.id, limit=len(buffer.entries) + 1)
                    accepted = self._validate_read(buffer, (*buffer.entries, admission), queued)
                except BaseException:
                    accepted = False
                if not accepted:
                    buffer.error = error
                    buffer.admission_failed = True
                    raise
            if not accepted:
                buffer.uncertain = None
                buffer.error = AIError(ErrorCode.STORAGE_UNAVAILABLE, "Trace payload buffer rejected admission")
                buffer.admission_failed = True
                buffer.admission_rejected = True
                raise buffer.error
            buffer.entries.append(admission)
            buffer.admitted_revision = revision
            buffer.uncertain = None
        self._schedule_changed()

    @staticmethod
    def _validate_read(
        buffer: _ObservationBuffer, expected: tuple[_Admission, ...] | list[_Admission],
        queued: tuple[WriteBufferEntry, ...],
    ) -> bool:
        positions = {entry.id: index for index, entry in enumerate(expected)}
        previous = -1
        pending = []
        for id, key, content in queued:
            index = positions.get(id)
            if (index is None or index <= previous or key != buffer.key or not isinstance(content, bytes)
                    or hashlib.sha256(content).digest() != expected[index].digest):
                return False
            previous = index
            if not expected[index].confirmed:
                pending.append(id)
        return pending == [entry.id for entry in expected if not entry.confirmed]

    def _schedule_changed(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        groups: dict[tuple[int, int, float], list[tuple[_ObservationBuffer, tuple[_Admission, ...]]]] = {}
        for run_id, buffer in self._buffers.items():
            if buffer.paused or buffer.error is not None:
                continue
            pending = tuple(entry for entry in buffer.pending if entry.id not in buffer.inflight)
            if not pending:
                continue
            offset = self.offsets[run_id]
            key = (id(buffer.backend), offset.observation_batch_size or DEFAULT_TRACE_BATCH_SIZE,
                   offset.observation_flush_interval or DEFAULT_TRACE_FLUSH_INTERVAL)
            groups.setdefault(key, []).append((buffer, pending))
        next_delay = None
        for (_, size, interval), members in groups.items():
            delay = min(entries[0].admitted_at for _, entries in members) + interval - monotonic()
            if sum(len(entries) for _, entries in members) >= size or delay <= 0:
                for buffer, _ in members:
                    if buffer.task is None:
                        task = asyncio.create_task(self._automatic_flush(buffer))
                        buffer.task = task
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
            else:
                next_delay = delay if next_delay is None else min(next_delay, delay)
        if next_delay is not None:
            self._timer = asyncio.get_running_loop().call_later(next_delay, self._schedule_changed)

    async def _automatic_flush(self, buffer: _ObservationBuffer) -> None:
        try:
            await self._publish_pending(buffer)
        except BaseException as error:
            buffer.source_error = error
            if buffer.error is None:
                buffer.error = error
        finally:
            buffer.task = None
            self._schedule_changed()

    async def _publish_pending(self, buffer: _ObservationBuffer) -> None:
        async with buffer.publish_lock:
            async with buffer.lock:
                if not buffer.pending:
                    return
                expected = tuple(buffer.entries)
                async with asyncio.timeout(_BUFFER_TIMEOUT_SECONDS):
                    queued = await buffer.backend.read_batch(buffer.id, limit=len(expected))
                if not self._validate_read(buffer, expected, queued):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Trace payload prefix is missing or changed")
                pending_ids = tuple(entry.id for entry in expected if not entry.confirmed)
                selected = {id: content for id, _, content in queued}
                captures = tuple(decode_projection_payload(selected[id]) for id in pending_ids)
                for capture in captures:
                    if (capture.run.agent_run_id != buffer.key[0]
                            or capture.producer_generation != buffer.key[2]):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for previous, current in zip(captures, captures[1:]):
                    self._validate_cumulative_prefix(previous, current)
                captured = captures[-1]
                buffer.inflight = pending_ids
            offset = self.offsets[buffer.key[0]]
            offset.observation_task = asyncio.current_task()
            try:
                await self._publish(captured, execution_id=buffer.key[1])
                # This synchronous confirmation cannot race an event-loop mutation;
                # do not wait behind a producer's unrelated in-flight append.
                for entry in buffer.entries:
                    if entry.id in pending_ids:
                        entry.confirmed = True
                buffer.source_error = None
                if not buffer.admission_failed:
                    buffer.error = None
                self._schedule_ack(buffer)
            finally:
                buffer.inflight = ()
                offset.observation_task = None

    @staticmethod
    def _validate_cumulative_prefix(
        previous: CapturedExecutionProjection, current: CapturedExecutionProjection,
    ) -> None:
        # Completed append-only facts cannot be replaced by re-reading a mutable
        # staging object at a later boundary. Interactions/pending response tails
        # are revisioned and deliberately excluded from this overlap check.
        ranges = (
            (previous.events, previous.base_event_offset, previous.target_event_offset,
             current.events, current.base_event_offset, current.target_event_offset),
            (previous.checkpoints, previous.base_checkpoint_offset, previous.target_checkpoint_offset,
             current.checkpoints, current.base_checkpoint_offset, current.target_checkpoint_offset),
            (previous.local_messages, 0, len(previous.local_messages),
             current.local_messages, 0, len(current.local_messages)),
        )
        if previous.run != current.run:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Trace run changed after buffer admission")
        for old, old_base, old_end, new, new_base, new_end in ranges:
            start, end = max(old_base, new_base), min(old_end, new_end)
            if (new_base < old_base or new_base > old_end or new_end < old_end
                    or tuple(old[start - old_base:end - old_base]) != tuple(new[start - new_base:end - new_base])):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Trace admitted prefix changed")

    def _schedule_ack(self, buffer: _ObservationBuffer) -> None:
        if buffer.ack_task is not None:
            return

        async def cleanup() -> None:
            succeeded = False
            try:
                await self._ack(buffer)
                succeeded = True
            except Exception:
                _logger.warning("Trace source confirmed; buffer acknowledgement pending: run=%s", buffer.key[0])
            finally:
                buffer.ack_task = None
                if succeeded and any(entry.confirmed for entry in buffer.entries):
                    self._schedule_ack(buffer)
                self._schedule_changed()

        task = asyncio.create_task(cleanup())
        buffer.ack_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _ack(self, buffer: _ObservationBuffer) -> None:
        async with buffer.lock:
            ids = tuple(entry.id for entry in buffer.entries if entry.confirmed)
        if ids:
            # Source-confirmed cleanup must not hold the producer/control lock.
            async with asyncio.timeout(_BUFFER_TIMEOUT_SECONDS):
                await buffer.backend.ack(buffer.id, ids)
        async with buffer.lock:
            buffer.entries[:] = [entry for entry in buffer.entries if entry.id not in ids]
            if not buffer.entries:
                self._cleanup.pop(buffer.id, None)

    def buffered(self, agent_run_id: str) -> bool:
        return self._caches.get(agent_run_id) is not None

    def failed_admission_settled(
        self, agent_run_id: str, *, execution_id: str | None = None,
        producer_generation: int | None = None,
    ) -> bool:
        failed = self._failed_runs.get(agent_run_id)
        return failed is not None and (
            execution_id is None or failed.key == (agent_run_id, execution_id, producer_generation)
        )

    def _require_failed_controls_settled(self) -> None:
        for failed in self._failed_runs.values():
            if not failed.terminal_confirmed:
                raise failed.error

    def confirm_failed_admission_terminal(self, agent_run_id: str, *, execution_id: str) -> None:
        failed = self._failed_runs.get(agent_run_id)
        if failed is not None:
            if failed.key[1] != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            failed.terminal_confirmed = True

    async def settle_failed_admission(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int, error: BaseException,
    ) -> bool:
        """Freeze a definitely rejected tail after publishing its admitted prefix."""
        key = (agent_run_id, execution_id, producer_generation)
        async with self._boundary_locks.setdefault(agent_run_id, asyncio.Lock()):
            failed = self._failed_runs.get(agent_run_id)
            if failed is not None:
                return failed.key == key and failed.error is error
            buffer = self._buffers.get(agent_run_id)
            if (buffer is None or buffer.key != key or buffer.error is not error
                    or not buffer.admission_rejected or buffer.uncertain is not None):
                return False
            async with buffer.lock:
                buffer.paused = True
            self._schedule_changed()
            if buffer.task is not None:
                await asyncio.shield(buffer.task)
            if self._has_flight(agent_run_id):
                if buffer.source_error is not None:
                    raise buffer.source_error
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            # A rejected packet has no buffer ownership. Earlier accepted packets
            # still require exact bytes and source confirmation before termination.
            try:
                await self._publish_pending(buffer)
            except BaseException as source_error:
                buffer.source_error = source_error
                raise
            self._failed_runs[agent_run_id] = _FailedAdmission(key, error)
            self._retire(agent_run_id, buffer)
            return True

    async def stop(
        self, agent_run_id: str, *, publish: bool = True,
        execution_id: str | None = None, producer_generation: int | None = None,
    ) -> None:
        await self._owned(self._stop(
            agent_run_id, publish=publish, execution_id=execution_id, producer_generation=producer_generation,
        ))

    async def _stop(
        self, agent_run_id: str, *, publish: bool,
        execution_id: str | None, producer_generation: int | None,
    ) -> None:
        async with self._boundary_locks.setdefault(agent_run_id, asyncio.Lock()):
            await self._stop_locked(
                agent_run_id, publish=publish, execution_id=execution_id, producer_generation=producer_generation,
            )

    async def _stop_locked(
        self, agent_run_id: str, *, publish: bool,
        execution_id: str | None, producer_generation: int | None,
    ) -> None:
        failed = self._failed_runs.get(agent_run_id)
        if failed is not None:
            if execution_id is not None and failed.key[1:] != (execution_id, producer_generation):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if publish and not failed.terminal_confirmed:
                raise failed.error
            return
        buffer = self._buffers.get(agent_run_id)
        if buffer is None:
            backend = self._caches.get(agent_run_id)
            if not publish or backend is None or execution_id is None or agent_run_id not in self.dirty:
                return
            buffer = _ObservationBuffer(backend, (agent_run_id, execution_id, producer_generation))
            self._buffers[agent_run_id] = buffer
        async with buffer.lock:
            buffer.paused = True
        self._schedule_changed()
        if buffer.task is not None:
            await asyncio.shield(buffer.task)
        if self._has_flight(agent_run_id) and buffer.source_error is not None:
            raise buffer.source_error
        if not publish:
            return
        if buffer.error is not None and (buffer.admission_failed or self._has_flight(agent_run_id)
                or isinstance(buffer.error, AIError) and buffer.error.code in (
                    ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_INTEGRITY_ERROR,
                )):
            raise buffer.error
        try:
            if buffer.error is not None:
                # A definitely-not-committed retry reuses the same admitted bytes.
                await self._publish_pending(buffer)
            # The caller's explicit boundary also freezes/admit its latest tail;
            # no direct source capture may silently include an unadmitted packet.
            await self._defer(
                agent_run_id, execution_id=buffer.key[1], producer_generation=buffer.key[2], closing=True,
            )
            await self._publish_pending(buffer)
        except BaseException as error:
            buffer.error = error
            raise
        self._retire(agent_run_id, buffer)

    def _retire(self, agent_run_id: str, buffer: _ObservationBuffer) -> None:
        if self._buffers.get(agent_run_id) is not buffer:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        del self._buffers[agent_run_id]
        if buffer.entries:
            self._cleanup[buffer.id] = buffer
            self._schedule_ack(buffer)
        self._schedule_changed()

    async def discard(self, agent_run_id: str) -> None:
        """Cleanup after the caller proved revocation or explicitly deleted the archive."""
        buffer = self._buffers.get(agent_run_id)
        if buffer is None:
            return
        if not buffer.paused or buffer.task is not None or self._has_flight(agent_run_id):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        async with buffer.lock:
            if buffer.uncertain is not None:
                id, _, content = buffer.uncertain
                buffer.entries.append(_Admission(id, hashlib.sha256(content).digest(), monotonic()))
                buffer.uncertain = None
            for entry in buffer.entries:
                entry.confirmed = True
        self._retire(agent_run_id, buffer)
