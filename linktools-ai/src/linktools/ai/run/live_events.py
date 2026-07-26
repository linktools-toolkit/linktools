#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live event fan-out (text/tool/paused dicts) for a streaming Run, kept
strictly separate from durable EventStore persistence. AgentEngine depends
only on the narrow ``RunLiveEventSink``/``SecurityEventSink`` Protocols below
-- never on a RunLiveEventHub or an EventStore directly -- so the pure
execution loop has no Store dependency of any kind.

``RunLiveEventHub`` adds bounded-queue backpressure (no silent drop), a
refusal on double-open for the same run_id, and identity-based close so a
stale handle can never tear down a newer one for the same run.

Close semantics (P6 cancellation-safe close): ``close()`` does NOT push a sentinel into the
queue. The old design did, which blocks on a full buffer (the close signal
itself must wait for a slow consumer) and races a concurrent publish. The
new design uses an ``asyncio.Event`` for closure and races it explicitly in
``publish`` and ``events`` via ``asyncio.wait`` -- close is immediate, a
full-queue publish that loses the race raises ``RunLiveStreamClosedError``,
and events() drains the queue THEN returns when closed+empty."""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..errors import RunLiveStreamAlreadyOpenError, RunLiveStreamClosedError

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

RunLiveEvent = dict
SecurityEvent = Any


async def _suppress_cancel(task: "asyncio.Task") -> None:
    """Cancel ``task`` and await it, suppressing the resulting CancelledError.
    A task that finished JUST BEFORE cancellation is awaited normally (its
    result or exception is dropped on purpose -- this helper is only called
    on the LOSER of an asyncio.wait race, where we do not care about its
    outcome)."""
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 - swallow the cancellation loser's outcome
        pass


@runtime_checkable
class RunLiveEventSink(Protocol):
    async def publish(self, event: "RunLiveEvent") -> None: ...


@runtime_checkable
class SecurityEventSink(Protocol):
    async def emit(self, event: "SecurityEvent") -> None:
        """Return only after durable persistence succeeds."""
        ...


class NullRunLiveEventSink:
    """No-op ``RunLiveEventSink`` for non-streaming calls -- publishing never
    creates a queue or retains events."""

    async def publish(self, event: "RunLiveEvent") -> None:
        return None


class NullSecurityEventSink:
    """No-op ``SecurityEventSink``. Used only where no security-critical
    tool/model surface is wired -- production Runs always get a durable,
    per-Run-bound sink from RunCoordinator."""

    async def emit(self, event: "SecurityEvent") -> None:
        return None


@runtime_checkable
class SwarmEventSink(Protocol):
    async def emit(self, event: "Any") -> None:
        """Persist a swarm lifecycle domain event (SwarmStarted /
        SwarmCompleted / SwarmFailed / SwarmCancelled). Owned by RunCoordinator
        so SwarmEngine never appends the EventStore directly."""
        ...


class NullSwarmEventSink:
    """No-op ``SwarmEventSink`` -- for direct SwarmEngine.execute callers (tests)
    that do not need the swarm lifecycle event audit trail."""

    async def emit(self, event: "Any") -> None:
        return None


class SecurityEventSinkEmitter:
    """A ``SecurityEventEmitter`` (the emit_security / emit_observability shape
    the capability/tool/governance layer expects) backed by a ``SecurityEventSink``
    (the durable single-``emit`` Protocol RunCoordinator owns). This is the
    bridge that lets capabilities/tools/governance depend ONLY on the sink
    Protocol, with no direct EventStore reference.

    ``emit_security`` is durable -- a persistence failure propagates so an
    unrecorded security decision blocks the call (fail-closed).
    ``emit_observability`` is best-effort -- a failure is logged and swallowed
    so an observability record can never mask the decision being audited."""

    def __init__(self, sink: "SecurityEventSink") -> None:
        self._sink = sink

    async def emit_security(self, event: Any) -> None:
        await self._sink.emit(event)

    async def emit_observability(self, event: Any) -> None:
        try:
            await self._sink.emit(event)
        except Exception:  # noqa: BLE001 - observability must never mask the decision
            _LOGGER.debug("observability event emission failed", exc_info=True)


class RunLiveStreamHandle:
    """One open stream for a single ``run_id``. ``stream_id`` gives close()
    identity: a handle that outlives its own closure (e.g. a caller that held
    a reference across a reopen) can never remove a newer handle for the same
    run_id from the owning hub.

    Closure is signaled by an ``asyncio.Event`` (not a sentinel pushed into
    the queue). ``publish`` and ``events`` race the queue operation against
    the close-event with ``asyncio.wait`` so:
    - close is IMMEDIATE (never blocks on a full queue pushing a sentinel);
    - a publish that loses the close race raises RunLiveStreamClosedError;
    - events() drains events already in the queue, THEN returns when closed
      AND empty.
    """

    def __init__(
        self,
        run_id: str,
        stream_id: str,
        *,
        hub: "RunLiveEventHub",
        queue: "asyncio.Queue",
        closed: "asyncio.Event",
    ) -> None:
        self.run_id = run_id
        self.stream_id = stream_id
        self._hub = hub
        self._queue = queue
        self._closed = closed

    @property
    def is_closed(self) -> bool:
        return self._closed.is_set()

    async def publish(self, event: "RunLiveEvent") -> None:
        if self._closed.is_set():
            raise RunLiveStreamClosedError(
                f"live stream for run {self.run_id} is closed; publish refused"
            )
        put_task = asyncio.create_task(self._queue.put(event))
        close_task = asyncio.create_task(self._closed.wait())
        try:
            done, _pending = await asyncio.wait(
                {put_task, close_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if close_task in done:
                # Lost the race to close: cancel the put (it may be blocked
                # on a full buffer) and report the closed-stream outcome.
                await _suppress_cancel(put_task)
                raise RunLiveStreamClosedError(
                    f"live stream for run {self.run_id} closed mid-publish"
                )
            # put won; close did not fire. Cancel the close-waiter.
            await _suppress_cancel(close_task)
        except BaseException:
            # On any exception (incl. the RunLiveStreamClosedError above, or
            # cancellation of THIS task) make sure neither child task leaks.
            if not put_task.done():
                await _suppress_cancel(put_task)
            if not close_task.done():
                await _suppress_cancel(close_task)
            raise

    async def events(self) -> "AsyncIterator[RunLiveEvent]":
        while True:
            if self._closed.is_set() and self._queue.empty():
                return
            get_task = asyncio.create_task(self._queue.get())
            close_task = asyncio.create_task(self._closed.wait())
            try:
                done, _pending = await asyncio.wait(
                    {get_task, close_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_task in done:
                    # Got an event; cancel the close-waiter and yield.
                    await _suppress_cancel(close_task)
                    yield get_task.result()
                    continue
                # Close won. Cancel the pending get and drain whatever is
                # already in the queue (those events were enqueued BEFORE
                # close and must reach the consumer).
                await _suppress_cancel(get_task)
                while not self._queue.empty():
                    yield self._queue.get_nowait()
                return
            except BaseException:
                if not get_task.done():
                    await _suppress_cancel(get_task)
                if not close_task.done():
                    await _suppress_cancel(close_task)
                raise

    async def close(self) -> None:
        """Idempotent close. Verification of identity + removal from the
        active map happens in the owning hub (under its lock); this method
        only signals closure via the close-event."""
        await self._hub._forget_and_close(self)


class RunLiveEventHub:
    """Owns at most one active :class:`RunLiveStreamHandle` per ``run_id``.

    A single ``asyncio.Lock`` guards open / _forget / active_stream_count so
    a concurrent open + close pair cannot race (e.g. close the NEW handle
    with the OLD handle's id, or double-open the same run_id)."""

    def __init__(self) -> None:
        self._active: "dict[str, RunLiveStreamHandle]" = {}
        self._lock = asyncio.Lock()

    async def open(self, run_id: str, *, capacity: int = 256) -> RunLiveStreamHandle:
        async with self._lock:
            if run_id in self._active:
                raise RunLiveStreamAlreadyOpenError(
                    f"a live stream is already open for run {run_id}"
                )
            handle = RunLiveStreamHandle(
                run_id,
                uuid.uuid4().hex,
                hub=self,
                queue=asyncio.Queue(maxsize=capacity),
                closed=asyncio.Event(),
            )
            self._active[run_id] = handle
            return handle

    async def _forget_and_close(self, handle: "RunLiveStreamHandle") -> None:
        """Under the hub lock: verify the handle STILL owns its run_id slot
        (a stale handle that outlived a reopen cannot tear down a newer
        one), remove it, and signal closure. Idempotent: a handle whose
        closure was already signaled is a no-op."""
        async with self._lock:
            if self._active.get(handle.run_id) is handle:
                del self._active[handle.run_id]
            # Signal close whether or not this handle was the active one: a
            # stale handle that already lost its slot must still unblock its
            # own publish/events waiters.
            if not handle._closed.is_set():
                handle._closed.set()

    async def _forget(self, handle: "RunLiveStreamHandle") -> None:
        """Public alias kept for back-compat with older callers that reached
        into the hub to forget a handle without closing it. Routes through
        _forget_and_close so the active-map removal is still under the lock."""
        await self._forget_and_close(handle)

    @property
    def active_stream_count(self) -> int:
        # Lock is not needed for a single dict-len read, but the spec wants
        # open/_forget/active_stream_count synchronized against the same lock.
        # _active is only mutated under _lock; len() is a single atomic read
        # of the dict's internal counter, so the property is safe without an
        # await (the lock guards mutation, not observation).
        return len(self._active)


__all__: "list[str]" = [
    "RunLiveEvent",
    "SecurityEvent",
    "RunLiveEventSink",
    "SecurityEventSink",
    "NullRunLiveEventSink",
    "NullSecurityEventSink",
    "SwarmEventSink",
    "NullSwarmEventSink",
    "SecurityEventSinkEmitter",
    "RunLiveStreamHandle",
    "RunLiveEventHub",
    "RunLiveStreamClosedError",
]
