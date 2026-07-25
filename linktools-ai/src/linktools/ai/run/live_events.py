#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live event fan-out (text/tool/paused dicts) for a streaming Run, kept
strictly separate from durable EventStore persistence. AgentEngine depends
only on the narrow ``RunLiveEventSink``/``SecurityEventSink`` Protocols below
-- never on a RunLiveEventHub or an EventStore directly -- so the pure
execution loop has no Store dependency of any kind.

``RunLiveEventHub`` adds bounded-queue backpressure (no silent drop), a
refusal on double-open for the same run_id, and identity-based close so a
stale handle can never tear down a newer one for the same run."""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..errors import RunLiveStreamAlreadyOpenError

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

RunLiveEvent = dict
SecurityEvent = Any

_CLOSED = object()


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
    run_id from the owning hub."""

    def __init__(self, run_id: str, stream_id: str, *, hub: "RunLiveEventHub", queue: "asyncio.Queue") -> None:
        self.run_id = run_id
        self.stream_id = stream_id
        self._hub = hub
        self._queue = queue
        self._closed = False

    async def publish(self, event: "RunLiveEvent") -> None:
        # Bounded queue: a full buffer backpressures the publisher (awaits
        # until a consumer drains) instead of silently dropping the event.
        await self._queue.put(event)

    async def events(self) -> "AsyncIterator[RunLiveEvent]":
        while True:
            event = await self._queue.get()
            if event is _CLOSED:
                return
            yield event

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(_CLOSED)
        except asyncio.QueueFull:
            # A consumer is behind on a full buffer -- wait for it to drain
            # rather than dropping the close signal (still not a silent drop).
            await self._queue.put(_CLOSED)
        await self._hub._forget(self)


class RunLiveEventHub:
    """Owns at most one active :class:`RunLiveStreamHandle` per ``run_id``."""

    def __init__(self) -> None:
        self._active: "dict[str, RunLiveStreamHandle]" = {}

    async def open(self, run_id: str, *, capacity: int = 256) -> RunLiveStreamHandle:
        if run_id in self._active:
            raise RunLiveStreamAlreadyOpenError(
                f"a live stream is already open for run {run_id}"
            )
        handle = RunLiveStreamHandle(
            run_id,
            uuid.uuid4().hex,
            hub=self,
            queue=asyncio.Queue(maxsize=capacity),
        )
        self._active[run_id] = handle
        return handle

    async def _forget(self, handle: "RunLiveStreamHandle") -> None:
        if self._active.get(handle.run_id) is handle:
            del self._active[handle.run_id]

    @property
    def active_stream_count(self) -> int:
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
]
