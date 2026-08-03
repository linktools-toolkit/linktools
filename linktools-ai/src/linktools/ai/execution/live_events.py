#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process-local event sink protocols used by AgentEngine."""

import asyncio
from typing import Any, Protocol


class RunLiveEventSink(Protocol):
    async def publish(self, event: Any) -> None: ...


class SecurityEventSink(Protocol):
    async def emit(self, event: Any) -> None: ...


class NoopRunLiveEventSink:
    async def publish(self, event: Any) -> None:
        return None


class NoopSecurityEventSink:
    async def emit(self, event: Any) -> None:
        return None


class StreamingRunLiveSink:
    """A :class:`RunLiveEventSink` that fans live engine events into an
    :class:`asyncio.Queue`, so a consumer can stream them as they are produced
    (rather than waiting for the run to finish).

    One sink instance is wired into the Runtime for its lifetime; each run
    attaches a fresh queue (:meth:`attach`) and detaches it when the run ends
    (:meth:`detach`). ``publish`` before attach / after detach is a no-op, so
    the sink is safe to share across runs and to construct before any queue
    exists. The engine does not signal end-of-stream itself; the consumer
    detects it by racing the queue against the run task's completion."""

    def __init__(self) -> None:
        self._queue: "asyncio.Queue[Any] | None" = None

    def attach(self) -> "asyncio.Queue[Any]":
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._queue = queue
        return queue

    def detach(self) -> None:
        self._queue = None

    async def publish(self, event: Any) -> None:
        queue = self._queue
        if queue is None:
            return
        await queue.put(event)


class SecurityEventSinkEmitter:
    def __init__(self, sink: SecurityEventSink) -> None:
        self._sink = sink

    async def emit_security(self, event: Any) -> None:
        await self._sink.emit(event)

    async def emit_observability(self, event: Any) -> None:
        await self._sink.emit(event)


__all__ = [
    "NoopRunLiveEventSink",
    "NoopSecurityEventSink",
    "RunLiveEventSink",
    "SecurityEventSink",
    "SecurityEventSinkEmitter",
    "StreamingRunLiveSink",
]
