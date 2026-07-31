#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process-local event sink protocols used by AgentEngine."""

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
]
