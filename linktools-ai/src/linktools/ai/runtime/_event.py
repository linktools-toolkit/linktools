#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable event queries and bounded process-local execution streaming."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionStatus,
    Page,
    Principal,
)
from ..errors import AIError, ErrorCode
from .service_api import ExecutionEvent, ExecutionStreamEvent
from .state._contracts import EventRepository, ExecutionRepository

_logger = environ.get_logger("ai.runtime.event")
_DEFAULT_BUFFER_BYTES = 1024 * 1024
_QUEUE_LIMIT = 256


@dataclass(frozen=True, slots=True)
class ExecutionDelta:
    execution_id: str
    delta_type: ExecutionDeltaType
    content: str
    stream_truncated: bool = False


class _LiveSubscription:
    def __init__(self, broker: "LiveExecutionEventBroker", execution_id: str, max_bytes: int) -> None:
        self._broker = broker
        self._execution_id = execution_id
        self._queue: asyncio.Queue[ExecutionDelta | None] = asyncio.Queue(_QUEUE_LIMIT)
        self._max_bytes = max_bytes
        self._queue_bytes = 0
        self._closed = False

    def __aiter__(self) -> "_LiveSubscription":
        return self

    async def __anext__(self) -> ExecutionDelta:
        value = await self._queue.get()
        if value is None:
            raise StopAsyncIteration
        self._queue_bytes -= len(value.content.encode("utf-8"))
        return value

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._broker._remove_subscription(self._execution_id, self)

    def put_nowait(self, value: ExecutionDelta) -> bool:
        if self._closed:
            return False
        value = _bounded_delta(value, self._max_bytes)
        if not value.content:
            return False
        value_size = len(value.content.encode("utf-8"))
        while self._queue.full() or self._queue_bytes + value_size > self._max_bytes:
            try:
                previous = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return False
            if previous is not None:
                self._queue_bytes -= len(previous.content.encode("utf-8"))
            value = ExecutionDelta(value.execution_id, value.delta_type, value.content, stream_truncated=True)
        self._queue.put_nowait(value)
        self._queue_bytes += value_size
        return True

    def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except asyncio.QueueEmpty:
                pass


class LiveExecutionEventBroker:
    """Deliver ephemeral deltas without making them part of persistence."""

    def __init__(self, *, max_bytes: int = _DEFAULT_BUFFER_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("live broker buffer must be positive")
        self._max_bytes = max_bytes
        self._buffers: dict[str, deque[ExecutionDelta]] = {}
        self._buffer_bytes: dict[str, int] = {}
        self._truncated: set[str] = set()
        self._last_type: dict[str, ExecutionDeltaType] = {}
        self._subscriptions: dict[str, set[_LiveSubscription]] = {}
        self._activity: dict[str, asyncio.Event] = {}

    def publish(self, delta: ExecutionDelta) -> None:
        if not delta.content:
            return
        delta = _bounded_delta(delta, self._max_bytes)
        if not delta.content:
            self._truncated.add(delta.execution_id)
            self._signal(delta.execution_id)
            return
        execution_id = delta.execution_id
        buffer = self._buffers.setdefault(execution_id, deque())
        size = len(delta.content.encode("utf-8"))
        if self._last_type.get(execution_id) is delta.delta_type and buffer:
            previous = buffer.pop()
            merged = _bounded_delta(
                ExecutionDelta(
                    execution_id,
                    delta.delta_type,
                    previous.content + delta.content,
                    previous.stream_truncated or delta.stream_truncated or execution_id in self._truncated,
                ),
                self._max_bytes,
            )
            buffer.append(merged)
            self._buffer_bytes[execution_id] += len(merged.content.encode("utf-8")) - len(
                previous.content.encode("utf-8")
            )
        else:
            buffer.append(
                ExecutionDelta(
                    execution_id,
                    delta.delta_type,
                    delta.content,
                    delta.stream_truncated or execution_id in self._truncated,
                )
            )
            self._buffer_bytes[execution_id] = self._buffer_bytes.get(execution_id, 0) + size
        self._last_type[execution_id] = delta.delta_type
        while self._buffer_bytes[execution_id] > self._max_bytes and buffer:
            removed = buffer.popleft()
            self._buffer_bytes[execution_id] -= len(removed.content.encode("utf-8"))
            self._truncated.add(execution_id)
        published = ExecutionDelta(
            delta.execution_id,
            delta.delta_type,
            delta.content,
            delta.stream_truncated or execution_id in self._truncated,
        )
        for subscription in tuple(self._subscriptions.get(execution_id, ())):
            subscription.put_nowait(published)
        self._signal(execution_id)

    def mark_boundary(self, execution_id: str) -> None:
        self._last_type.pop(execution_id, None)

    def notify_durable(self, execution_id: str) -> None:
        self._signal(execution_id)

    def subscribe(self, execution_id: str) -> _LiveSubscription:
        subscription = _LiveSubscription(self, execution_id, self._max_bytes)
        for delta in self._buffers.get(execution_id, ()):
            subscription.put_nowait(
                ExecutionDelta(
                    delta.execution_id,
                    delta.delta_type,
                    delta.content,
                    delta.stream_truncated or execution_id in self._truncated,
                )
            )
        self._subscriptions.setdefault(execution_id, set()).add(subscription)
        return subscription

    async def wait_for_activity(self, execution_id: str) -> None:
        event = self._activity.setdefault(execution_id, asyncio.Event())
        await event.wait()
        event.clear()

    def complete(self, execution_id: str) -> None:
        for subscription in tuple(self._subscriptions.pop(execution_id, ())):
            subscription.finish()
        self._buffers.pop(execution_id, None)
        self._buffer_bytes.pop(execution_id, None)
        self._truncated.discard(execution_id)
        self._last_type.pop(execution_id, None)
        self._signal(execution_id)
        self._activity.pop(execution_id, None)

    def _signal(self, execution_id: str) -> None:
        self._activity.setdefault(execution_id, asyncio.Event()).set()

    def _remove_subscription(self, execution_id: str, subscription: _LiveSubscription) -> None:
        values = self._subscriptions.get(execution_id)
        if values is not None:
            values.discard(subscription)
            if not values:
                self._subscriptions.pop(execution_id, None)


def _bounded_delta(delta: ExecutionDelta, max_bytes: int) -> ExecutionDelta:
    content = delta.content
    if len(content.encode("utf-8")) <= max_bytes:
        return delta
    parts: list[str] = []
    size = 0
    for character in content:
        character_size = len(character.encode("utf-8"))
        if size + character_size > max_bytes:
            break
        parts.append(character)
        size += character_size
    return ExecutionDelta(delta.execution_id, delta.delta_type, "".join(parts), True)


class _ExecutionWorkerFailureProbe(Protocol):
    def __call__(self, execution_id: str, *, tenant_id: str) -> AIError | None: ...


class EventApi(Protocol):
    async def list(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> "Page[ExecutionEvent]": ...

    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> "AsyncIterator[ExecutionStreamEvent]": ...


class DefaultEventService:
    """Read durable events and merge them with ephemeral live deltas."""

    def __init__(
        self,
        executions: ExecutionRepository,
        events: EventRepository,
        authorization: AuthorizationPolicy,
        worker_failure: _ExecutionWorkerFailureProbe,
        live_broker: LiveExecutionEventBroker | None = None,
    ) -> None:
        self._executions = executions
        self._events = events
        self._authorization = authorization
        self._worker_failure = worker_failure
        self._live = live_broker or LiveExecutionEventBroker()

    @property
    def live_broker(self) -> LiveExecutionEventBroker:
        return self._live

    async def list(self, execution_id: str, *, principal: Principal, after_sequence: int = 0, limit: int = 100) -> Page[ExecutionEvent]:
        header = await self._executions.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.EVENT_READ, header)
        await self._authorization.authorize(principal, AuthorizationAction.EXECUTION_READ, header)
        page = await self._events.list(
            execution_id,
            tenant_id=principal.tenant_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        return Page(
            tuple(
                ExecutionEvent(item.execution_id, item.sequence, item.event_type, item.payload)
                for item in page.items
            ),
            page.next_cursor,
        )

    async def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        await self.list(execution_id, principal=principal, after_sequence=after_sequence, limit=1)
        cursor = after_sequence
        live = self._live.subscribe(execution_id)
        live_task: asyncio.Task[ExecutionDelta] | None = None
        activity_task: asyncio.Task[None] | None = None
        try:
            while True:
                page = await self.list(execution_id, principal=principal, after_sequence=cursor, limit=200)
                if page.items:
                    for event in page.items:
                        cursor = event.sequence
                        yield ExecutionStreamEvent(
                            event.execution_id,
                            event.sequence,
                            event.event_type,
                            event.payload,
                        )
                        if event.event_type in {
                            ExecutionEventType.EXECUTION_SUCCEEDED,
                            ExecutionEventType.EXECUTION_FAILED,
                            ExecutionEventType.EXECUTION_CANCELLED,
                        }:
                            _logger.debug(
                                "event stream reached terminal: execution=%s sequence=%s",
                                execution_id,
                                event.sequence,
                            )
                            return
                    continue
                if live_task is None:
                    live_task = asyncio.create_task(live.__anext__())
                if activity_task is None:
                    activity_task = asyncio.create_task(self._live.wait_for_activity(execution_id))
                done, _ = await asyncio.wait(
                    (live_task, activity_task),
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if live_task in done:
                    try:
                        delta = live_task.result()
                    except StopAsyncIteration:
                        live_task = None
                    else:
                        live_task = None
                        yield ExecutionStreamEvent(
                            delta.execution_id,
                            None,
                            delta.delta_type,
                            {"text": delta.content, "stream_truncated": delta.stream_truncated},
                        )
                if activity_task in done:
                    activity_task = None
                execution = await self._executions.get(execution_id, tenant_id=principal.tenant_id)
                if execution is None:
                    return
                failure = self._worker_failure(execution_id, tenant_id=principal.tenant_id)
                if failure is not None:
                    raise failure
                if execution.status in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    terminal = await self.list(
                        execution_id,
                        principal=principal,
                        after_sequence=cursor,
                        limit=1,
                    )
                    if not terminal.items:
                        return
        finally:
            if live_task is not None:
                live_task.cancel()
            if activity_task is not None:
                activity_task.cancel()
            await asyncio.gather(
                *(task for task in (live_task, activity_task) if task is not None),
                return_exceptions=True,
            )
            await live.close()


__all__ = ["DefaultEventService", "EventApi", "ExecutionDelta", "LiveExecutionEventBroker"]
