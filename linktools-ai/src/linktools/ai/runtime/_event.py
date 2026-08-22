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


@dataclass(frozen=True, slots=True)
class _DurableMarker:
    execution_id: str
    sequence: int


@dataclass(slots=True)
class _PreparedStreamLease:
    execution_id: str
    state: str = "PREPARED"
    subscription: "_LiveSubscription | None" = None


_OrderedItem = ExecutionDelta | _DurableMarker


class _LiveSubscription:
    def __init__(
        self,
        broker: "LiveExecutionEventBroker",
        execution_id: str,
        max_bytes: int,
    ) -> None:
        self._broker = broker
        self._execution_id = execution_id
        self._queue: deque[_OrderedItem] = deque()
        self._max_bytes = max_bytes
        self._queue_bytes = 0
        self._queue_delta_count = 0
        self._truncated_pending = False
        self._wakeup = asyncio.Event()
        self._completed = False
        self._closed = False

    def __aiter__(self) -> "_LiveSubscription":
        return self

    async def __anext__(self) -> _OrderedItem:
        while True:
            if self._queue:
                value = self._queue.popleft()
                if isinstance(value, ExecutionDelta):
                    self._queue_bytes -= len(value.content.encode("utf-8"))
                    self._queue_delta_count -= 1
                return value
            if self._closed or self._completed:
                raise StopAsyncIteration
            self._wakeup.clear()
            if self._queue or self._completed or self._closed:
                continue
            await self._wakeup.wait()

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._broker._remove_subscription(self._execution_id, self)

    def put_nowait(self, value: ExecutionDelta) -> bool:
        if self._closed:
            return False
        value = _bounded_delta(value, self._max_bytes)
        if not value.content:
            self._truncated_pending = True
            return False
        value_size = len(value.content.encode("utf-8"))
        while self._queue_delta_count >= _QUEUE_LIMIT or self._queue_bytes + value_size > self._max_bytes:
            if not self._drop_oldest_delta():
                break
        truncated = value.stream_truncated or self._truncated_pending
        self._truncated_pending = False
        value = ExecutionDelta(value.execution_id, value.delta_type, value.content, truncated)
        self._queue.append(value)
        self._queue_bytes += value_size
        self._queue_delta_count += 1
        self._wakeup.set()
        return True

    def put_marker(self, value: _DurableMarker) -> None:
        if self._closed:
            return
        self._queue.append(value)
        self._wakeup.set()

    def finish(self) -> None:
        if self._closed:
            return
        self._completed = True
        self._wakeup.set()

    def first_marker_sequence(self, *, after_sequence: int) -> int | None:
        for value in self._queue:
            if isinstance(value, _DurableMarker) and value.sequence > after_sequence:
                return value.sequence
        return None

    def _drop_oldest_delta(self) -> bool:
        for index, value in enumerate(self._queue):
            if isinstance(value, ExecutionDelta):
                values = list(self._queue)
                removed = values.pop(index)
                self._queue = deque(values)
                self._queue_bytes -= len(removed.content.encode("utf-8"))
                self._queue_delta_count -= 1
                self._truncated_pending = True
                return True
        return False


class LiveExecutionEventBroker:
    """Deliver ephemeral deltas without making them part of persistence."""

    def __init__(self, *, max_bytes: int = _DEFAULT_BUFFER_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("live broker buffer must be positive")
        self._max_bytes = max_bytes
        self._buffers: dict[str, deque[_OrderedItem]] = {}
        self._buffer_bytes: dict[str, int] = {}
        self._truncated: set[str] = set()
        self._last_type: dict[str, ExecutionDeltaType] = {}
        self._subscriptions: dict[str, set[_LiveSubscription]] = {}
        self._activity: dict[str, asyncio.Event] = {}
        self._completed: set[str] = set()
        self._durable_sequences: dict[str, int] = {}
        self._prepared: dict[str, _PreparedStreamLease] = {}
        self._local_producers: dict[str, str] = {}

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
            if not isinstance(previous, ExecutionDelta):
                raise RuntimeError("live broker delta ordering is corrupt")
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
        while self._buffer_bytes[execution_id] > self._max_bytes:
            if not self._drop_oldest_delta(buffer, execution_id):
                break
        published = ExecutionDelta(
            delta.execution_id,
            delta.delta_type,
            delta.content,
            delta.stream_truncated or execution_id in self._truncated,
        )
        for subscription in tuple(self._subscriptions.get(execution_id, ())):
            subscription.put_nowait(published)
        self._signal(execution_id)

    def publish_durable(self, execution_id: str, sequence: int) -> None:
        if sequence < 1:
            raise ValueError("durable marker sequence must be positive")
        previous = self._durable_sequences.get(execution_id, 0)
        if sequence <= previous:
            return
        marker = _DurableMarker(execution_id, sequence)
        self._durable_sequences[execution_id] = sequence
        self._buffers.setdefault(execution_id, deque()).append(marker)
        self._buffer_bytes.setdefault(execution_id, 0)
        self._last_type.pop(execution_id, None)
        for subscription in tuple(self._subscriptions.get(execution_id, ())):
            subscription.put_marker(marker)
        _logger.debug(
            "live durable marker published: execution=%s sequence=%s",
            execution_id,
            sequence,
        )
        self._signal(execution_id)

    def subscribe(self, execution_id: str) -> _LiveSubscription:
        subscription = _LiveSubscription(self, execution_id, self._max_bytes)
        self._subscriptions.setdefault(execution_id, set()).add(subscription)
        for item in self._buffers.get(execution_id, ()):
            if isinstance(item, ExecutionDelta):
                subscription.put_nowait(
                    ExecutionDelta(
                        item.execution_id,
                        item.delta_type,
                        item.content,
                        item.stream_truncated or execution_id in self._truncated,
                    )
                )
            else:
                subscription.put_marker(item)
        if execution_id in self._completed:
            subscription.finish()
        return subscription

    def prepare_local_producer(self, execution_id: str) -> _PreparedStreamLease:
        current = self._prepared.get(execution_id)
        if current is not None and current.state == "PREPARED":
            return current
        lease = _PreparedStreamLease(execution_id)
        self._prepared[execution_id] = lease
        self._local_producers[execution_id] = "prepared"
        _logger.debug("local event stream prepared: execution=%s", execution_id)
        return lease

    def claim_local_producer(self, lease: _PreparedStreamLease) -> _LiveSubscription:
        current = self._prepared.get(lease.execution_id)
        if current is not lease or lease.state != "PREPARED":
            raise RuntimeError("prepared event stream lease is not claimable")
        subscription = _LiveSubscription(self, lease.execution_id, self._max_bytes)
        lease.state = "CLAIMED"
        lease.subscription = subscription
        self._prepared.pop(lease.execution_id, None)
        self._local_producers[lease.execution_id] = "active"
        self._subscriptions.setdefault(lease.execution_id, set()).add(subscription)
        self._fill_subscription(subscription, lease.execution_id)
        if lease.execution_id in self._completed:
            subscription.finish()
        return subscription

    def abort_local_producer(self, lease: _PreparedStreamLease, *, worker_installed: bool) -> None:
        if lease.state != "PREPARED":
            return
        current = self._prepared.get(lease.execution_id)
        if current is not lease:
            lease.state = "ABANDONED"
            return
        lease.state = "ABANDONED"
        self._prepared.pop(lease.execution_id, None)
        if worker_installed:
            if self._local_producers.get(lease.execution_id) == "prepared":
                self._local_producers[lease.execution_id] = "active"
        else:
            self._release_execution(lease.execution_id)
        self._signal(lease.execution_id)
        _logger.debug(
            "local event stream lease abandoned: execution=%s worker_installed=%s",
            lease.execution_id,
            worker_installed,
        )

    def is_local_producer(self, execution_id: str) -> bool:
        return execution_id in self._local_producers

    async def wait_for_activity(self, execution_id: str) -> None:
        event = self._activity.setdefault(execution_id, asyncio.Event())
        await event.wait()
        event.clear()

    def complete(self, execution_id: str) -> None:
        self._completed.add(execution_id)
        if execution_id in self._local_producers:
            self._local_producers[execution_id] = "completed"
        prepared = self._prepared.get(execution_id)
        if prepared is not None and prepared.subscription is not None:
            prepared.subscription.finish()
        subscriptions = tuple(self._subscriptions.get(execution_id, ()))
        for subscription in subscriptions:
            subscription.finish()
        self._signal(execution_id)
        if not subscriptions and prepared is None:
            self._release_execution(execution_id)

    def _signal(self, execution_id: str) -> None:
        self._activity.setdefault(execution_id, asyncio.Event()).set()

    def _remove_subscription(self, execution_id: str, subscription: _LiveSubscription) -> None:
        values = self._subscriptions.get(execution_id)
        if values is not None:
            values.discard(subscription)
            if not values:
                self._subscriptions.pop(execution_id, None)
                if execution_id in self._completed:
                    self._release_execution(execution_id)

    def _fill_subscription(self, subscription: _LiveSubscription, execution_id: str) -> None:
        for item in self._buffers.get(execution_id, ()):
            if isinstance(item, ExecutionDelta):
                subscription.put_nowait(
                    ExecutionDelta(
                        item.execution_id,
                        item.delta_type,
                        item.content,
                        item.stream_truncated or execution_id in self._truncated,
                    )
                )
            else:
                subscription.put_marker(item)

    def _drop_oldest_delta(self, buffer: deque[_OrderedItem], execution_id: str) -> bool:
        for index, value in enumerate(buffer):
            if isinstance(value, ExecutionDelta):
                values = list(buffer)
                removed = values.pop(index)
                buffer.clear()
                buffer.extend(values)
                self._buffer_bytes[execution_id] -= len(removed.content.encode("utf-8"))
                self._truncated.add(execution_id)
                return True
        return False

    def _release_execution(self, execution_id: str) -> None:
        self._buffers.pop(execution_id, None)
        self._buffer_bytes.pop(execution_id, None)
        self._truncated.discard(execution_id)
        self._last_type.pop(execution_id, None)
        self._completed.discard(execution_id)
        self._durable_sequences.pop(execution_id, None)
        self._activity.pop(execution_id, None)
        self._prepared.pop(execution_id, None)
        self._local_producers.pop(execution_id, None)


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

    async def _prepare_local_stream(self, execution_id: str) -> _PreparedStreamLease:
        return self._live.prepare_local_producer(execution_id)

    async def _claim_local_stream(self, lease: _PreparedStreamLease) -> _LiveSubscription:
        return self._live.claim_local_producer(lease)

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
        async for event in self._stream_with_live(
            execution_id,
            principal=principal,
            after_sequence=after_sequence,
        ):
            yield event

    async def _stream_prepared(
        self,
        lease: _PreparedStreamLease,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        await self._authorize_stream(lease.execution_id, principal)
        live = self._live.claim_local_producer(lease)
        async for event in self._stream_with_live(
            lease.execution_id,
            principal=principal,
            after_sequence=after_sequence,
            live=live,
            authorized=True,
        ):
            yield event

    async def _stream_claimed(
        self,
        lease: _PreparedStreamLease,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        if lease.state != "CLAIMED" or lease.subscription is None:
            raise RuntimeError("claimed event stream lease is not active")
        async for event in self._stream_with_live(
            lease.execution_id,
            principal=principal,
            after_sequence=after_sequence,
            live=lease.subscription,
            authorized=True,
        ):
            yield event

    async def _stream_with_live(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int,
        live: "_LiveSubscription | None" = None,
        authorized: bool = False,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        cursor = after_sequence
        live = self._live.subscribe(execution_id) if live is None else live
        local_producer = self._live.is_local_producer(execution_id)
        live_task: asyncio.Task[_OrderedItem] | None = None
        activity_task: asyncio.Task[None] | None = None
        try:
            if not authorized:
                await self._authorize_stream(execution_id, principal)

            while True:
                page = await self._read_durable(
                    execution_id,
                    tenant_id=principal.tenant_id,
                    after_sequence=cursor,
                    limit=200,
                )
                marker_sequence = live.first_marker_sequence(after_sequence=cursor)
                items = page.items
                if marker_sequence is not None:
                    items = tuple(item for item in items if item.sequence < marker_sequence)
                terminal_seen = False
                for event in items:
                    cursor = event.sequence
                    yield ExecutionStreamEvent(
                        event.execution_id,
                        event.sequence,
                        event.event_type,
                        event.payload,
                    )
                    terminal_seen = event.event_type in {
                        ExecutionEventType.EXECUTION_SUCCEEDED,
                        ExecutionEventType.EXECUTION_FAILED,
                        ExecutionEventType.EXECUTION_CANCELLED,
                    }
                if terminal_seen:
                    return
                if marker_sequence is not None or page.next_cursor is None:
                    break
                cursor = int(page.next_cursor)

            while True:
                if live_task is None:
                    live_task = asyncio.create_task(live.__anext__())
                if not local_producer and activity_task is None:
                    activity_task = asyncio.create_task(self._live.wait_for_activity(execution_id))
                wait_tasks = (live_task,) if local_producer else (live_task, activity_task)
                done, _ = await asyncio.wait(
                    wait_tasks,
                    timeout=None if local_producer else 1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    _logger.debug(
                        "event stream durable fallback read: execution=%s cursor=%s",
                        execution_id,
                        cursor,
                    )
                    page = await self._read_durable(
                        execution_id,
                        tenant_id=principal.tenant_id,
                        after_sequence=cursor,
                        limit=200,
                    )
                    marker_sequence = live.first_marker_sequence(after_sequence=cursor)
                    items = page.items
                    if marker_sequence is not None:
                        items = tuple(item for item in items if item.sequence < marker_sequence)
                    for event in items:
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
                            return
                    continue
                if live_task in done:
                    try:
                        item = live_task.result()
                    except StopAsyncIteration:
                        live_task = None
                        execution = await self._executions.get(
                            execution_id,
                            tenant_id=principal.tenant_id,
                        )
                        if execution is None:
                            return
                        failure = self._worker_failure(
                            execution_id,
                            tenant_id=principal.tenant_id,
                        )
                        if failure is not None:
                            raise failure
                        if execution.status in {
                            ExecutionStatus.SUCCEEDED,
                            ExecutionStatus.FAILED,
                            ExecutionStatus.CANCELLED,
                        }:
                            page = await self._read_durable(
                                execution_id,
                                tenant_id=principal.tenant_id,
                                after_sequence=cursor,
                                limit=200,
                            )
                            for event in page.items:
                                cursor = event.sequence
                                yield ExecutionStreamEvent(
                                    event.execution_id,
                                    event.sequence,
                                    event.event_type,
                                    event.payload,
                                )
                            return
                        return
                    else:
                        live_task = None
                        if isinstance(item, ExecutionDelta):
                            yield ExecutionStreamEvent(
                                item.execution_id,
                                None,
                                item.delta_type,
                                {
                                    "text": item.content,
                                    "stream_truncated": item.stream_truncated,
                                },
                            )
                        elif item.sequence > cursor:
                            while cursor < item.sequence:
                                page = await self._read_durable(
                                    execution_id,
                                    tenant_id=principal.tenant_id,
                                    after_sequence=cursor,
                                    limit=min(200, item.sequence - cursor),
                                )
                                if not page.items:
                                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                                for event in page.items:
                                    if event.sequence > item.sequence:
                                        break
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
                        if activity_task in done:
                            activity_task = None
                        continue
                if activity_task in done:
                    activity_task = None
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

    async def _authorize_stream(self, execution_id: str, principal: Principal) -> None:
        header = await self._executions.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.EVENT_READ, header)
        await self._authorization.authorize(principal, AuthorizationAction.EXECUTION_READ, header)

    async def _read_durable(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[ExecutionEvent]:
        page = await self._events.list(
            execution_id,
            tenant_id=tenant_id,
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


__all__ = ["DefaultEventService", "ExecutionDelta", "LiveExecutionEventBroker"]
