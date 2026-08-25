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
    JsonValue,
    Page,
    Principal,
)
from ..errors import AIError, ErrorCode
from .service_api import ExecutionEvent, ExecutionStreamEvent
from .state._contracts import EventRepository, ExecutionRepository

_logger = environ.get_logger("ai.runtime.event")
_DEFAULT_BUFFER_BYTES = 1024 * 1024
_QUEUE_LIMIT = 256
_TERMINAL_EVENT_TYPES = frozenset({
    ExecutionEventType.EXECUTION_SUCCEEDED,
    ExecutionEventType.EXECUTION_FAILED,
    ExecutionEventType.EXECUTION_CANCELLED,
})


@dataclass(frozen=True, slots=True)
class ExecutionDelta:
    execution_id: str
    delta_type: ExecutionDeltaType
    content: str
    stream_truncated: bool = False




@dataclass(slots=True)
class _LiveEvent:
    execution_id: str
    event_type: ExecutionEventType
    payload: JsonValue
    durable_sequence: int | None = None



@dataclass(slots=True)
class _PreparedStreamLease:
    execution_id: str
    state: str = "PREPARED"
    subscription: "_LiveSubscription | None" = None
    base_sequence: int | None = None


_OrderedItem = ExecutionDelta | _LiveEvent



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

    def put_delta(self, value: ExecutionDelta) -> bool:
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

    def put_event(self, value: _LiveEvent) -> None:
        if self._closed:
            return
        self._queue.append(value)
        self._wakeup.set()

    def finish(self) -> None:
        if self._closed:
            return
        self._completed = True
        self._wakeup.set()

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
        self._base_sequences: dict[str, int] = {}
        self._prepared: dict[str, _PreparedStreamLease] = {}
        self._durable_events: dict[
            tuple[str, int],
            tuple[ExecutionEventType, JsonValue],
        ] = {}

    def register_local_producer(self, execution_id: str, base_sequence: int) -> None:
        if base_sequence < 0:
            raise ValueError("local stream base sequence cannot be negative")
        previous = self._base_sequences.get(execution_id)
        if previous is not None:
            if previous != base_sequence:
                raise RuntimeError("local stream base sequence changed")
            return
        self._base_sequences[execution_id] = base_sequence
        lease = self._prepared.get(execution_id)
        if lease is not None:
            lease.base_sequence = base_sequence
        _logger.debug(
            "local event producer registered: execution=%s base_sequence=%s",
            execution_id,
            base_sequence,
        )
        self._signal(execution_id)

    def base_sequence(self, execution_id: str) -> int | None:
        return self._base_sequences.get(execution_id)

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
                    previous.stream_truncated
                    or delta.stream_truncated
                    or execution_id in self._truncated,
                ),
                self._max_bytes,
            )
            buffer.append(merged)
            self._buffer_bytes[execution_id] += (
                len(merged.content.encode("utf-8"))
                - len(previous.content.encode("utf-8"))
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
            execution_id,
            delta.delta_type,
            delta.content,
            delta.stream_truncated or execution_id in self._truncated,
        )
        for subscription in tuple(self._subscriptions.get(execution_id, ())):
            subscription.put_delta(published)
        self._signal(execution_id)

    def publish_event(
        self,
        execution_id: str,
        event_type: ExecutionEventType,
        payload: JsonValue,
        *,
        durable_sequence: int | None,
    ) -> None:
        if execution_id not in self._base_sequences:
            return
        if durable_sequence is not None:
            if durable_sequence < 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            key = (execution_id, durable_sequence)
            previous = self._durable_events.get(key)
            if previous is not None:
                if previous != (event_type, payload):
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        safe_details={
                            "phase": "live_durable_event_dedupe",
                            "execution_id": execution_id,
                            "durable_sequence": durable_sequence,
                        },
                    )
                return
            self._durable_events[key] = (event_type, payload)
        event = _LiveEvent(execution_id, event_type, payload, durable_sequence)
        self._buffers.setdefault(execution_id, deque()).append(event)
        self._buffer_bytes.setdefault(execution_id, 0)
        self._last_type.pop(execution_id, None)
        for subscription in tuple(self._subscriptions.get(execution_id, ())):
            subscription.put_event(event)
        self._signal(execution_id)

    def confirm_events(
        self,
        execution_id: str,
        *,
        first_sequence: int,
        count: int,
    ) -> None:
        if count < 0 or first_sequence < 1:
            raise ValueError("durable event confirmation range is invalid")
        if count == 0 or execution_id not in self._base_sequences:
            return
        sequence = first_sequence
        confirmed = 0
        for value in self._buffers.get(execution_id, ()):
            if not isinstance(value, _LiveEvent) or value.durable_sequence is not None:
                continue
            value.durable_sequence = sequence
            sequence += 1
            confirmed += 1
            if confirmed == count:
                break
        if confirmed != count:
            raise RuntimeError("live event confirmation does not match pending audit")
        self._signal(execution_id)

    def subscribe(self, execution_id: str) -> _LiveSubscription:
        subscription = _LiveSubscription(self, execution_id, self._max_bytes)
        self._subscriptions.setdefault(execution_id, set()).add(subscription)
        self._fill_subscription(subscription, execution_id)
        if execution_id in self._completed:
            subscription.finish()
        return subscription

    def prepare_local_producer(self, execution_id: str) -> _PreparedStreamLease:
        current = self._prepared.get(execution_id)
        if current is not None and current.state == "PREPARED":
            return current
        lease = _PreparedStreamLease(execution_id)
        self._prepared[execution_id] = lease
        _logger.debug("local event stream prepared: execution=%s", execution_id)
        return lease

    def claim_local_producer(self, lease: _PreparedStreamLease) -> _LiveSubscription:
        current = self._prepared.get(lease.execution_id)
        if current is not lease or lease.state != "PREPARED":
            raise RuntimeError("prepared event stream lease is not claimable")
        if lease.base_sequence is None:
            raise RuntimeError("prepared event stream has no durable base")
        subscription = _LiveSubscription(self, lease.execution_id, self._max_bytes)
        lease.state = "CLAIMED"
        lease.subscription = subscription
        self._prepared.pop(lease.execution_id, None)
        self._subscriptions.setdefault(lease.execution_id, set()).add(subscription)
        self._fill_subscription(subscription, lease.execution_id)
        if lease.execution_id in self._completed:
            subscription.finish()
        return subscription

    def abort_local_producer(self, lease: _PreparedStreamLease) -> None:
        if lease.state != "PREPARED":
            return
        current = self._prepared.get(lease.execution_id)
        if current is not lease:
            lease.state = "ABANDONED"
            return
        lease.state = "ABANDONED"
        self._prepared.pop(lease.execution_id, None)
        if lease.base_sequence is None or (
            lease.execution_id in self._completed
            and not self._subscriptions.get(lease.execution_id)
        ):
            self._release_execution(lease.execution_id)
        else:
            self._signal(lease.execution_id)
        _logger.debug(
            "local event stream lease abandoned: execution=%s base_sequence=%s",
            lease.execution_id,
            lease.base_sequence,
        )

    def is_local_producer(self, execution_id: str) -> bool:
        return execution_id in self._base_sequences

    def is_completed(self, execution_id: str) -> bool:
        return execution_id in self._completed

    async def wait_for_activity(self, execution_id: str) -> None:
        event = self._activity.setdefault(execution_id, asyncio.Event())
        await event.wait()
        event.clear()

    def complete(self, execution_id: str) -> None:
        self._completed.add(execution_id)
        subscriptions = tuple(self._subscriptions.get(execution_id, ()))
        for subscription in subscriptions:
            subscription.finish()
        self._signal(execution_id)
        if not subscriptions and execution_id not in self._prepared:
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
                subscription.put_delta(
                    ExecutionDelta(
                        item.execution_id,
                        item.delta_type,
                        item.content,
                        item.stream_truncated or execution_id in self._truncated,
                    )
                )
            else:
                subscription.put_event(item)

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
        self._base_sequences.pop(execution_id, None)
        self._activity.pop(execution_id, None)
        self._prepared.pop(execution_id, None)
        for key in tuple(self._durable_events):
            if key[0] == execution_id:
                self._durable_events.pop(key, None)


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
        if not authorized:
            await self._authorize_stream(execution_id, principal)
        if not self._live.is_local_producer(execution_id):
            async for event in self._stream_durable(
                execution_id,
                tenant_id=principal.tenant_id,
                after_sequence=after_sequence,
            ):
                yield event
            return

        base_sequence = self._live.base_sequence(execution_id)
        if base_sequence is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        live = self._live.subscribe(execution_id) if live is None else live
        cursor = after_sequence
        try:
            while cursor < base_sequence:
                page = await self._read_durable(
                    execution_id,
                    tenant_id=principal.tenant_id,
                    after_sequence=cursor,
                    limit=min(200, base_sequence - cursor),
                )
                items = tuple(item for item in page.items if item.sequence <= base_sequence)
                if not items:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for event in items:
                    if event.sequence != cursor + 1:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    cursor = event.sequence
                    yield ExecutionStreamEvent(
                        event.execution_id,
                        event.sequence,
                        event.event_type,
                        event.payload,
                    )
                    if event.event_type in _TERMINAL_EVENT_TYPES:
                        return

            replay_cursor = after_sequence if after_sequence > base_sequence else None
            async for item in live:
                if replay_cursor is not None:
                    if isinstance(item, ExecutionDelta):
                        continue
                    while item.durable_sequence is None:
                        if self._live.is_completed(execution_id):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        try:
                            await asyncio.wait_for(
                                self._live.wait_for_activity(execution_id),
                                timeout=1.0,
                            )
                        except TimeoutError:
                            pass
                    if item.durable_sequence <= replay_cursor:
                        if item.event_type in _TERMINAL_EVENT_TYPES:
                            return
                        if item.durable_sequence == replay_cursor:
                            replay_cursor = None
                        continue
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if isinstance(item, ExecutionDelta):
                    yield ExecutionStreamEvent(
                        item.execution_id,
                        None,
                        item.delta_type,
                        {"text": item.content, "stream_truncated": item.stream_truncated},
                    )
                    continue
                if item.durable_sequence is not None:
                    if item.durable_sequence <= after_sequence:
                        if item.event_type in _TERMINAL_EVENT_TYPES:
                            return
                        continue
                    cursor = max(cursor, item.durable_sequence)
                yield ExecutionStreamEvent(
                    item.execution_id,
                    item.durable_sequence,
                    item.event_type,
                    item.payload,
                )
                if item.event_type in _TERMINAL_EVENT_TYPES:
                    return
        finally:
            await live.close()

        failure = self._worker_failure(execution_id, tenant_id=principal.tenant_id)
        if failure is not None:
            raise failure
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


    async def _stream_durable(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        cursor = after_sequence
        while True:
            page = await self._read_durable(
                execution_id,
                tenant_id=tenant_id,
                after_sequence=cursor,
                limit=200,
            )
            if page.items:
                for event in page.items:
                    if event.sequence != cursor + 1:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    cursor = event.sequence
                    yield ExecutionStreamEvent(
                        event.execution_id,
                        event.sequence,
                        event.event_type,
                        event.payload,
                    )
                    if event.event_type in _TERMINAL_EVENT_TYPES:
                        return
                continue
            failure = self._worker_failure(execution_id, tenant_id=tenant_id)
            if failure is not None:
                raise failure
            execution = await self._executions.get(execution_id, tenant_id=tenant_id)
            if execution is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                if cursor >= execution.event_sequence:
                    return
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                await asyncio.wait_for(
                    self._live.wait_for_activity(execution_id),
                    timeout=1.0,
                )
            except TimeoutError:
                pass

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
