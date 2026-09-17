#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable event queries and execution streaming service."""

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    ExecutionEventType,
    ExecutionStatus,
    Page,
    Principal,
)
from ..errors import AIError, ErrorCode
from . import _event_live
from .service_api import ExecutionEvent, ExecutionStreamEvent
from .state._contracts import EventRepository, ExecutionRepository

_TERMINAL_EVENT_TYPES = frozenset(
    {
        ExecutionEventType.EXECUTION_SUCCEEDED.value,
        ExecutionEventType.EXECUTION_FAILED.value,
        ExecutionEventType.EXECUTION_CANCELLED.value,
    }
)
_OBSERVATION_BOUNDARY_EVENT_TYPES = _TERMINAL_EVENT_TYPES | frozenset(
    {ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value}
)
_OBSERVATION_BOUNDARY_STATUSES = frozenset(
    {
        ExecutionStatus.SUCCEEDED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.RECOVERY_REQUIRED,
    }
)


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
        live_broker: _event_live.LiveExecutionEventBroker | None = None,
    ) -> None:
        self._executions = executions
        self._events = events
        self._authorization = authorization
        self._worker_failure = worker_failure
        self._live = live_broker or _event_live.LiveExecutionEventBroker()

    @property
    def live_broker(self) -> _event_live.LiveExecutionEventBroker:
        return self._live

    async def list(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> Page[ExecutionEvent]:
        header = await self._executions.get_header(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            principal,
            AuthorizationAction.EVENT_READ,
            header,
        )
        await self._authorization.authorize(
            principal,
            AuthorizationAction.EXECUTION_READ,
            header,
        )
        page = await self._events.list(
            execution_id,
            tenant_id=principal.tenant_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        return Page(
            tuple(
                ExecutionEvent(
                    item.execution_id,
                    item.sequence,
                    item.event_type,
                    item.payload,
                )
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
        await self._authorize_stream(execution_id, principal)
        live = self._live.claim_local_producer(execution_id)
        async for event in self._stream_with_live(
            execution_id,
            principal=principal,
            after_sequence=after_sequence,
            live=live,
            authorized=True,
        ):
            yield event

    async def _stream_with_live(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int,
        live: _event_live._LiveSubscription | None = None,
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
                items = tuple(
                    item for item in page.items if item.sequence <= base_sequence
                )
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
                    if event.event_type in _OBSERVATION_BOUNDARY_EVENT_TYPES:
                        return

            replay_cursor = after_sequence if after_sequence > base_sequence else None
            poll_backoff = 1.0
            async for item in live:
                if replay_cursor is not None:
                    if isinstance(item, _event_live.ExecutionDelta):
                        poll_backoff = 1.0
                        continue
                    while item.durable_sequence is None:
                        if self._live.is_completed(execution_id):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        try:
                            await asyncio.wait_for(
                                self._live.wait_for_activity(execution_id),
                                timeout=poll_backoff,
                            )
                        except TimeoutError:
                            poll_backoff = min(30.0, poll_backoff * 2)
                        else:
                            poll_backoff = 1.0
                    poll_backoff = 1.0
                    if item.durable_sequence <= replay_cursor:
                        if item.event_type in _TERMINAL_EVENT_TYPES:
                            return
                        if item.durable_sequence == replay_cursor:
                            replay_cursor = None
                        continue
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if isinstance(item, _event_live.ExecutionDelta):
                    poll_backoff = 1.0
                    yield ExecutionStreamEvent(
                        item.execution_id,
                        None,
                        item.delta_type,
                        {
                            "text": item.content,
                            "stream_truncated": item.stream_truncated,
                        },
                    )
                    continue
                if item.durable_sequence is not None:
                    poll_backoff = 1.0
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
                if item.event_type in _OBSERVATION_BOUNDARY_EVENT_TYPES:
                    return
        finally:
            await live.close()

        failure = self._worker_failure(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if failure is not None:
            raise failure
        execution = await self._executions.get(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.status in _OBSERVATION_BOUNDARY_STATUSES:
            if cursor >= execution.event_sequence:
                return
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _stream_durable(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        cursor = after_sequence
        poll_backoff = 1.0
        while True:
            page = await self._read_durable(
                execution_id,
                tenant_id=tenant_id,
                after_sequence=cursor,
                limit=200,
            )
            if page.items:
                poll_backoff = 1.0
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
                    if event.event_type in _OBSERVATION_BOUNDARY_EVENT_TYPES:
                        return
                continue
            failure = self._worker_failure(execution_id, tenant_id=tenant_id)
            if failure is not None:
                raise failure
            execution = await self._executions.get(
                execution_id,
                tenant_id=tenant_id,
            )
            if execution is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution.status in _OBSERVATION_BOUNDARY_STATUSES:
                if cursor >= execution.event_sequence:
                    return
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                await asyncio.wait_for(
                    self._live.wait_for_activity(execution_id),
                    timeout=poll_backoff,
                )
            except TimeoutError:
                poll_backoff = min(30.0, poll_backoff * 2)
            else:
                poll_backoff = 1.0

    async def _authorize_stream(
        self,
        execution_id: str,
        principal: Principal,
    ) -> None:
        header = await self._executions.get_header(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            principal,
            AuthorizationAction.EVENT_READ,
            header,
        )
        await self._authorization.authorize(
            principal,
            AuthorizationAction.EXECUTION_READ,
            header,
        )

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
                ExecutionEvent(
                    item.execution_id,
                    item.sequence,
                    item.event_type,
                    item.payload,
                )
                for item in page.items
            ),
            page.next_cursor,
        )


__all__ = ["DefaultEventService"]
