#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable event query and stream API."""

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    ExecutionEventType,
    ExecutionStatus,
    Page,
    Principal,
)
from ..errors import AIError, ErrorCode
from ._persistence import RuntimeStores
from ._services import ExecutionEvent

_logger = environ.get_logger("ai.runtime.event")


class EventApi(Protocol):
    async def list(self, execution_id: str, *, principal: Principal, after_sequence: int = 0, limit: int = 100) -> 'Page[ExecutionEvent]': ...
    def stream(self, execution_id: str, *, principal: Principal, after_sequence: int = 0) -> 'AsyncIterator[ExecutionEvent]': ...


class DefaultEventService:
    """Read durable events and expose a bounded durable-first stream."""

    def __init__(self, persistence: RuntimeStores, authorization: AuthorizationPolicy) -> None:
        self._persistence = persistence
        self._authorization = authorization

    async def list(self, execution_id: str, *, principal: Principal, after_sequence: int = 0, limit: int = 100) -> Page[ExecutionEvent]:
        header = await self._persistence.execution.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.EVENT_READ, header)
        await self._authorization.authorize(principal, AuthorizationAction.EXECUTION_READ, header)
        page = await self._persistence.execution_events.list(execution_id, tenant_id=principal.tenant_id, after_sequence=after_sequence, limit=limit)
        return Page(tuple(ExecutionEvent(item.execution_id, item.sequence, item.event_type, item.payload) for item in page.items), page.next_cursor)

    async def stream(self, execution_id: str, *, principal: Principal, after_sequence: int = 0) -> AsyncIterator[ExecutionEvent]:
        cursor = after_sequence
        while True:
            page = await self.list(execution_id, principal=principal, after_sequence=cursor, limit=200)
            if not page.items:
                execution = await self._persistence.execution.get(execution_id, tenant_id=principal.tenant_id)
                if execution is None or execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    return
                await asyncio.sleep(0.1)
                continue
            for event in page.items:
                cursor = event.sequence
                yield event
                if event.event_type in {
                    ExecutionEventType.EXECUTION_SUCCEEDED,
                    ExecutionEventType.EXECUTION_FAILED,
                    ExecutionEventType.EXECUTION_CANCELLED,
                }:
                    _logger.debug("event stream reached terminal: execution=%s sequence=%s", execution_id, event.sequence)
                    return
            await asyncio.sleep(0)


__all__ = ["DefaultEventService", "EventApi"]
