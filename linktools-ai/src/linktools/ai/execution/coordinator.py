#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinate execution streaming and terminal ownership handoff."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from linktools.core import environ

from ..core import ExecutionEventType

if TYPE_CHECKING:
    from ..core import Page, Principal
    from ._service_api import EventService, ExecutionEvent, ExecutionService, ExecutionStreamEvent

_logger = environ.get_logger("ai.execution.coordinator")


class ExecutionStreamCoordinator:
    """Coordinate local event streaming and terminal handoff."""

    def __init__(
        self,
        execution: ExecutionService,
        event: EventService,
    ) -> None:
        self._execution = execution
        self._event = event

    async def list(
        self, execution_id: str, *, principal: Principal,
        after_event_seq: int = 0, limit: int = 100,
    ) -> Page[ExecutionEvent]:
        return await self._event.list(
            execution_id, principal=principal, after_event_seq=after_event_seq, limit=limit,
        )

    async def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        terminal_yielded = False
        try:
            async for event in self._event.stream(
                execution_id,
                principal=principal,
                after_event_seq=after_event_seq,
            ):
                if event.event_type in {
                    ExecutionEventType.EXECUTION_SUCCEEDED.value,
                    ExecutionEventType.EXECUTION_FAILED.value,
                    ExecutionEventType.EXECUTION_CANCELLED.value,
                }:
                    terminal_yielded = True
                yield event
        finally:
            if terminal_yielded:
                _logger.debug(
                    "local stream yielded terminal event; requesting execution handoff: execution=%s",
                    execution_id,
                )
                await self._execution.request_terminal_handoff(
                    execution_id,
                    tenant_id=principal.tenant_id,
                )


__all__ = ["ExecutionStreamCoordinator"]
