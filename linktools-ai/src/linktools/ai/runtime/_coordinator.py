#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private local stream coordination for Runtime."""

from collections.abc import AsyncIterator

from linktools.core import environ

from ..core import ExecutionEventType, Principal
from ._event import DefaultEventService
from ._execution import DefaultExecutionService
from .service_api import ExecutionStreamEvent

_logger = environ.get_logger("ai.runtime.coordinator")


class _LocalRuntimeCoordinator:
    """Coordinate local event streaming and terminal handoff."""

    def __init__(
        self,
        execution: DefaultExecutionService,
        event: DefaultEventService,
    ) -> None:
        self._execution = execution
        self._event = event

    async def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        terminal_yielded = False
        try:
            async for event in self._event.stream(
                execution_id,
                principal=principal,
                after_sequence=after_sequence,
            ):
                if event.event_type in {
                    ExecutionEventType.EXECUTION_SUCCEEDED,
                    ExecutionEventType.EXECUTION_FAILED,
                    ExecutionEventType.EXECUTION_CANCELLED,
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


__all__ = ["_LocalRuntimeCoordinator"]
