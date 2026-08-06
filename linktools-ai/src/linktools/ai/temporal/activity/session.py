#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session Activity with one injected mutation-operation port."""

from typing import Protocol

from linktools.core import environ

from ..workflow.session import SessionWorkflowInput, SessionWorkflowResult
from .execution import ActivityOptions

try:
    from temporalio import activity as _temporal_activity
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_activity = None

_logger = environ.get_logger("ai.temporal.activity.session")


class SessionOperation(Protocol):
    async def execute(self, request: SessionWorkflowInput) -> SessionWorkflowResult: ...


class SessionActivity:
    options = ActivityOptions(start_to_close_seconds=60, retry_max_attempts=3, heartbeat_timeout_seconds=15)

    def __init__(self, operation: SessionOperation) -> None:
        self._operation = operation

    async def run(self, request: SessionWorkflowInput) -> SessionWorkflowResult:
        _logger.info("executing session activity: session_id=%s mutation_id=%s", request.session_id, request.mutation_id)
        return await self._operation.execute(request)


if _temporal_activity is not None:
    SessionActivity.run = _temporal_activity.defn(name="session_mutation")(SessionActivity.run)


__all__ = ["SessionActivity", "SessionOperation"]
