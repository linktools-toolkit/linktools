#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation Activity with one injected case-operation port."""

from typing import Protocol

from linktools.core import environ

from ..workflow.evaluation import EvaluationWorkflowInput, EvaluationWorkflowResult
from .execution import ActivityOptions

try:
    from temporalio import activity as _temporal_activity
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_activity = None

_logger = environ.get_logger("ai.temporal.activity.evaluation")


class EvaluationOperation(Protocol):
    async def execute(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult: ...


class EvaluationActivity:
    options = ActivityOptions(start_to_close_seconds=1800, retry_max_attempts=2, heartbeat_timeout_seconds=60)

    def __init__(self, operation: EvaluationOperation) -> None:
        self._operation = operation

    async def run(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult:
        _logger.info("executing evaluation activity: evaluation_id=%s", request.evaluation_id)
        return await self._operation.execute(request)


if _temporal_activity is not None:
    EvaluationActivity.run = _temporal_activity.defn(name="evaluation")(EvaluationActivity.run)


__all__ = ["EvaluationActivity", "EvaluationOperation"]
