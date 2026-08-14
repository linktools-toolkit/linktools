#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current task, result, and temporal public contracts."""

import pytest
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionResult
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.task import TaskCompletionLedger
from linktools.ai.temporal import (
    EvaluationActivity,
    ExecuteActivity,
    SessionActivity,
    TaskActivity,
    WorkerActivities,
    production_registration,
)


def test_task_completion_uses_owner_fence_and_result_identity() -> None:
    ledger = TaskCompletionLedger()
    assert ledger.complete("task", "owner", 1, "digest") == ledger.complete("task", "owner", 1, "digest")
    with pytest.raises(AIError) as error:
        ledger.complete("task", "owner", 1, "other")
    assert error.value.code is ErrorCode.TASK_RESULT_CONFLICT


def test_temporal_registration_has_one_explicit_worker_surface() -> None:
    class Operation:
        pass

    operation = Operation()
    activities = WorkerActivities(
        ExecuteActivity(operation),
        SessionActivity(operation),
        TaskActivity(operation),
        EvaluationActivity(operation),
    )
    registration = production_registration(activities)
    assert len(registration.activities) == 4
    assert len(registration.workflows) == 4


def test_execution_service_exposes_terminal_result_contract() -> None:
    assert DefaultExecutionService.result.__annotations__["return"] is ExecutionResult
    assert ExecutionStatus.CANCELLED.value == "CANCELLED"
