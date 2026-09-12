#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for stable error propagation across runtime boundaries."""

import asyncio
from types import SimpleNamespace

import pytest
from httpx import Request, Response
from openai import APIStatusError, APITimeoutError
from pydantic_ai.usage import RunUsage, UsageLimits

from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.runtime._agent_executor import _execution_error
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._local import _RecoveryCoordinator
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
)
from linktools.ai.task import DefaultTaskGraphService


def _map_provider_error(error: Exception) -> AIError:
    return _execution_error(
        error,
        usage_limits=UsageLimits(),
        run_usage=RunUsage(),
    )


def test_raw_openai_timeout_is_model_timeout() -> None:
    error = APITimeoutError(
        request=Request("POST", "https://provider.invalid/v1/chat/completions")
    )
    mapped = _map_provider_error(error)
    assert mapped.code is ErrorCode.MODEL_TIMEOUT
    assert mapped.retryable is True
    assert mapped.safe_details == {}
    assert mapped.diagnostics == ErrorDiagnostics.from_exception(error)


@pytest.mark.parametrize(
    ("status_code", "expected", "retryable"),
    [
        (400, ErrorCode.MODEL_REQUEST_REJECTED, False),
        (408, ErrorCode.MODEL_TIMEOUT, True),
        (429, ErrorCode.MODEL_RATE_LIMITED, True),
        (500, ErrorCode.MODEL_UNAVAILABLE, True),
        (503, ErrorCode.MODEL_UNAVAILABLE, True),
    ],
)
def test_raw_openai_status_errors_use_model_http_classification(
    status_code: int,
    expected: ErrorCode,
    retryable: bool,
) -> None:
    request = Request("POST", "https://provider.invalid/v1/chat/completions")
    error = APIStatusError(
        "provider secret",
        response=Response(status_code, request=request),
        body={"secret": "provider body must not escape"},
    )
    mapped = _map_provider_error(error)
    assert mapped.code is expected
    assert mapped.retryable is retryable
    assert mapped.safe_details == {"status_code": status_code}
    assert "secret" not in str(mapped.safe_details)


@pytest.mark.asyncio
async def test_task_scheduler_arm_preserves_classified_ai_error() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("launch failed"))

    class Launcher:
        async def start(self, launch: object) -> None:
            del launch
            raise AIError(
                ErrorCode.EXECUTION_START_UNKNOWN,
                category="EXECUTION",
                retryable=True,
                operation_id="launch-operation",
                safe_details={"source": "launcher"},
                diagnostics=diagnostics,
            )

    service = DefaultTaskGraphService(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        Launcher(),  # type: ignore[arg-type]
    )
    launch = SimpleNamespace(
        principal=SimpleNamespace(tenant_id="tenant"),
        graph=SimpleNamespace(graph_id="graph"),
    )
    with pytest.raises(AIError) as captured:
        await service._arm_graph(launch)  # type: ignore[arg-type]
    error = captured.value
    assert error.code is ErrorCode.EXECUTION_START_UNKNOWN
    assert error.category == "EXECUTION"
    assert error.retryable is True
    assert error.operation_id == "launch-operation"
    assert error.diagnostics is diagnostics
    assert error.safe_details == {
        "source": "launcher",
        "phase": "task_scheduler_arm",
        "graph_id": "graph",
        "durable_admitted": True,
    }


@pytest.mark.asyncio
async def test_task_service_finalizer_preserves_classified_ai_error() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("task finalizer"))
    service = DefaultTaskGraphService(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )

    async def fail() -> None:
        raise AIError(
            ErrorCode.MODEL_TIMEOUT,
            retryable=True,
            operation_id="task-finalizer",
            safe_details={"source": "task"},
            diagnostics=diagnostics,
        )

    task = asyncio.create_task(fail())
    service._detach_finalizer(task, "graph")
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    with pytest.raises(AIError) as captured:
        await service.preflight_close()
    error = captured.value
    assert error.code is ErrorCode.MODEL_TIMEOUT
    assert error.retryable is True
    assert error.operation_id == "task-finalizer"
    assert error.diagnostics is diagnostics
    assert error.safe_details == {
        "source": "task",
        "phase": "task_service_finalizer",
        "graph_id": "graph",
    }


@pytest.mark.asyncio
async def test_execution_cancel_finalizer_preserves_classified_ai_error() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("execution finalizer"))
    service = object.__new__(DefaultExecutionService)
    service._detached_cancel_finalizers = set()
    service._detached_cancel_failure = None

    async def fail() -> object:
        raise AIError(
            ErrorCode.MODEL_UNAVAILABLE,
            retryable=True,
            operation_id="execution-finalizer",
            safe_details={"source": "execution"},
            diagnostics=diagnostics,
        )

    task = asyncio.create_task(fail())
    service._detach_cancel_finalizer(task, "execution")  # type: ignore[arg-type]
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    with pytest.raises(AIError) as captured:
        await service.preflight_close()
    error = captured.value
    assert error.code is ErrorCode.MODEL_UNAVAILABLE
    assert error.retryable is True
    assert error.operation_id == "execution-finalizer"
    assert error.diagnostics is diagnostics
    assert error.safe_details == {
        "source": "execution",
        "phase": "execution_cancel_finalizer",
        "execution_id": "execution",
    }


def test_subagent_background_failure_preserves_classified_ai_error() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("subagent cleanup"))
    dispatcher = object.__new__(SubagentDispatcher)
    dispatcher._background_failures = {}
    source = AIError(
        ErrorCode.MODEL_RATE_LIMITED,
        retryable=True,
        operation_id="subagent-cleanup",
        safe_details={"status_code": 429},
        diagnostics=diagnostics,
    )
    stored = dispatcher._record_background_failure(
        "execution",
        source,
        phase="subagent_cancel_cleanup",
    )
    assert stored.code is ErrorCode.MODEL_RATE_LIMITED
    assert stored.retryable is True
    assert stored.operation_id == "subagent-cleanup"
    assert stored.diagnostics is diagnostics
    assert stored.safe_details == {
        "status_code": 429,
        "phase": "subagent_cancel_cleanup",
        "execution_id": "execution",
    }
    replayed = dispatcher.background_failure
    assert replayed is not None
    assert replayed.code is ErrorCode.MODEL_RATE_LIMITED
    assert replayed.retryable is True
    assert replayed.operation_id == "subagent-cleanup"
    assert replayed.diagnostics is diagnostics
    assert replayed.safe_details == stored.safe_details


@pytest.mark.asyncio
async def test_recovery_start_unknown_uses_execution_error_domain() -> None:
    execution = SimpleNamespace(
        execution_id="execution",
        tenant_id="tenant",
        principal_id="principal",
        principal_kind="user",
        binding_digest="binding",
        parent_execution_id=None,
        root_execution_id="execution",
        parent_invocation_id=None,
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        planning=False,
        thinking=False,
        binding="snapshot",
        repository_instructions=None,
        correlation={},
        session_id=None,
        status=ExecutionStatus.START_UNKNOWN,
    )

    class Port:
        async def load_execution(self, execution_id: str, *, tenant_id: str) -> object:
            assert execution_id == "execution"
            assert tenant_id == "tenant"
            return execution

        def _validate_recovery_identity(self, value: object) -> None:
            assert value is execution

        def validate_binding(self, value: object) -> None:
            assert value is execution

        async def _reconcile_session_recovery(
            self,
            checkpoint: object,
            value: object,
        ) -> bool:
            del checkpoint
            assert value is execution
            return True

        async def _recovery_idempotency(self, value: object) -> object:
            assert value is execution
            return object()

    coordinator = _RecoveryCoordinator(Port(), None)
    checkpoint = SimpleNamespace(
        execution_id="execution",
        tenant_id="tenant",
        handoff_phase=RecoveryHandoffPhase.NONE,
        state=RecoveryCheckpointState.ACTIVE,
    )
    with pytest.raises(AIError) as captured:
        await coordinator.reconcile_checkpoint(checkpoint)  # type: ignore[arg-type]
    assert captured.value.code is ErrorCode.EXECUTION_START_UNKNOWN
