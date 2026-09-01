#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for Task background error metadata propagation."""

import asyncio
from types import SimpleNamespace

import pytest

from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.runtime._planner import _AgentTaskNodeHandler, RuntimeTaskNodeRunner
from linktools.ai.task._local import LocalTaskGraphLauncher, _scheduler_failure


def _source_error() -> AIError:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("provider unavailable"))
    return AIError(
        ErrorCode.MODEL_UNAVAILABLE,
        category="MODEL",
        retryable=True,
        operation_id="provider-operation",
        safe_details={"status_code": 503},
        diagnostics=diagnostics,
    )


def _assert_source_metadata(error: AIError) -> None:
    assert error.code is ErrorCode.MODEL_UNAVAILABLE
    assert error.category == "MODEL"
    assert error.retryable is True
    assert error.operation_id == "provider-operation"
    assert error.diagnostics == _source_error().diagnostics


def test_agent_task_background_failure_preserves_metadata() -> None:
    handler = object.__new__(_AgentTaskNodeHandler)
    handler._background_failures = {}
    source = _source_error()

    stored = handler._record_background_failure(
        ("tenant", "graph", "node"),
        source,
        phase="task_execution_bind",
    )
    _assert_source_metadata(stored)
    assert stored.safe_details == {
        "status_code": 503,
        "phase": "task_execution_bind",
        "graph_id": "graph",
        "node_id": "node",
    }

    replayed = handler.background_failure
    assert replayed is not None
    _assert_source_metadata(replayed)
    assert replayed.safe_details == stored.safe_details


def test_runtime_task_runner_background_failure_preserves_metadata() -> None:
    runner = object.__new__(RuntimeTaskNodeRunner)
    runner._background_failure = _source_error()
    runner._agent = SimpleNamespace(background_failure=None)

    failure = runner.background_failure
    assert failure is not None
    _assert_source_metadata(failure)
    assert failure.safe_details == {"status_code": 503}


def test_task_scheduler_failure_preserves_metadata() -> None:
    failure = _scheduler_failure(_source_error(), "graph")
    _assert_source_metadata(failure)
    assert failure.safe_details == {"status_code": 503, "graph_id": "graph"}


@pytest.mark.asyncio
async def test_cached_task_failure_rethrows_full_metadata() -> None:
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._accepting = True
    launcher._lock = asyncio.Lock()
    launcher._graphs = {
        ("tenant", "graph"): SimpleNamespace(
            failure=_source_error(),
            closed=False,
        )
    }
    request = SimpleNamespace(
        principal=SimpleNamespace(tenant_id="tenant"),
        graph=SimpleNamespace(graph_id="graph"),
    )

    with pytest.raises(AIError) as captured:
        await launcher.start(request)  # type: ignore[arg-type]
    _assert_source_metadata(captured.value)
    assert captured.value.safe_details == {"status_code": 503}


@pytest.mark.asyncio
async def test_task_waiter_rethrows_full_failure_metadata() -> None:
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._graphs = {
        ("tenant", "graph"): SimpleNamespace(
            failure=_source_error(),
            closed=True,
            generation=0,
        )
    }

    with pytest.raises(AIError) as captured:
        await launcher.wait_graph_activity("graph", tenant_id="tenant")
    _assert_source_metadata(captured.value)
    assert captured.value.safe_details == {"status_code": 503}


@pytest.mark.asyncio
async def test_task_recovery_preserves_full_failure_metadata() -> None:
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._lock = asyncio.Lock()
    request = SimpleNamespace(
        principal=SimpleNamespace(tenant_id="tenant"),
        graph=SimpleNamespace(graph_id="graph"),
    )
    run = SimpleNamespace(
        request=request,
        condition=asyncio.Condition(),
        generation=0,
        failure=None,
        closed=False,
    )
    launcher._graphs = {("tenant", "graph"): run}
    node = SimpleNamespace(node_id="node")

    await launcher._defer_recovery(run, node, cause=_source_error())

    assert run.failure is not None
    _assert_source_metadata(run.failure)
    assert run.failure.safe_details == {
        "status_code": 503,
        "phase": "task_node_recovery",
        "graph_id": "graph",
        "node_id": "node",
        "cause_code": ErrorCode.MODEL_UNAVAILABLE.value,
    }
    assert run.closed is True
