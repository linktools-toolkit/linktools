#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for Task background error metadata propagation."""

import asyncio
from types import SimpleNamespace
import pytest

from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
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
        graph_id="graph",
    )

    with pytest.raises(AIError) as captured:
        await launcher.start(request)  # type: ignore[arg-type]
    _assert_source_metadata(captured.value)
    assert captured.value.safe_details == {"status_code": 503}


@pytest.mark.asyncio
async def test_task_waiter_rethrows_full_failure_metadata() -> None:
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._lock = asyncio.Lock()
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
