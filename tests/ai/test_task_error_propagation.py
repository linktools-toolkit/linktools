#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for Agent execution errors crossing TaskGraph boundaries."""

import pytest

from ._task_test_helpers import admit_graph
from linktools.ai.core import ExecutionStatus, TaskStatus, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model._openai import _OpenAIModelBinding
from linktools.ai.runtime import ExecutionResult, RuntimeState
from linktools.ai.runtime._planner import _execution_failure
from linktools.ai.task import TaskGraph, TaskNode, TaskNodeRunError


def test_model_config_invalid_is_non_retryable() -> None:
    assert AIError(ErrorCode.MODEL_CONFIG_INVALID).retryable is False


def test_execution_failure_preserves_task_execution_identity() -> None:
    failure = _execution_failure(
        ExecutionResult(
            "execution-failed",
            ExecutionStatus.FAILED,
            None,
            None,
            UsageMetrics(),
            ErrorCode.MODEL_CONFIG_INVALID.value,
            {"reason": "missing_required_field", "field": "base_url"},
        )
    )

    assert isinstance(failure, TaskNodeRunError)
    assert failure.code is ErrorCode.MODEL_CONFIG_INVALID
    assert failure.execution_id == "execution-failed"
    assert failure.safe_details == {
        "reason": "missing_required_field",
        "field": "base_url",
    }


def test_openai_materialization_normalizes_provider_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(AIError) as raised:
        _OpenAIModelBinding("route", "gpt-4o-mini").materialize()

    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.retryable is False
    assert raised.value.safe_details == {
        "provider": "openai",
        "reason": "provider_configuration_invalid",
    }


@pytest.mark.asyncio
async def test_failed_task_node_persists_execution_identity() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-error-propagation", tenant_id="tenant")
    try:
        repository = state.task.tasks
        await admit_graph(
            state,
            TaskGraph("graph", (TaskNode("node"),)),
        )
        lease = await repository.claim(
            "graph",
            "node",
            tenant_id="tenant",
            owner="owner",
            lease_seconds=10,
        )
        await repository.fail(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.MODEL_CONFIG_INVALID.value,
            error_digest="a" * 64,
            execution_id="execution-failed",
        )

        nodes = await repository.list_nodes("graph", tenant_id="tenant")
        assert len(nodes) == 1
        assert nodes[0].status is TaskStatus.FAILED
        assert nodes[0].execution_id == "execution-failed"
        assert nodes[0].error_code == ErrorCode.MODEL_CONFIG_INVALID.value
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_generic_failed_task_node_keeps_execution_identity_empty() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-error-generic", tenant_id="tenant")
    try:
        repository = state.task.tasks
        await admit_graph(
            state,
            TaskGraph("graph", (TaskNode("node"),)),
        )
        lease = await repository.claim(
            "graph",
            "node",
            tenant_id="tenant",
            owner="owner",
            lease_seconds=10,
        )
        await repository.fail(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="b" * 64,
        )

        nodes = await repository.list_nodes("graph", tenant_id="tenant")
        assert len(nodes) == 1
        assert nodes[0].status is TaskStatus.FAILED
        assert nodes[0].execution_id is None
    finally:
        await state.close()
