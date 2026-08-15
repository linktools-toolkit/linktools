#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused evidence for the Temporal TaskGraph repair contract."""

from collections.abc import Mapping
from types import SimpleNamespace

import linktools.ai.temporal._activity as temporal_activity
import pytest
from linktools.ai.core import Principal, TaskStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    ExecutionRequest,
    RuntimeDomain,
    RuntimeObjectKeyFactory,
    RuntimeState,
)
from linktools.ai.runtime.state._repositories import _task_dependencies_from_sql
from linktools.ai.storage import InMemoryObjectStore, ObjectStore
from linktools.ai.task import (
    TaskDependencyResult,
    TaskGraph,
    TaskGraphRequest,
    TaskNode,
    TaskNodeView,
)
from linktools.ai.temporal import load_execution_request
from linktools.ai.temporal._activity import TaskActivity
from linktools.ai.temporal._request import put_execution_request, put_task_request
from linktools.ai.temporal.workflow import (
    ExecutionWorkflow,
    ExecutionWorkflowInput,
    ExecutionWorkflowResult,
    TaskWorkflow,
    TaskWorkflowInput,
)


def _task_request(graph_id: str = "graph") -> TaskGraphRequest:
    principal = Principal("principal", "tenant", "service")
    graph = TaskGraph(graph_id, (TaskNode("node", input={"kind": "test"}),))
    return TaskGraphRequest(graph, principal, idempotency_key=f"request-{graph_id}")


@pytest.fixture(autouse=True)
def temporal_activity_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        temporal_activity._temporal_activity,
        "info",
        lambda: SimpleNamespace(workflow_run_id="test-workflow-run"),
    )


class _ExecutionActivity:
    async def run(self, request: ExecutionWorkflowInput) -> ExecutionWorkflowResult:
        return ExecutionWorkflowResult(request.execution_id, "SUCCEEDED", None, 0)


class _TaskRunner:
    def __init__(self, failure: BaseException | None = None) -> None:
        self._failure = failure

    async def prepare(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> tuple[str, ExecutionRequest]:
        if self._failure is not None:
            raise self._failure
        return "binding", ExecutionRequest("prompt", principal, f"execution-{graph_id}")


class _UnavailableObjectStore:
    @property
    def store_id(self) -> str:
        return "requests"

    async def stat(self, key: str) -> None:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)


async def _task_activity(
    runner: _TaskRunner,
    graph_id: str = "graph",
) -> tuple[RuntimeState, TaskActivity, TaskWorkflowInput]:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="temporal-repair", tenant_id="tenant")
    request = _task_request(graph_id)
    await state.task.tasks.create_graph(request.graph, tenant_id="tenant")
    request_store = InMemoryObjectStore("requests")
    request_ref = await put_task_request(
        request_store,
        RuntimeObjectKeyFactory("temporal-repair"),
        request,
    )
    workflow_request = TaskWorkflowInput.from_request(
        request,
        request_ref=request_ref,
        worker_build="worker",
    )
    activity = TaskActivity.from_runtime(
        task_state=state.task,
        runner=runner,
        request_store=request_store,
        namespace="temporal-repair",
    )
    return state, activity, workflow_request


@pytest.mark.asyncio
async def test_public_execution_request_loader_uses_persisted_request() -> None:
    principal = Principal("principal", "tenant", "service")
    store = InMemoryObjectStore("requests")
    request = ExecutionRequest("prompt", principal, "execution-key")
    request_ref = await put_execution_request(
        store,
        RuntimeObjectKeyFactory("temporal-repair"),
        request,
    )
    workflow_result = await ExecutionWorkflow(_ExecutionActivity()).run(
        ExecutionWorkflowInput(
            "execution",
            "tenant",
            "binding",
            "bundle",
            request_ref,
            "worker",
        )
    )

    assert workflow_result.state is not None
    loaded = await load_execution_request(
        store,
        namespace="temporal-repair",
        state=workflow_result.state,
    )

    assert loaded == request


@pytest.mark.asyncio
async def test_public_execution_request_loader_rejects_invalid_namespace() -> None:
    workflow_result = await ExecutionWorkflow(_ExecutionActivity()).run(
        ExecutionWorkflowInput(
            "execution", "tenant", "binding", "bundle", "request", "worker"
        )
    )
    assert workflow_result.state is not None

    with pytest.raises(AIError) as error:
        await load_execution_request(
            InMemoryObjectStore("requests"),
            namespace=" ",
            state=workflow_result.state,
        )

    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_loader_preserves_store_availability_errors() -> None:
    request_ref = RuntimeObjectKeyFactory("temporal-repair").key(
        RuntimeDomain.TASK,
        "tenant",
        "0" * 64,
    )
    workflow_result = await ExecutionWorkflow(_ExecutionActivity()).run(
        ExecutionWorkflowInput(
            "execution", "tenant", "binding", "bundle", request_ref, "worker"
        )
    )
    assert workflow_result.state is not None

    with pytest.raises(AIError) as error:
        await load_execution_request(
            _UnavailableObjectStore(),
            namespace="temporal-repair",
            state=workflow_result.state,
        )

    assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [AIError(ErrorCode.STORAGE_UNAVAILABLE), RuntimeError("runner failed")],
    ids=["retryable-ai-error", "non-ai-error"],
)
async def test_prepare_propagates_retryable_and_non_ai_errors(
    failure: BaseException,
) -> None:
    state, activity, request = await _task_activity(_TaskRunner(failure))
    try:
        with pytest.raises(type(failure)):
            await activity.prepare(request, "node", {})

        nodes = await state.task.tasks.list_nodes("graph", tenant_id="tenant")
        assert nodes[0].status is TaskStatus.RUNNING
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_prepare_does_not_terminalize_object_store_write_failure(
    monkeypatch,
) -> None:
    state, activity, request = await _task_activity(_TaskRunner())

    async def fail_put(
        store: ObjectStore,
        key_factory: RuntimeObjectKeyFactory,
        execution_request: ExecutionRequest,
    ) -> str:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(
        "linktools.ai.temporal._task_operation.put_execution_request",
        fail_put,
    )
    try:
        with pytest.raises(AIError) as error:
            await activity.prepare(request, "node", {})

        nodes = await state.task.tasks.list_nodes("graph", tenant_id="tenant")
        assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert nodes[0].status is TaskStatus.RUNNING
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_nonretryable_prepare_persists_failure_and_returns_view() -> None:
    state, activity, request = await _task_activity(
        _TaskRunner(AIError(ErrorCode.PROMPT_TOO_LARGE))
    )
    try:
        workflow_result = await TaskWorkflow(activity).run(request)

        nodes = await state.task.tasks.list_nodes("graph", tenant_id="tenant")
        graph = await state.task.tasks.get_graph("graph", tenant_id="tenant")
        assert workflow_result.status == "FAILED"
        assert nodes[0].status is TaskStatus.FAILED
        assert nodes[0].error_code == ErrorCode.PROMPT_TOO_LARGE.value
        assert nodes[0].error_digest == canonical_sha256(
            {"type": "AIError", "code": ErrorCode.PROMPT_TOO_LARGE.value}
        )
        assert graph is not None
        assert graph.status is TaskStatus.FAILED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_workflow_treats_terminal_prepare_as_graph_failure() -> None:
    request = TaskWorkflowInput.from_request(
        _task_request(),
        request_ref="request",
        worker_build="worker",
    )

    class TerminalPrepareActivity:
        async def prepare(
            self,
            request: TaskWorkflowInput,
            node_id: str,
            dependency_results: Mapping[str, TaskDependencyResult],
        ) -> TaskNodeView:
            return TaskNodeView(
                request.graph_id,
                node_id,
                (),
                TaskStatus.FAILED,
                None,
                1,
                None,
                None,
                ErrorCode.PROMPT_TOO_LARGE.value,
                "a" * 64,
            )

    result = await TaskWorkflow(TerminalPrepareActivity()).run(request)

    assert result.status == "FAILED"
    assert result.completed_node_ids == ()


@pytest.mark.parametrize(
    "value",
    [None, {}, "node", 1, [""], ["node", 1], ["node", "node"]],
)
def test_sql_task_dependencies_reject_malformed_values(value: object) -> None:
    with pytest.raises(AIError) as error:
        _task_dependencies_from_sql(value)

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_sql_task_dependencies_preserve_valid_values() -> None:
    assert _task_dependencies_from_sql([" upstream "]) == (" upstream ",)
