#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regressions for TaskGraph and ToolOperation optimistic CAS convergence."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.core import (
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    TenantAuthorizationPolicy,
    ToolOperationStatus,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._repositories import ToolRepositoryImpl
from linktools.ai.runtime.state._tool_repository import DurableToolRepositoryImpl
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy, StoredPayload
from linktools.ai.task import (
    TaskDependencyResult,
    TaskGraph,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
)
from linktools.ai.task._api import open_local_task_api
from linktools.ai.task._local import TaskNodeRunResult
from linktools.ai.task._service_impl import DefaultTaskService
from sqlalchemy.ext.asyncio import create_async_engine


class _DigestRunner:
    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> TaskNodeRunResult:
        del principal, dependency_results
        return TaskNodeRunResult(
            canonical_sha256({"graph_id": graph_id, "node_id": node.node_id})
        )

    async def cancel(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> None:
        del node, graph_id, principal, dependency_results


@pytest.mark.asyncio
async def test_sqlite_task_graph_dependency_and_parallel_runs_are_stable(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.sqlite'}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="task-cas", tenant_id="tenant")
    principal = Principal("tester", "tenant")
    try:
        async with open_local_task_api(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            runner=_DigestRunner(),
            owner="task-cas-runner",
        ) as api:
            for index in range(20):
                graph = TaskGraph(
                    f"serial-{index}",
                    (
                        TaskNode("a"),
                        TaskNode("b", ("a",)),
                    ),
                )
                result = await api.run_graph_and_wait(
                    TaskGraphRequest(
                        graph,
                        principal,
                        f"submit:serial:{index}",
                        TaskGraphLimits(max_concurrency=1),
                    ),
                    timeout_seconds=10,
                )
                assert result.status is TaskStatus.SUCCEEDED
                assert all(
                    node.status is TaskStatus.SUCCEEDED
                    for node in result.node_results
                )

            for index in range(20):
                graph = TaskGraph(
                    f"parallel-{index}",
                    (
                        TaskNode("a"),
                        TaskNode("b"),
                        TaskNode("c"),
                        TaskNode("join", ("a", "b", "c")),
                    ),
                )
                result = await api.run_graph_and_wait(
                    TaskGraphRequest(
                        graph,
                        principal,
                        f"submit:parallel:{index}",
                        TaskGraphLimits(max_concurrency=3),
                    ),
                    timeout_seconds=10,
                )
                assert result.status is TaskStatus.SUCCEEDED
                assert all(
                    node.status is TaskStatus.SUCCEEDED
                    for node in result.node_results
                )
    finally:
        await state.close()
        await engine.dispose()


class _ReadOnlyTaskRepository:
    def __init__(self, view: TaskGraphView) -> None:
        self.view = view
        self.reconcile_calls = 0

    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef:
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView:
        del graph_id, tenant_id
        return self.view

    async def reconcile_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView:
        del graph_id, tenant_id
        self.reconcile_calls += 1
        raise AssertionError("observer must not reconcile Task state")

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[object, ...]:
        del graph_id, tenant_id
        return ()


@pytest.mark.asyncio
async def test_task_inspect_and_wait_are_read_only() -> None:
    tasks = _ReadOnlyTaskRepository(
        TaskGraphView("observed", TaskStatus.SUCCEEDED, ())
    )
    service = DefaultTaskService(
        SimpleNamespace(tasks=tasks),
        TenantAuthorizationPolicy("tenant"),
    )
    principal = Principal("tester", "tenant")

    inspected = await service.inspect_graph("observed", principal=principal)
    waited = await service.wait_graph("observed", principal=principal)

    assert inspected.status is TaskStatus.SUCCEEDED
    assert waited.status is TaskStatus.SUCCEEDED
    assert tasks.reconcile_calls == 0


def _tool_record(
    *,
    status: ToolOperationStatus,
    owner: str = "tool-owner",
    fence: int = 1,
    result_payload: StoredPayload | None = None,
    error_code: str | None = None,
    error_payload: StoredPayload | None = None,
) -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        "tool-op",
        "tenant",
        "step-run",
        "tool-call",
        "idempotency",
        "tool",
        canonical_sha256({"arguments": True}),
        canonical_sha256({"binding": True}),
        True,
        status,
        owner,
        fence,
        None if status is not ToolOperationStatus.CLAIMED else now + timedelta(seconds=60),
        error_code,
        now,
        now,
        result_payload=result_payload,
        error_payload=error_payload,
    )


@pytest.mark.asyncio
async def test_durable_tool_terminal_retries_only_storage_conflict(monkeypatch) -> None:
    payload = StoredPayload.inline_bytes(b"result")
    committed = _tool_record(
        status=ToolOperationStatus.COMPLETED,
        result_payload=payload,
    )
    attempts = 0

    async def complete_payload(
        self: ToolRepositoryImpl,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord:
        nonlocal attempts
        del self, tool_operation_id, tenant_id, owner, fence, result_payload
        attempts += 1
        if attempts == 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return committed

    monkeypatch.setattr(ToolRepositoryImpl, "complete_payload", complete_payload)
    repository = object.__new__(DurableToolRepositoryImpl)

    result = await repository.complete_payload(
        "tool-op",
        tenant_id="tenant",
        owner="tool-owner",
        fence=1,
        result_payload=payload,
    )

    assert result == committed
    assert attempts == 2


@pytest.mark.asyncio
async def test_durable_tool_terminal_preserves_fence_conflict(monkeypatch) -> None:
    payload = StoredPayload.inline_bytes(b"result")
    attempts = 0

    async def complete_payload(
        self: ToolRepositoryImpl,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord:
        nonlocal attempts
        del self, tool_operation_id, tenant_id, owner, fence, result_payload
        attempts += 1
        if attempts == 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)

    monkeypatch.setattr(ToolRepositoryImpl, "complete_payload", complete_payload)
    repository = object.__new__(DurableToolRepositoryImpl)

    with pytest.raises(AIError) as raised:
        await repository.complete_payload(
            "tool-op",
            tenant_id="tenant",
            owner="tool-owner",
            fence=1,
            result_payload=payload,
        )

    assert raised.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
    assert attempts == 2


class _ToolReadbackRepository:
    def __init__(self, record: ToolOperationRecord) -> None:
        self.record = record

    async def get_operation(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord:
        del tool_operation_id, tenant_id
        return self.record


@pytest.mark.asyncio
async def test_tool_terminal_readback_preserves_result_conflict() -> None:
    expected = StoredPayload.inline_bytes(b"expected")
    actual = StoredPayload.inline_bytes(b"actual")
    bridge = RuntimeToolOperationBridge(
        _ToolReadbackRepository(
            _tool_record(
                status=ToolOperationStatus.COMPLETED,
                result_payload=actual,
            )
        ),
        InMemoryObjectStore(),
        namespace="tool-cas",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="step-run",
        binding_digest=canonical_sha256({"binding": True}),
        owner="tool-owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )
    decision = SimpleNamespace(operation_id="tool-op", owner="tool-owner", fence=1)

    async def lost_response() -> ToolOperationRecord:
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    with pytest.raises(AIError) as raised:
        await bridge._finish_with_readback(
            lost_response,
            decision,
            expected_status=ToolOperationStatus.COMPLETED,
            expected_payload=expected,
        )

    assert raised.value.code is ErrorCode.TOOL_RESULT_CONFLICT


@pytest.mark.asyncio
async def test_tool_terminal_readback_requires_exact_owner_and_fence() -> None:
    payload = StoredPayload.inline_bytes(b"result")
    bridge = RuntimeToolOperationBridge(
        _ToolReadbackRepository(
            _tool_record(
                status=ToolOperationStatus.COMPLETED,
                owner="other-owner",
                fence=2,
                result_payload=payload,
            )
        ),
        InMemoryObjectStore(),
        namespace="tool-cas",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="step-run",
        binding_digest=canonical_sha256({"binding": True}),
        owner="tool-owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )
    decision = SimpleNamespace(operation_id="tool-op", owner="tool-owner", fence=1)

    async def lost_response() -> ToolOperationRecord:
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    with pytest.raises(AIError) as raised:
        await bridge._finish_with_readback(
            lost_response,
            decision,
            expected_status=ToolOperationStatus.COMPLETED,
            expected_payload=payload,
        )

    assert raised.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
