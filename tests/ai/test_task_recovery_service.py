#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from linktools.ai.core import (
    OperationKind,
    OperationStatus,
    Principal,
    ResourceKind,
    TaskStatus,
    TenantAuthorizationPolicy,
    canonical_sha256,
    idempotency_key_digest,
)
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskService,
    RecoverGraphRequest,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
)


class _Launcher:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.cancelled: list[str] = []

    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
        self.started.append(launch.graph.graph_id)
        return TaskGraphHandle(launch.graph.graph_id, f"test:{launch.graph.graph_id}")

    async def cancel(self, launch: TaskGraphLaunch) -> TaskGraphView:
        self.cancelled.append(launch.graph.graph_id)
        return TaskGraphView(
            launch.graph.graph_id,
            TaskStatus.RECOVERY_REQUIRED,
            launch.graph.nodes,
        )


def _request(graph_id: str) -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph(graph_id, (TaskNode("node"),)),
        Principal("tester", "tenant"),
        f"submit:{graph_id}",
    )


async def _recovery_graph(state: RuntimeState, graph_id: str) -> TaskGraphRequest:
    request = _request(graph_id)
    await state.task.admissions.admit(
        TaskGraphAdmission.from_request(request),
        request.graph,
    )
    lease = await state.task.tasks.claim(
        graph_id,
        "node",
        tenant_id="tenant",
        owner="worker",
        lease_seconds=30,
    )
    await state.task.tasks.mark_recovery_required(
        lease,
        tenant_id="tenant",
        error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
        error_digest=canonical_sha256(
            {"graph_id": graph_id, "node_id": "node", "code": ErrorCode.TOOL_EFFECT_UNKNOWN.value}
        ),
        execution_id="execution",
    )
    return request


@pytest.mark.asyncio
async def test_wait_and_stream_end_at_durable_recovery_boundary() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "boundary")
        service = DefaultTaskService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            _Launcher(),
        )

        result = await service.wait_graph(
            "boundary",
            principal=request.principal,
            timeout_seconds=1,
        )
        events = [
            event
            async for event in service.stream_graph_events(
                "boundary",
                principal=request.principal,
            )
        ]

        assert result.status is TaskStatus.RECOVERY_REQUIRED
        assert result.execution_ids == ("execution",)
        assert result.node_results[0].status is TaskStatus.RECOVERY_REQUIRED
        assert events[-1].node_id is None
        assert events[-1].status is TaskStatus.RECOVERY_REQUIRED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_explicit_recovery_rearms_original_graph() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "resume")
        launcher = _Launcher()
        service = DefaultTaskService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )

        result = await service.recover_graph(
            "resume",
            RecoverGraphRequest(request.principal, "recover:resume"),
        )

        assert result.status is TaskStatus.PENDING
        assert launcher.started == ["resume"]
        assert result.node_results[0].status is TaskStatus.READY
        assert result.node_results[0].execution_id == "execution"
        operation = await state.task.operations.get(
            idempotency_key_digest("recover:resume"),
            tenant_id="tenant",
        )
        assert operation is not None
        assert operation.operation_kind is OperationKind.TASK_RECOVER
        assert operation.status is OperationStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancel_intent_stays_pending_until_recovery_settles_it() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-service-recovery", tenant_id="tenant")
    try:
        request = await _recovery_graph(state, "cancel")
        launcher = _Launcher()
        service = DefaultTaskService(
            state.task,
            TenantAuthorizationPolicy("tenant"),
            launcher,
        )

        deferred = await service.cancel_graph(
            "cancel",
            CancelGraphRequest(request.principal, "cancel:intent"),
        )
        cancel_operation = await state.task.operations.get(
            idempotency_key_digest("cancel:intent"),
            tenant_id="tenant",
        )
        pending = await state.task.operations.list_pending(
            ResourceKind.TASK_GRAPH,
            "cancel",
            tenant_id="tenant",
            limit=10,
        )

        assert deferred.status is TaskStatus.RECOVERY_REQUIRED
        assert cancel_operation is not None
        assert cancel_operation.operation_kind is OperationKind.TASK_CANCEL
        assert cancel_operation.status is OperationStatus.RUNNING
        assert any(item.operation_id == cancel_operation.operation_id for item in pending)

        result = await service.recover_graph(
            "cancel",
            RecoverGraphRequest(request.principal, "recover:cancel"),
        )
        settled_cancel = await state.task.operations.get(
            cancel_operation.operation_id,
            tenant_id="tenant",
        )

        assert result.status is TaskStatus.CANCELLED
        assert settled_cancel is not None
        assert settled_cancel.status is OperationStatus.SUCCEEDED
        assert launcher.started == []
        assert launcher.cancelled == ["cancel", "cancel"]
    finally:
        await state.close()
