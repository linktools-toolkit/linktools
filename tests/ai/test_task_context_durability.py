#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable TaskGraph correlation and cancellation provenance regressions."""

import pytest
from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskGraphService,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
)


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _CaptureLauncher:
    def __init__(self) -> None:
        self.started: TaskGraphLaunch | None = None
        self.cancelled: TaskGraphLaunch | None = None

    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
        self.started = launch
        return TaskGraphHandle(
            launch.graph_id,
            f"capture:{launch.principal.tenant_id}:{launch.graph_id}",
        )

    async def cancel(self, launch: TaskGraphLaunch) -> TaskGraphView:
        self.cancelled = launch
        return TaskGraphView(launch.graph_id, TaskStatus.CANCELLED, ())


def _request(
    graph: TaskGraph,
    *,
    correlation: dict[str, str | int],
    principal: Principal | None = None,
) -> TaskGraphRequest:
    return TaskGraphRequest(
        graph,
        principal or Principal("submitter", "tenant"),
        "task-correlation-request-0001",
        correlation=correlation,
    )


@pytest.mark.asyncio
async def test_task_admission_correlation_is_durable_but_not_semantic_identity() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-correlation-durable", tenant_id="tenant")
    try:
        graph = TaskGraph("task-correlation-durable", (TaskNode("node"),))
        first = TaskGraphAdmission.from_request(
            _request(graph, correlation={"trace_id": "trace-a", "attempt": 7})
        )
        second = TaskGraphAdmission.from_request(
            _request(graph, correlation={"trace_id": "trace-b", "attempt": 8})
        )

        assert first.operation_id == second.operation_id
        assert first.initial_request_digest == second.initial_request_digest
        assert first.correlation != second.correlation

        await state.task.admissions.admit(first, graph)
        stored = await state.task.admissions.get(graph.graph_id, tenant_id="tenant")

        assert stored == first
        assert dict(stored.correlation) == {"attempt": 7, "trace_id": "trace-a"}
        assert await state.task.admissions.get(graph.graph_id, tenant_id="other") is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_admission_correlation_drift_conflicts() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-correlation-conflict", tenant_id="tenant")
    try:
        graph = TaskGraph("task-correlation-conflict", (TaskNode("node"),))
        original = TaskGraphAdmission.from_request(
            _request(graph, correlation={"trace_id": "trace-a"})
        )
        drifted = TaskGraphAdmission.from_request(
            _request(graph, correlation={"trace_id": "trace-b"})
        )
        assert original.initial_request_digest == drifted.initial_request_digest

        await state.task.admissions.admit(original, graph)
        with pytest.raises(AIError) as raised:
            await state.task.admissions.admit(drifted, graph)

        assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        stored = await state.task.admissions.get(graph.graph_id, tenant_id="tenant")
        assert stored == original
        assert dict(stored.correlation) == {"trace_id": "trace-a"}
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_service_replay_rejects_correlation_drift() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-correlation-service-replay", tenant_id="tenant")
    launcher = _CaptureLauncher()
    try:
        graph = TaskGraph("task-correlation-service-replay", (TaskNode("node"),))
        original = TaskGraphAdmission.from_request(
            _request(graph, correlation={"trace_id": "trace-a"})
        )
        await state.task.admissions.admit(original, graph)
        service = DefaultTaskGraphService(
            state.task,
            _AllowAuthorization(),
            launcher,
        )

        with pytest.raises(AIError) as raised:
            await service.start(
                _request(graph, correlation={"trace_id": "trace-b"})
            )

        assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert launcher.started is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancel_cleanup_restores_durable_submission_principal_and_correlation() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-correlation-cancel", tenant_id="tenant")
    launcher = _CaptureLauncher()
    try:
        graph = TaskGraph("task-correlation-cancel", (TaskNode("node"),))
        submitter = Principal("submitter", "tenant")
        admission = TaskGraphAdmission.from_request(
            _request(
                graph,
                principal=submitter,
                correlation={"trace_id": "trace-cancel", "attempt": 3},
            )
        )
        await state.task.admissions.admit(admission, graph)
        service = DefaultTaskGraphService(
            state.task,
            _AllowAuthorization(),
            launcher,
        )

        view = await service.cancel(
            graph.graph_id,
            CancelGraphRequest(
                Principal("operator", "tenant"),
                "task-correlation-cancel-0001",
            ),
        )

        assert view.status is TaskStatus.CANCELLED
        assert launcher.cancelled is not None
        assert launcher.cancelled.graph_id == graph.graph_id
        assert launcher.cancelled.principal == submitter
        assert dict(launcher.cancelled.correlation) == {
            "attempt": 3,
            "trace_id": "trace-cancel",
        }
    finally:
        await state.close()


def test_task_admission_empty_correlation_uses_current_wire_shape() -> None:
    graph = TaskGraph("task-correlation-wire", (TaskNode("node"),))
    admission = TaskGraphAdmission.from_request(_request(graph, correlation={}))

    payload = _encode_persisted_domain(admission)
    assert isinstance(payload, dict)
    assert payload["$dataclass"] == "task_graph_admission"
    fields = payload["fields"]
    assert isinstance(fields, dict)
    assert "correlation" in fields

    decoded = _decode_enveloped_domain(
        encode_envelope({"type": "task_graph_admission", "payload": payload}),
        TaskGraphAdmission,
    )
    assert decoded == admission
    assert dict(decoded.correlation) == {}

    with_correlation = TaskGraphAdmission.from_request(
        _request(graph, correlation={"trace_id": "trace-wire"})
    )
    persisted = _encode_persisted_domain(with_correlation)
    assert isinstance(persisted, dict)
    persisted_fields = persisted["fields"]
    assert isinstance(persisted_fields, dict)
    assert "correlation" in persisted_fields
