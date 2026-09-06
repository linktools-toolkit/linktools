#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable TaskGraph correlation and cancellation provenance regressions."""

import pytest
from linktools.ai.core import Principal, TaskStatus
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.task import (
    CancelGraphRequest,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
)
from linktools.ai.task._service_impl import DefaultTaskService


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _CaptureLauncher:
    def __init__(self) -> None:
        self.cancelled: TaskGraphLaunch | None = None

    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle:
        raise AssertionError(f"unexpected scheduler start for {launch.graph.graph_id}")

    async def cancel(self, launch: TaskGraphLaunch) -> TaskGraphView:
        self.cancelled = launch
        return TaskGraphView(launch.graph.graph_id, TaskStatus.CANCELLED, launch.graph.nodes)


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
        assert first.request_digest == second.request_digest
        assert first.correlation != second.correlation

        await state.task.admissions.admit(first, graph)
        stored = await state.task.admissions.get(graph.graph_id, tenant_id="tenant")

        assert stored == first
        assert dict(stored.correlation) == {"attempt": 7, "trace_id": "trace-a"}
        assert await state.task.admissions.get(graph.graph_id, tenant_id="other") is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_admission_correlation_drift_reuses_first_durable_value() -> None:
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
        assert original.request_digest == drifted.request_digest

        first_view = await state.task.admissions.admit(original, graph)
        replay_view = await state.task.admissions.admit(drifted, graph)

        assert replay_view == first_view
        stored = await state.task.admissions.get(graph.graph_id, tenant_id="tenant")
        assert stored == original
        assert dict(stored.correlation) == {"trace_id": "trace-a"}
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
        service = DefaultTaskService(
            state.task,
            _AllowAuthorization(),
            launcher,
        )

        view = await service.cancel_graph(
            graph.graph_id,
            CancelGraphRequest(
                Principal("operator", "tenant"),
                "task-correlation-cancel-0001",
            ),
        )

        assert view.status is TaskStatus.CANCELLED
        assert launcher.cancelled is not None
        assert launcher.cancelled.graph == graph
        assert launcher.cancelled.principal == submitter
        assert dict(launcher.cancelled.correlation) == {
            "attempt": 3,
            "trace_id": "trace-cancel",
        }
    finally:
        await state.close()


def test_task_admission_v1_empty_correlation_uses_canonical_wire() -> None:
    graph = TaskGraph("task-correlation-wire", (TaskNode("node"),))
    admission = TaskGraphAdmission.from_request(_request(graph, correlation={}))

    payload = _encode_persisted_domain(admission)
    assert isinstance(payload, dict)
    assert payload["$dataclass"] == "task_graph_admission"
    fields = payload["fields"]
    assert isinstance(fields, dict)
    assert "correlation" not in fields

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
