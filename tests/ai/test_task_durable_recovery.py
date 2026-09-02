#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Task admission and recovery contracts."""

from datetime import datetime, timezone

import pytest
from linktools.ai.migrate import provision_database
from sqlalchemy.ext.asyncio import create_async_engine
from linktools.ai.core import (
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    Principal,
    ResourceKind,
    TaskStatus,
    canonical_sha256,
    principal_identity_payload,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeState, RuntimeStatePlan, RuntimeStateRoute
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._store import OperationQuery, stream_digest
from linktools.ai.task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskNode,
)


def _request(
    graph_id: str,
    *,
    idempotency_key: str | None = None,
    nodes: tuple[TaskNode, ...] | None = None,
) -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph(graph_id, (TaskNode("root"),) if nodes is None else nodes),
        Principal("tester", "tenant"),
        idempotency_key or f"submit:{graph_id}",
        TaskGraphLimits(),
    )


def _expected_request_digest(request: TaskGraphRequest) -> str:
    return canonical_sha256(
        {
            "principal": principal_identity_payload(request.principal),
            "graph_id": request.graph.graph_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "dependencies": sorted(node.dependencies),
                    "input": node.input,
                    "budget_cost": node.budget_cost,
                }
                for node in sorted(request.graph.nodes, key=lambda item: item.node_id)
            ],
            "limits": {
                "max_nodes": request.limits.max_nodes,
                "max_depth": request.limits.max_depth,
                "max_budget": request.limits.max_budget,
                "max_concurrency": request.limits.max_concurrency,
            },
        }
    )


def _submit_result_digest(graph: TaskGraph) -> str:
    status = TaskStatus.SUCCEEDED if not graph.nodes else TaskStatus.PENDING
    return canonical_sha256({"graph_id": graph.graph_id, "status": status.value})


def _partial_operation(
    admission: TaskGraphAdmission,
    graph: TaskGraph,
    status: OperationStatus,
) -> OperationLedgerInput:
    now = datetime.now(timezone.utc)
    terminal = status is OperationStatus.SUCCEEDED
    return OperationLedgerInput(
        admission.operation_id,
        "tenant",
        ResourceKind.TASK_GRAPH,
        graph.graph_id,
        None,
        OperationKind.TASK_NODE,
        status,
        admission.request_digest,
        graph.graph_id if terminal else None,
        _submit_result_digest(graph) if terminal else None,
        None,
        False,
        now,
        now,
    )


def test_task_admission_freezes_legacy_request_digest_without_raw_key() -> None:
    request = _request("digest", idempotency_key="raw-secret-idempotency-key")
    admission = TaskGraphAdmission.from_request(request)

    assert admission.request_digest == _expected_request_digest(request)
    assert admission.operation_id != request.idempotency_key
    assert not hasattr(admission, "idempotency_key")
    assert request.idempotency_key not in repr(admission)


def test_task_admission_rejects_unsupported_version() -> None:
    request = _request("unsupported")
    admission = TaskGraphAdmission.from_request(request)
    unsupported = TaskGraphAdmission(
        2,
        admission.graph_id,
        admission.principal,
        admission.limits,
        admission.operation_id,
        admission.request_digest,
    )

    with pytest.raises(AIError) as raised:
        unsupported.bind(request.graph)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
async def test_memory_admission_is_atomic_replay_safe_and_recoverable() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-test", tenant_id="tenant")
    try:
        request = _request("memory")
        admission = TaskGraphAdmission.from_request(request)

        first = await state.task.admissions.admit(admission, request.graph)
        replay = await state.task.admissions.admit(admission, request.graph)
        operation = await state.task.operations.get(
            admission.operation_id,
            tenant_id="tenant",
        )
        page = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

        assert first == replay
        assert first.status is TaskStatus.PENDING
        assert operation is not None
        assert operation.status is OperationStatus.SUCCEEDED
        assert operation.result_ref == request.graph.graph_id
        assert operation.compactable is False
        await state.task.operations.compact_terminal(
            ResourceKind.TASK_GRAPH,
            request.graph.graph_id,
            tenant_id="tenant",
            through_sequence=operation.sequence,
        )
        assert (
            await state.task.operations.get(
                admission.operation_id,
                tenant_id="tenant",
            )
            == operation
        )
        assert page.items == (admission.bind(request.graph),)

        changed = _request(
            "memory",
            idempotency_key=request.idempotency_key,
            nodes=(TaskNode("root", input={"changed": True}),),
        )
        with pytest.raises(AIError) as raised:
            await state.task.admissions.admit(
                TaskGraphAdmission.from_request(changed),
                changed.graph,
            )
        assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation_status",
    (
        OperationStatus.PENDING,
        OperationStatus.RUNNING,
        OperationStatus.EFFECT_UNKNOWN,
        OperationStatus.SUCCEEDED,
    ),
)
async def test_partial_admission_operation_without_graph_fails_closed(
    operation_status: OperationStatus,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-test", tenant_id="tenant")
    try:
        request = _request(f"partial-operation-{operation_status.value.lower()}")
        admission = TaskGraphAdmission.from_request(request)
        await state.task.operations.append(
            _partial_operation(admission, request.graph, operation_status)
        )

        with pytest.raises(AIError) as raised:
            await state.task.admissions.admit(admission, request.graph)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_existing_admitted_graph_rejects_different_operation_as_storage_conflict() -> (
    None
):
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-test", tenant_id="tenant")
    try:
        first = _request("occupied", idempotency_key="submit:occupied:first")
        second = _request("occupied", idempotency_key="submit:occupied:second")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(first),
            first.graph,
        )

        with pytest.raises(AIError) as raised:
            await state.task.admissions.admit(
                TaskGraphAdmission.from_request(second),
                second.graph,
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_corrupt_occupied_graph_without_its_admission_operation_fails_closed(
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-test", tenant_id="tenant")
    try:
        first = _request(
            "occupied-corrupt",
            idempotency_key="submit:occupied:original",
        )
        second = _request(
            "occupied-corrupt",
            idempotency_key="submit:occupied:second",
        )
        admission = TaskGraphAdmission.from_request(first)
        await state.task.admissions.admit(admission, first.graph)
        operation = await state.task.operations.get(
            admission.operation_id,
            tenant_id="tenant",
        )
        assert operation is not None
        operation_stream = stream_digest(
            "task-test",
            "tenant",
            RuntimeDomain.TASK.value,
            "operation",
            [ResourceKind.TASK_GRAPH.value, first.graph.graph_id],
        )
        deleted = await state.task.operations.state_store.mutate(
            lambda transaction: transaction.delete_operations(
                OperationQuery(
                    stream_digest=operation_stream,
                    states=frozenset({OperationStatus.SUCCEEDED.value}),
                    through_sequence=operation.sequence,
                )
            )
        )
        assert len(deleted) == 1

        with pytest.raises(AIError) as recovery_error:
            await state.task.admissions.list_recoverable_page(
                cursor=None,
                limit=128,
            )
        assert recovery_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        with pytest.raises(AIError) as raised:
            await state.task.admissions.admit(
                TaskGraphAdmission.from_request(second),
                second.graph,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_durable_admission_survives_reopen_and_remains_recoverable(
    tmp_path,
    backend: str,
) -> None:
    path = tmp_path / ("runtime" if backend == "filesystem" else "runtime.db")
    request = _request(f"reopen-{backend}")
    admission = TaskGraphAdmission.from_request(request)
    engine = None
    if backend == "filesystem":
        create_state = lambda: RuntimeState.filesystem(path)
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        await provision_database(engine)
        create_state = lambda: RuntimeState.sql(engine)

    try:
        state = create_state()
        await state.initialize(namespace="task-test", tenant_id="tenant")
        try:
            view = await state.task.admissions.admit(admission, request.graph)
            assert view.status is TaskStatus.PENDING
        finally:
            await state.close()

        reopened = create_state()
        await reopened.initialize(namespace="task-test", tenant_id="tenant")
        try:
            page = await reopened.task.admissions.list_recoverable_page(
                cursor=None,
                limit=128,
            )
            assert page.items == (admission.bind(request.graph),)
            operation = await reopened.task.operations.get(
                admission.operation_id,
                tenant_id="tenant",
            )
            assert operation is not None
            assert operation.status is OperationStatus.SUCCEEDED
        finally:
            await reopened.close()
    finally:
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_recovery_discovery_uses_bounded_keyset_pages() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-test", tenant_id="tenant")
    try:
        for index in range(129):
            request = _request(f"page-{index:03d}")
            await state.task.admissions.admit(
                TaskGraphAdmission.from_request(request),
                request.graph,
            )

        first = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )
        assert len(first.items) == 128
        assert first.next_cursor is not None

        second = await state.task.admissions.list_recoverable_page(
            cursor=first.next_cursor,
            limit=128,
        )
        assert len(second.items) == 1
        assert second.next_cursor is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_empty_graph_is_terminal_and_excluded_from_recovery() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-test", tenant_id="tenant")
    try:
        request = _request("empty", nodes=())
        admission = TaskGraphAdmission.from_request(request)
        view = await state.task.admissions.admit(admission, request.graph)
        page = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

        assert view.status is TaskStatus.SUCCEEDED
        assert page.items == ()
    finally:
        await state.close()


def test_durable_task_requires_durable_execution_and_recovery(tmp_path) -> None:
    task = RuntimeStateRoute.filesystem(tmp_path / "task")
    recovery = RuntimeStateRoute.filesystem(tmp_path / "recovery")
    execution = RuntimeStateRoute.filesystem(tmp_path / "execution")

    with pytest.raises(ValueError, match="durable task requires durable execution"):
        RuntimeState.from_plan(
            RuntimeStatePlan(
                task=task,
                recovery=recovery,
                execution=RuntimeStateRoute.transient(),
            )
        )

    with pytest.raises(ValueError, match="durable task requires durable recovery"):
        RuntimeState.from_plan(
            RuntimeStatePlan(
                task=task,
                execution=execution,
                recovery=RuntimeStateRoute.transient(),
            )
        )
