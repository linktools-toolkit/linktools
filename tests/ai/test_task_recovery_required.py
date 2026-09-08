#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from linktools.ai.core import Principal, TaskStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphRequest,
    TaskNode,
)


def _request(graph_id: str) -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph(
            graph_id,
            (
                TaskNode("root"),
                TaskNode("dependent", ("root",)),
            ),
        ),
        Principal("tester", "tenant"),
        f"submit:{graph_id}",
    )


def _recovery_digest(code: ErrorCode) -> str:
    return canonical_sha256({"code": code.value, "phase": "test"})


@pytest.mark.asyncio
async def test_task_recovery_required_is_durable_and_not_dependency_failure() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-recovery", tenant_id="tenant")
    try:
        request = _request("graph")
        admission = TaskGraphAdmission.from_request(request)
        await state.task.admissions.admit(admission, request.graph)
        lease = await state.task.tasks.claim(
            "graph",
            "root",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await state.task.tasks.bind_execution(
            lease,
            tenant_id="tenant",
            execution_id="execution",
        )

        recovered = await state.task.tasks.mark_recovery_required(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            error_digest=_recovery_digest(ErrorCode.TOOL_EFFECT_UNKNOWN),
            execution_id="execution",
        )
        snapshot = await state.task.tasks.snapshot_graph(
            "graph",
            tenant_id="tenant",
        )
        page = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

        assert recovered.status is TaskStatus.RECOVERY_REQUIRED
        assert recovered.owner is None
        assert recovered.lease_expires_at is None
        assert recovered.fence == lease.fence
        assert recovered.execution_id == "execution"
        assert snapshot is not None
        assert snapshot.status is TaskStatus.RECOVERY_REQUIRED
        by_id = {node.node_id: node for node in snapshot.node_states}
        assert by_id["root"].status is TaskStatus.RECOVERY_REQUIRED
        assert by_id["dependent"].status is TaskStatus.PENDING
        assert page.items == ()

        with pytest.raises(AIError) as claim_error:
            await state.task.tasks.claim(
                "graph",
                "dependent",
                tenant_id="tenant",
                owner="other",
                lease_seconds=30,
            )
        assert claim_error.value.code is ErrorCode.TASK_NOT_READY
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_recovery_preserves_lineage_and_advances_fence_on_reclaim() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-recovery", tenant_id="tenant")
    try:
        request = _request("recover")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        first = await state.task.tasks.claim(
            "recover",
            "root",
            tenant_id="tenant",
            owner="worker-1",
            lease_seconds=30,
        )
        await state.task.tasks.bind_execution(
            first,
            tenant_id="tenant",
            execution_id="execution",
        )
        await state.task.tasks.mark_recovery_required(
            first,
            tenant_id="tenant",
            error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value,
            error_digest=_recovery_digest(ErrorCode.STORAGE_RECOVERY_REQUIRED),
            execution_id="execution",
        )

        view = await state.task.tasks.recover_graph(
            "recover",
            tenant_id="tenant",
        )
        snapshot = await state.task.tasks.snapshot_graph(
            "recover",
            tenant_id="tenant",
        )
        assert view.status is TaskStatus.PENDING
        assert snapshot is not None
        root = {node.node_id: node for node in snapshot.node_states}["root"]
        assert root.status is TaskStatus.READY
        assert root.fence == first.fence
        assert root.execution_id == "execution"
        assert root.error_code is None
        assert root.error_digest is None

        second = await state.task.tasks.claim(
            "recover",
            "root",
            tenant_id="tenant",
            owner="worker-2",
            lease_seconds=30,
        )
        assert second.fence == first.fence + 1
        bound = await state.task.tasks.bind_execution(
            second,
            tenant_id="tenant",
            execution_id="execution",
        )
        assert bound.execution_id == "execution"
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancel_does_not_overwrite_recovery_required() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-recovery", tenant_id="tenant")
    try:
        request = _request("cancel")
        await state.task.admissions.admit(
            TaskGraphAdmission.from_request(request),
            request.graph,
        )
        lease = await state.task.tasks.claim(
            "cancel",
            "root",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await state.task.tasks.mark_recovery_required(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.EXECUTION_START_UNKNOWN.value,
            error_digest=_recovery_digest(ErrorCode.EXECUTION_START_UNKNOWN),
        )

        cancelled = await state.task.tasks.cancel_graph(
            "cancel",
            tenant_id="tenant",
        )
        assert cancelled.status is TaskStatus.RECOVERY_REQUIRED
        snapshot = await state.task.tasks.snapshot_graph(
            "cancel",
            tenant_id="tenant",
        )
        assert snapshot is not None
        by_id = {node.node_id: node for node in snapshot.node_states}
        assert by_id["root"].status is TaskStatus.RECOVERY_REQUIRED
        assert by_id["dependent"].status is TaskStatus.CANCELLED

        final = await state.task.tasks.recover_graph(
            "cancel",
            tenant_id="tenant",
            cancel_requested=True,
        )
        assert final.status is TaskStatus.CANCELLED
        terminal = await state.task.tasks.snapshot_graph(
            "cancel",
            tenant_id="tenant",
        )
        assert terminal is not None
        assert all(
            node.status is TaskStatus.CANCELLED
            for node in terminal.node_states
        )
    finally:
        await state.close()
