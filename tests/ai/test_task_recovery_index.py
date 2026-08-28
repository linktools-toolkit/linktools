#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task recovery-index convergence contracts."""

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._repositories import TaskAdmissionRepositoryImpl
from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphRequest, TaskNode


def _request(graph_id: str) -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph(graph_id, (TaskNode("root"),)),
        Principal("tester", "tenant"),
        f"submit:{graph_id}",
    )


@pytest.mark.asyncio
async def test_terminal_graph_is_removed_from_recovery_index_after_reconcile() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-recovery-index", tenant_id="tenant")
    try:
        request = _request("terminal-index")
        admission = TaskGraphAdmission.from_request(request)
        await state.task.admissions.admit(admission, request.graph)

        lease = await state.task.tasks.claim(
            request.graph.graph_id,
            "root",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=60,
        )
        await state.task.tasks.complete(
            lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="0" * 64,
        )

        view = await state.task.tasks.reconcile_graph(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        recoverable = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

        assert view.status is TaskStatus.SUCCEEDED
        assert recoverable.items == ()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_exact_submit_replay_repairs_running_and_terminal_projections() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-replay-projection", tenant_id="tenant")
    try:
        request = _request("replay-projection")
        admission = TaskGraphAdmission.from_request(request)
        await state.task.admissions.admit(admission, request.graph)

        lease = await state.task.tasks.claim(
            request.graph.graph_id,
            "root",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=60,
        )
        running = await state.task.admissions.admit(admission, request.graph)
        assert running.status is TaskStatus.RUNNING

        await state.task.tasks.complete(
            lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="0" * 64,
        )
        terminal = await state.task.admissions.admit(admission, request.graph)
        recoverable = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

        assert terminal.status is TaskStatus.SUCCEEDED
        assert recoverable.items == ()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_previous_graph_scoped_recovery_layout_remains_recoverable() -> None:
    namespace = "task-legacy-recovery-index"
    state = RuntimeState.in_memory()
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        request = _request("legacy-index")
        admission = TaskGraphAdmission.from_request(request)
        legacy = TaskAdmissionRepositoryImpl(
            state.task.tasks.state_store,
            namespace=namespace,
            tenant_id="tenant",
        )
        await legacy.admit(admission, request.graph)

        recoverable = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )
        assert tuple(item.graph.graph_id for item in recoverable.items) == (
            request.graph.graph_id,
        )

        lease = await state.task.tasks.claim(
            request.graph.graph_id,
            "root",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=60,
        )
        running = await state.task.admissions.admit(admission, request.graph)
        assert running.status is TaskStatus.RUNNING

        await state.task.tasks.complete(
            lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="0" * 64,
        )
        terminal = await state.task.tasks.reconcile_graph(
            request.graph.graph_id,
            tenant_id="tenant",
        )
        recoverable = await state.task.admissions.list_recoverable_page(
            cursor=None,
            limit=128,
        )

        assert terminal.status is TaskStatus.SUCCEEDED
        assert recoverable.items == ()
    finally:
        await state.close()
