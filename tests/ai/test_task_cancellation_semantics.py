#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskGraph explicit cancellation semantics."""

import asyncio

import pytest

from ._task_test_helpers import admit_graph
from linktools.ai.core import TaskStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.task import TaskGraph, TaskNode


@pytest.mark.asyncio
async def test_cancel_preserves_terminal_nodes_and_unsettled_active_work() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-cancel-semantics", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "cancel-semantics",
            (
                TaskNode("failed"),
                TaskNode("blocked", ("failed",)),
                TaskNode("active"),
            ),
        )
        await admit_graph(state, graph)
        active = await repository.claim(
            graph.graph_id,
            "active",
            tenant_id="tenant",
            owner="active-worker",
            lease_seconds=30,
        )
        del active
        failed = await repository.claim(
            graph.graph_id,
            "failed",
            tenant_id="tenant",
            owner="failed-worker",
            lease_seconds=30,
        )
        await repository.fail(
            failed,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="a" * 64,
        )
        await repository.scheduler_state(graph.graph_id, tenant_id="tenant")

        cancelled = await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
        graph_state = await repository.graph_state(graph.graph_id, tenant_id="tenant")
        assert graph_state is not None
        states = {item.node_id: item.status for item in graph_state.node_states}

        assert cancelled.status is TaskStatus.RUNNING
        assert graph_state.status is TaskStatus.RUNNING
        assert states == {
            "failed": TaskStatus.FAILED,
            "blocked": TaskStatus.BLOCKED,
            "active": TaskStatus.RUNNING,
        }
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancel_does_not_terminalize_expired_claim_without_execution_binding() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="task-cancel-unbound-claim", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("cancel-unbound-claim", (TaskNode("node"),))
        await admit_graph(state, graph)
        lease = await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=1,
        )
        await asyncio.sleep(1.05)
        await repository.scheduler_state(graph.graph_id, tenant_id="tenant")

        cancelled = await repository.cancel_graph(
            graph.graph_id,
            tenant_id="tenant",
        )
        graph_state = await repository.graph_state(
            graph.graph_id,
            tenant_id="tenant",
        )

        assert cancelled.status is TaskStatus.RECOVERY_REQUIRED
        assert graph_state is not None
        node = graph_state.node_states[0]
        assert node.status is TaskStatus.RECOVERY_REQUIRED
        assert node.fence == lease.fence
        assert node.execution_id is None
        assert node.error_code == ErrorCode.EXECUTION_START_UNKNOWN.value
        assert node.safe_error_details == {
            "phase": "task_cancel",
            "reason": "claimed_execution_unbound",
        }
        recovered = await repository.recover_graph(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert recovered.status is TaskStatus.RECOVERY_REQUIRED
    finally:
        await state.close()
