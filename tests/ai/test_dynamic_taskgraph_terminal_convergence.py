#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal convergence contracts for dynamic TaskGraph execution."""

import pytest

from ._task_test_helpers import admit_graph
from linktools.ai.core import TaskStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.task import TaskExpanderRef, TaskGraph, TaskNode


@pytest.mark.asyncio
async def test_failure_does_not_terminalize_graph_before_independent_expansion_settles() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-terminal-convergence", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "terminal-convergence",
            (
                TaskNode("failed"),
                TaskNode(
                    "expand",
                    expander=TaskExpanderRef("application.expand", 1),
                ),
            ),
        )
        await admit_graph(state, graph)
        failed_lease = await repository.claim(
            graph.graph_id,
            "failed",
            tenant_id="tenant",
            owner="failed-worker",
            lease_seconds=30,
        )
        expand_lease = await repository.claim(
            graph.graph_id,
            "expand",
            tenant_id="tenant",
            owner="expand-worker",
            lease_seconds=30,
        )

        await repository.fail(
            failed_lease,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="a" * 64,
        )
        active = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert active is not None
        assert active.status is TaskStatus.RUNNING

        await repository.complete(
            expand_lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="b" * 64,
            expanded_nodes=(TaskNode("child"),),
        )
        expanded = await repository.scheduler_snapshot(
            graph.graph_id,
            tenant_id="tenant",
        )
        states = {node.node_id: node.status for node in expanded.node_states}
        assert expanded.status is TaskStatus.PENDING
        assert states == {
            "child": TaskStatus.READY,
            "expand": TaskStatus.SUCCEEDED,
            "failed": TaskStatus.FAILED,
        }

        child_lease = await repository.claim(
            graph.graph_id,
            "child",
            tenant_id="tenant",
            owner="child-worker",
            lease_seconds=30,
        )
        await repository.complete(
            child_lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="c" * 64,
        )
        terminal = await repository.scheduler_snapshot(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert terminal.status is TaskStatus.FAILED
        assert all(
            node.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
                TaskStatus.CANCELLED,
            }
            for node in terminal.node_states
        )
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_waiting_node_keeps_graph_running_without_task_lease() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-waiting-aggregate", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("waiting-aggregate", (TaskNode("agent"),))
        await admit_graph(state, graph)
        lease = await repository.claim(
            graph.graph_id,
            "agent",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        waiting = await repository.handoff_execution(
            lease,
            tenant_id="tenant",
            execution_id="execution",
        )

        snapshot = await repository.scheduler_snapshot(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert waiting.status is TaskStatus.WAITING
        assert waiting.owner is None
        assert waiting.lease_expires_at is None
        assert snapshot.status is TaskStatus.RUNNING
        assert snapshot.node_states[0].status is TaskStatus.WAITING
    finally:
        await state.close()
