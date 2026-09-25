#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskGraph explicit cancellation semantics."""

import pytest

from ._task_test_helpers import admit_graph
from linktools.ai.core import TaskStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.task import TaskGraph, TaskNode


@pytest.mark.asyncio
async def test_explicit_cancel_preserves_terminal_nodes_and_cancels_active_work() -> None:
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

        assert cancelled.status is TaskStatus.CANCELLED
        assert graph_state.status is TaskStatus.CANCELLED
        assert states == {
            "failed": TaskStatus.FAILED,
            "blocked": TaskStatus.BLOCKED,
            "active": TaskStatus.CANCELLED,
        }
    finally:
        await state.close()
