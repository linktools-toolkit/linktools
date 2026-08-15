#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable TaskGraph state-machine regression coverage."""

from pathlib import Path

import pytest
from linktools.ai.core import TaskStatus, canonical_sha256
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.task import TaskGraph, TaskNode


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_reconcile_propagates_transitive_blocked_state(
    backend: str,
    tmp_path: Path,
) -> None:
    state = (
        RuntimeState.in_memory()
        if backend == "memory"
        else RuntimeState.sqlite(tmp_path / "runtime.db")
    )
    await state.initialize(namespace=f"task-reconcile-{backend}", tenant_id="tenant")
    try:
        graph = TaskGraph(
            "graph",
            (
                TaskNode("a", ("b",), input={"kind": "test"}),
                TaskNode("b", ("c",), input={"kind": "test"}),
                TaskNode("c", input={"kind": "test"}),
            ),
        )
        repository = state.task.tasks
        await repository.create_graph(graph, tenant_id="tenant")
        await repository.reconcile_graph("graph", tenant_id="tenant")
        lease = await repository.claim(
            "graph",
            "c",
            tenant_id="tenant",
            owner="owner-c",
            lease_seconds=60,
        )
        await repository.fail(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.EXECUTION_FAILED.value,
            error_digest=canonical_sha256({"node": "c"}),
        )

        view = await repository.reconcile_graph("graph", tenant_id="tenant")

        assert view.status is TaskStatus.FAILED
        nodes = {
            node.node_id: node
            for node in await repository.list_nodes("graph", tenant_id="tenant")
        }
        assert nodes["b"].status is TaskStatus.BLOCKED
        assert nodes["a"].status is TaskStatus.BLOCKED
        assert nodes["a"].error_code == ErrorCode.TASK_DEPENDENCY_FAILED.value
    finally:
        await state.close()
