#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for vendor-neutral Task lease and fencing semantics."""

import asyncio

import pytest

from ._task_test_helpers import admit_graph
from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.task import TaskGraph, TaskNode


@pytest.mark.asyncio
async def test_active_foreign_task_lease_cannot_be_stolen() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-lease-test", tenant_id="tenant")
    try:
        repository = state.task.tasks
        await admit_graph(
            state,
            TaskGraph("graph", (TaskNode("node"),)),
        )
        first = await repository.claim(
            "graph",
            "node",
            tenant_id="tenant",
            owner="owner-a",
            lease_seconds=10,
        )

        with pytest.raises(AIError) as raised:
            await repository.claim(
                "graph",
                "node",
                tenant_id="tenant",
                owner="owner-b",
                lease_seconds=10,
            )

        assert raised.value.code is ErrorCode.TASK_OWNER_CONFLICT
        nodes = await repository.list_nodes("graph", tenant_id="tenant")
        assert len(nodes) == 1
        assert nodes[0].status is TaskStatus.RUNNING
        assert nodes[0].owner == first.owner
        assert nodes[0].fence == first.fence
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_expired_task_lease_reclaim_fences_stale_terminal_write() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-reclaim-test", tenant_id="tenant")
    try:
        repository = state.task.tasks
        await admit_graph(
            state,
            TaskGraph("graph", (TaskNode("node"),)),
        )
        stale = await repository.claim(
            "graph",
            "node",
            tenant_id="tenant",
            owner="owner-a",
            lease_seconds=1,
        )
        await asyncio.sleep(1.05)

        current = await repository.claim(
            "graph",
            "node",
            tenant_id="tenant",
            owner="owner-b",
            lease_seconds=10,
        )
        assert current.owner == "owner-b"
        assert current.fence == stale.fence + 1

        result_digest = "a" * 64
        await repository.complete(
            current,
            tenant_id="tenant",
            execution_id="execution-current",
            result_digest=result_digest,
        )
        with pytest.raises(AIError) as raised:
            await repository.fail(
                stale,
                tenant_id="tenant",
                error_code=ErrorCode.TASK_NODE_FAILED.value,
                error_digest="b" * 64,
            )

        assert raised.value.code is ErrorCode.TASK_FENCE_STALE
        nodes = await repository.list_nodes("graph", tenant_id="tenant")
        assert len(nodes) == 1
        assert nodes[0].status is TaskStatus.SUCCEEDED
        assert nodes[0].owner is None
        assert nodes[0].execution_id == "execution-current"
        assert nodes[0].result_digest == result_digest
        assert nodes[0].fence == current.fence
    finally:
        await state.close()
