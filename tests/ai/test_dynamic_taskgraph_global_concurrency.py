#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distributed capacity invariants for durable TaskGraph claims."""

import asyncio

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskLease,
    TaskNode,
)
from linktools.ai.workspace import trusted_workspace_principal


@pytest.mark.asyncio
async def test_concurrent_claims_cannot_exceed_graph_capacity() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-global-claim-capacity", tenant_id="tenant")
    try:
        graph = TaskGraph("global-capacity", (TaskNode("a"), TaskNode("b")))
        request = TaskGraphRequest(
            graph,
            trusted_workspace_principal("tenant"),
            idempotency_key="global-capacity-submit-0001",
            limits=TaskGraphLimits(max_concurrency=1),
        )
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)

        async def claim(node_id: str, owner: str) -> TaskLease | ErrorCode:
            try:
                return await state.task.tasks.claim(
                    graph.graph_id,
                    node_id,
                    tenant_id="tenant",
                    owner=owner,
                    lease_seconds=30,
                )
            except AIError as error:
                return error.code

        first, second = await asyncio.gather(
            claim("a", "worker-a"),
            claim("b", "worker-b"),
        )

        results = (first, second)
        assert sum(isinstance(value, TaskLease) for value in results) == 1
        assert sum(value is ErrorCode.TASK_NOT_READY for value in results) == 1
    finally:
        await state.close()
