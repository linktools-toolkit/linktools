#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task recovery-index convergence contracts."""

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphRequest, TaskNode


@pytest.mark.asyncio
async def test_terminal_graph_is_removed_from_recovery_index_after_reconcile() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-recovery-index", tenant_id="tenant")
    try:
        request = TaskGraphRequest(
            TaskGraph("terminal-index", (TaskNode("root"),)),
            Principal("tester", "tenant"),
            "submit:terminal-index",
        )
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
