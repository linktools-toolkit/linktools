#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheduler reconciliation returns its applied projection without reloading it."""

from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.errors import ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import TaskGraph, TaskNode
from tests.ai._task_test_helpers import admit_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_scheduler_reuses_applied_projection(tmp_path: Path, failed: bool) -> None:
    async with Persistence.open(
        "scheduler-read-budget", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"),
        tenant_id="tenant",
    ) as state:
        repository = state.task.tasks
        graph = TaskGraph("graph", (TaskNode("first"), TaskNode("dependent", ("first",))))
        await admit_graph(state, graph)
        if failed:
            lease = await repository.claim(
                graph.graph_id, "first", tenant_id="tenant", owner="worker", lease_seconds=30,
            )
            await repository.fail(
                lease, tenant_id="tenant", error_code=ErrorCode.TASK_NODE_FAILED.value,
                error_digest="a" * 64,
            )
        for _ in range(2):
            record_reads: list[str] = []

            def before_sql(connection, cursor, statement, parameters, context, many):
                if statement.lstrip().upper().startswith("SELECT") and "FROM ai_state_records" in statement:
                    record_reads.append(statement)

            event.listen(Engine, "before_cursor_execute", before_sql)
            try:
                projected = await repository.scheduler_state(graph.graph_id, tenant_id="tenant")
            finally:
                event.remove(Engine, "before_cursor_execute", before_sql)
            assert projected == await repository.graph_state(graph.graph_id, tenant_id="tenant")
            # One graph read and one read each for node definitions and node state.
            assert len(record_reads) <= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [False, True])
async def test_task_mutation_batches_admission_read(tmp_path: Path, complete: bool) -> None:
    async with Persistence.open(
        "claim-read-budget", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"),
        tenant_id="tenant",
    ) as state:
        repository = state.task.tasks
        graph = TaskGraph("graph", (TaskNode("first"),))
        await admit_graph(state, graph)
        if complete:
            lease = await repository.claim(
                graph.graph_id, "first", tenant_id="tenant", owner="worker", lease_seconds=30,
            )
        record_reads: list[str] = []

        def before_sql(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "FROM ai_state_records" in statement:
                record_reads.append(statement)

        event.listen(Engine, "before_cursor_execute", before_sql)
        try:
            if complete:
                await repository.complete(
                    lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
                )
            else:
                lease = await repository.claim(
                    graph.graph_id, "first", tenant_id="tenant", owner="worker", lease_seconds=30,
                )
        finally:
            event.remove(Engine, "before_cursor_execute", before_sql)
        saved = await repository.graph_state(graph.graph_id, tenant_id="tenant")
        assert saved is not None
        assert saved.node_states[0].fence == lease.fence
        if complete:
            assert saved.node_states[0].result_digest == "a" * 64
        else:
            assert saved.node_states[0].owner == lease.owner
        # Graph, node and admission share one read; node lists remain separate.
        assert len(record_reads) <= 3
